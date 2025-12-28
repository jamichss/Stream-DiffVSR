#
# Copyright 2024 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import PIL.Image
import tensorrt as trt
import torch
import torch.nn.functional as F
from cuda import cudart
from huggingface_hub import snapshot_download
from huggingface_hub.utils import validate_hf_hub_args
from onnx import shape_inference
from packaging import version
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.onnx.loader import fold_constants
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers import DiffusionPipeline, AutoencoderTiny, ControlNetModel
from diffusers.configuration_utils import FrozenDict, deprecate
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import logging

from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny, load_from_pretrained

from torchvision.models.optical_flow import raft_small, Raft_Large_Weights
import util.flow_utils as of

import onnxscript

# Assuming you use opset17
from onnxscript.onnx_opset import opset17 as op

custom_opset = onnxscript.values.Opset(domain="torch.onnx", version=1)


@onnxscript.script(custom_opset)
def ScaledDotProductAttention(
    query,
    key,
    value,
    dropout_p,
):
    # Swap the last two axes of key
    key_shape = op.Shape(key)
    key_last_dim = key_shape[-1:]
    key_second_last_dim = key_shape[-2:-1]
    key_first_dims = key_shape[:-2]
    # Contract the dimensions that are not the last two so we can transpose
    # with a static permutation.
    key_squeezed_shape = op.Concat(
        op.Constant(value_ints=[-1]), key_second_last_dim, key_last_dim, axis=0
    )
    key_squeezed = op.Reshape(key, key_squeezed_shape)
    key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
    key_transposed_shape = op.Concat(key_first_dims, key_last_dim, key_second_last_dim, axis=0)
    key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

    embedding_size = op.CastLike(op.Shape(query)[-1], query)
    scale = op.Div(1.0, op.Sqrt(embedding_size))

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = op.Mul(query, op.Sqrt(scale))
    key_transposed_scaled = op.Mul(key_transposed, op.Sqrt(scale))
    attn_weight = op.Softmax(
        op.MatMul(query_scaled, key_transposed_scaled),
        axis=-1,
    )
    attn_weight, _ = op.Dropout(attn_weight, dropout_p)
    return op.MatMul(attn_weight, value)


def custom_scaled_dot_product_attention(g, query, key, value, attn_mask, dropout, is_causal, scale=None):
    return g.onnxscript_op(ScaledDotProductAttention, query, key, value, dropout).setType(value.type())


torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::scaled_dot_product_attention",
    symbolic_fn=custom_scaled_dot_product_attention,
    opset_version=17,
)


"""
Installation instructions
python3 -m pip install --upgrade transformers diffusers>=0.16.0
python3 -m pip install --upgrade tensorrt~=10.2.0
python3 -m pip install --upgrade polygraphy>=0.47.0 onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
python3 -m pip install onnxruntime
"""

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}


def preprocess_image(image):
    """
    image: torch.Tensor
    """
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h))
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).contiguous()
    return 2.0 * image - 1.0


class Engine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()

    def __del__(self):
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray)]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_all_tactics=False,
        timing_cache=None,
    ):
        logger.warning(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        extra_build_args = {}
        if not enable_all_tactics:
            extra_build_args["tactic_sources"] = []

        engine = engine_from_network(
            network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
            config=CreateConfig(fp16=fp16, profiles=[p], load_timing_cache=timing_cache, **extra_build_args),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        logger.warning(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self):
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[name] = tensor

    def infer(self, feed_dict, stream):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")

        return self.tensors


class Optimizer:
    def __init__(self, onnx_graph):
        self.graph = gs.import_onnx(onnx_graph)

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph


class BaseModel:
    def __init__(self, model, fp16=False, device="cuda", max_batch_size=16, embedding_dim=768, text_maxlen=77):
        self.model = model
        self.name = "SD Model"
        self.fp16 = fp16
        self.device = device

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 2048  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 4
        self.max_latent_shape = self.max_image_shape // 4

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self):
        return self.model

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 4 == 0 or image_width % 4 == 0
        ########################
        latent_height = image_height // 4
        latent_width = image_width // 4
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 4
        latent_width = image_width // 4
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )


def getOnnxPath(model_name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, model_name + (".opt" if opt else "") + ".onnx")


def getEnginePath(model_name, engine_dir):
    return os.path.join(engine_dir, model_name + ".plan")


def build_engines(
    models: dict,
    engine_dir,
    onnx_dir,
    onnx_opset,
    opt_image_height,
    opt_image_width,
    opt_batch_size=1,
    force_engine_rebuild=False,
    static_batch=False,
    static_shape=True,
    enable_all_tactics=False,
    timing_cache=None,
):
    built_engines = {}
    if not os.path.isdir(onnx_dir):
        os.makedirs(onnx_dir)
    if not os.path.isdir(engine_dir):
        os.makedirs(engine_dir)

    # Export models to ONNX
    for model_name, model_obj in models.items():
        engine_path = getEnginePath(model_name, engine_dir)
        if force_engine_rebuild or not os.path.exists(engine_path):
            logger.warning("Building Engines...")
            logger.warning("Engine build can take a while to complete")
            onnx_path = getOnnxPath(model_name, onnx_dir, opt=False)
            onnx_opt_path = getOnnxPath(model_name, onnx_dir)
            if force_engine_rebuild or not os.path.exists(onnx_opt_path):
                if force_engine_rebuild or not os.path.exists(onnx_path):
                    logger.warning(f"Exporting model: {onnx_path}")
                    model = model_obj.get_model()
                    with torch.inference_mode(), torch.autocast("cuda"):
                        inputs = model_obj.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
                        torch.onnx.export(
                            model,
                            inputs,
                            onnx_path,
                            export_params=True,
                            opset_version=onnx_opset,
                            do_constant_folding=True,
                            input_names=model_obj.get_input_names(),
                            output_names=model_obj.get_output_names(),
                            dynamic_axes=model_obj.get_dynamic_axes(),
                        )
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    logger.warning(f"Found cached model: {onnx_path}")

                # Optimize onnx
                if force_engine_rebuild or not os.path.exists(onnx_opt_path):
                    logger.warning(f"Generating optimizing model: {onnx_opt_path}")
                    onnx_opt_graph = model_obj.optimize(onnx.load(onnx_path))
                    onnx.save(onnx_opt_graph, onnx_opt_path)
                else:
                    logger.warning(f"Found cached optimized model: {onnx_opt_path} ")

    # Build TensorRT engines
    for model_name, model_obj in models.items():
        engine_path = getEnginePath(model_name, engine_dir)
        engine = Engine(engine_path)
        onnx_path = getOnnxPath(model_name, onnx_dir, opt=False)
        onnx_opt_path = getOnnxPath(model_name, onnx_dir)

        if force_engine_rebuild or not os.path.exists(engine.engine_path):
            engine.build(
                onnx_opt_path,
                fp16=True,
                input_profile=model_obj.get_input_profile(
                    opt_batch_size,
                    opt_image_height,
                    opt_image_width,
                    static_batch=static_batch,
                    static_shape=static_shape,
                ),
                timing_cache=timing_cache,
            )
        built_engines[model_name] = engine

    # Load and activate TensorRT engines
    for model_name, model_obj in models.items():
        engine = built_engines[model_name]
        engine.load()
        engine.activate()

    return built_engines


def runEngine(engine, feed_dict, stream):
    return engine.infer(feed_dict, stream)


class RAFTWrapper(BaseModel):
    def __init__(self, model, device, max_batch_size, embedding_dim):
        super(RAFTWrapper, self).__init__(
            model=model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim
        )
        self.name = "RAFT"

    def get_input_names(self):
        return ["image1", "image2"]

    def get_output_names(self):
        return ["flow"]

    def get_dynamic_axes(self):
        return {
            "image1": {0: "B", 2: "H", 3: "W"},
            "image2": {0: "B", 2: "H", 3: "W"},
            "flow":   {0: "B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return {
            "image1": [
                (1, 3, image_height, image_width),
                (1, 3, image_height, image_width),
                (1, 3, image_height, image_width),
            ],
            "image2": [
                (1, 3, image_height, image_width),
                (1, 3, image_height, image_width),
                (1, 3, image_height, image_width),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        return {
            "image1": (1, 3, image_height, image_width),
            "image2": (1, 3, image_height, image_width),
            "flow": (1, 2, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        return (
            torch.randn(1, 3, image_height, image_width, dtype=torch.float32, device=self.device),
            torch.randn(1, 3, image_height, image_width, dtype=torch.float32, device=self.device),
        )

def make_RAFT(model, device, max_batch_size, embedding_dim=768, **kwargs):
    return RAFTWrapper(model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)


class CLIP(BaseModel):
    def __init__(self, model, device, max_batch_size, embedding_dim):
        super(CLIP, self).__init__(
            model=model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim
        )
        self.name = "CLIP"

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings", "pooler_output"]

    def get_dynamic_axes(self):
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        return {
            "input_ids": [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.select_outputs([0])  # delete graph output#1
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        opt.select_outputs([0], names=["text_embeddings"])  # rename network output
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        return opt_onnx_graph


def make_CLIP(model, device, max_batch_size, embedding_dim, inpaint=False, upscale=True):
    return CLIP(model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)


class UNet(BaseModel):
    def __init__(
        self, model, fp16=False, device="cuda", max_batch_size=16, embedding_dim=768, text_maxlen=77, unet_dim=4
    ):
        super(UNet, self).__init__(
            model=model,
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.name = "UNet"

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states", "mid_block_additional_residual"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "2B"},
            "mid_block_additional_residual": {0: "2B", 2: "mid_H", 3: "mid_W"},
            "latent": {0: "2B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "sample": [
                (2 * min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (2 * batch_size, self.unet_dim, latent_height, latent_width),
                (2 * max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "encoder_hidden_states": [
                (2 * min_batch, self.text_maxlen, self.embedding_dim),
                (2 * batch_size, self.text_maxlen, self.embedding_dim),
                (2 * max_batch, self.text_maxlen, self.embedding_dim),
            ],
            "mid_block_additional_residual": [
                (2 * min_batch, 1024, min_latent_height // 8, min_latent_width // 8),
                (2 * batch_size, 1024, latent_height // 8, latent_width // 8),
                (2 * max_batch, 1024, max_latent_height // 8, max_latent_width // 8),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "mid_block_additional_residual": (2 * batch_size, 1024, latent_height // 8, latent_width // 8),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        sample = torch.randn(2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device)
        timestep = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        encoder_hidden_states = torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device)
        mid_block_additional_residual = torch.randn(2 * batch_size, 1024, latent_height // 8, latent_width // 8, dtype=dtype, device=self.device)
        return (
            sample,
            timestep,
            encoder_hidden_states,
            mid_block_additional_residual,
        )

    def forward(self, sample, timestep, encoder_hidden_states, mid_block_additional_residual):
        output = self.model(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            mid_block_additional_residual=mid_block_additional_residual,
        )

        return output


def make_UNet(model, device, max_batch_size, embedding_dim, inpaint=False, upscale=True):
    return UNet(
        model,
        fp16=False,
        device=device,
        max_batch_size=max_batch_size,
        embedding_dim=embedding_dim,
        unet_dim=(7 if upscale else 4),
    )

class ControlNet(BaseModel):
    def __init__(
        self, model, fp16=False, device="cuda", max_batch_size=16, embedding_dim=768, text_maxlen=77, controlnet_dim=4
    ):
        super(ControlNet, self).__init__(
            model=model,
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.controlnet_dim = controlnet_dim
        self.name = "ControlNet"

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states", "controlnet_cond"]

    def get_output_names(self):
        return [f"down_block_res_sample_{i}" for i in range(12)] + ["mid_block_res_sample"]

    def get_dynamic_axes(self):
        return {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "2B"},
            "controlnet_cond": {0: "2B", 2: "image_H", 3: "image_W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "sample": [
                (2 * min_batch, self.controlnet_dim, min_latent_height, min_latent_width),
                (2 * batch_size, self.controlnet_dim, latent_height, latent_width),
                (2 * max_batch, self.controlnet_dim, max_latent_height, max_latent_width),
            ],
            "encoder_hidden_states": [
                (2 * min_batch, self.text_maxlen, self.embedding_dim),
                (2 * batch_size, self.text_maxlen, self.embedding_dim),
                (2 * max_batch, self.text_maxlen, self.embedding_dim),
            ],
            'controlnet_cond': [
                (2 * min_batch, 3, min_image_height, min_image_width),
                (2 * batch_size, 3, image_height, image_width),
                (2 * max_batch, 3, max_image_height, max_image_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.controlnet_dim, latent_height, latent_width),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "controlnet_cond": (2 * batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(2 * batch_size, self.controlnet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device), # sample
            torch.tensor([1.0], dtype=torch.float32, device=self.device), # timestep
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device), # encoder_hidden_states
            torch.randn(2 * batch_size, 3, image_height, image_width, dtype=torch.float32, device=self.device), # controlnet_cond
        )

def make_ControlNet(model, device, max_batch_size, embedding_dim, inpaint=False, upscale=True):
    return ControlNet(
        model,
        fp16=False,
        device=device,
        max_batch_size=max_batch_size,
        embedding_dim=embedding_dim,
        controlnet_dim=(7 if upscale else 4),
    )

class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(
        self,
        unet,
        controlnet: ControlNetModel,
    ):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        controlnet_cond,
    ):
        if torch.any(controlnet_cond != 0):
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                return_dict=False,
            )
        else:
            down_block_res_samples, mid_block_res_sample = None, None

        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]
        return noise_pred

class UNet2DConditionControlNet(BaseModel):
    def __init__(
        self, model, fp16=False, device="cuda", max_batch_size=16, embedding_dim=768, text_maxlen=77, input_dim=4
    ):
        super(UNet2DConditionControlNet, self).__init__(
            model=model,
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.input_dim = input_dim
        self.name = "UNet2DConditionControlNet"

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states", "controlnet_cond"]

    def get_output_names(self):
        return ["noise_pred"]

    def get_dynamic_axes(self):
        return {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "2B"},
            "controlnet_cond": {0: "2B", 2: "image_H", 3: "image_W"},
            "noise_pred": {0: "2B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "sample": [
                (2 * min_batch, self.input_dim, min_latent_height, min_latent_width),
                (2 * batch_size, self.input_dim, latent_height, latent_width),
                (2 * max_batch, self.input_dim, max_latent_height, max_latent_width),
            ],
            "encoder_hidden_states": [
                (2 * min_batch, self.text_maxlen, self.embedding_dim),
                (2 * batch_size, self.text_maxlen, self.embedding_dim),
                (2 * max_batch, self.text_maxlen, self.embedding_dim),
            ],
            'controlnet_cond': [
                (2 * min_batch, 3, min_image_height, min_image_width),
                (2 * batch_size, 3, image_height, image_width),
                (2 * max_batch, 3, max_image_height, max_image_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.input_dim, latent_height, latent_width),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "controlnet_cond": (2 * batch_size, 3, image_height, image_width),
            "noise_pred": (2 * batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        sample = torch.randn(2 * batch_size, self.input_dim, latent_height, latent_width, dtype=dtype, device=self.device)
        timestep = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        encoder_hidden_states = torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device)
        controlnet_cond = torch.randn(2 * batch_size, 3, image_height, image_width, dtype=torch.float32, device=self.device)
        return (
            sample,
            timestep,
            encoder_hidden_states,
            controlnet_cond,
        )


def make_UNet2DConditionControlNet(model, device, max_batch_size, embedding_dim, inpaint=False, upscale=True):
    return UNet2DConditionControlNet(
        model,
        fp16=False,
        device=device,
        max_batch_size=max_batch_size,
        embedding_dim=embedding_dim,
        input_dim=(7 if upscale else 4),
    )


class VAE(BaseModel):
    def __init__(self, model, device, max_batch_size, embedding_dim):
        super(VAE, self).__init__(
            model=model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim
        )
        self.name = "VAE decoder"

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {"latent": {0: "B", 2: "H", 3: "W"}, "images": {0: "B", 2: "4H", 3: "4W"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)


def make_VAE(model, device, max_batch_size, embedding_dim, inpaint=False, upscale=True):
    return VAE(model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vae_encoder = model

    def forward(self, x):
        return retrieve_latents(self.vae_encoder.encode(x))


class VAEEncoder(BaseModel):
    def __init__(self, model, device, max_batch_size, embedding_dim):
        super(VAEEncoder, self).__init__(
            model=model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim
        )
        self.name = "VAE encoder"

    def get_model(self):
        vae_encoder = TorchVAEEncoder(self.model)
        return vae_encoder

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {"images": {0: "B", 2: "4H", 3: "4W"}, "latent": {0: "B", 2: "H", 3: "W"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            "images": [
                (min_batch, 3, min_image_height, min_image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_image_height, max_image_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "images": (batch_size, 3, image_height, image_width),
            "latent": (batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32, device=self.device)

def make_VAEEncoder(model, device, max_batch_size, embedding_dim, inpaint=False, upscale=True):
    return VAEEncoder(model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)

class TemporalVAE(BaseModel):
    def __init__(self, model, device, max_batch_size, embedding_dim):
        super(TemporalVAE, self).__init__(
            model=model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim
        )
        self.name = "Temporal VAE"

    def get_input_names(self):
        return ["current_latent", "prev_rgb"]

    def get_output_names(self):
        return ["current_rgb"]

    def get_dynamic_axes(self):
        return {"current_latent":{0: "B", 2: "H", 3: "W"}, "prev_rgb": {0: "B", 2: "4H", 3: "4W"}, "current_rgb": {0: "B", 2: "4H", 3: "4W"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            "current_latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ],
            "prev_rgb": [
                (min_batch, 3, min_image_height, min_image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_image_height, max_image_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "current_latent": (batch_size, 4, latent_height, latent_width),
            "prev_rgb": (batch_size, 3, image_height, image_width),
            "current_rgb": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        prev_rgb = torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32, device=self.device)
        current_latent = torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)
        return (
            current_latent,
            prev_rgb,
        )

def make_TemporalVAE(model, device, max_batch_size, embedding_dim, inpaint=False, upscale=True):
    return TemporalVAE(model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)



class TensorRTStableDiffusionImg2ImgPipeline(DiffusionPipeline):
    r"""
    Pipeline for image-to-image generation using TensorRT accelerated Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]

    def __init__(
        self,
        vae: [AutoencoderTiny, AutoencoderKL],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: DDIMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
        stages=["clip", "UNet2DConditionControlNet", "TemporalVAE", "raft"],
        image_height: int = 720,
        image_width: int = 1280,
        max_batch_size: int = 16,
        # ONNX export parameters
        onnx_opset: int = 17,
        onnx_dir: str = "onnx",
        # TensorRT engine build parameters
        engine_dir: str = "engine",
        force_engine_rebuild: bool = False,
        timing_cache: str = "timing_cache",
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )

        self.stages = stages
        self.image_height, self.image_width = image_height, image_width
        self.inpaint = False
        self.upscale = True
        self.onnx_opset = onnx_opset
        self.onnx_dir = onnx_dir
        self.engine_dir = engine_dir
        self.force_engine_rebuild = force_engine_rebuild
        self.timing_cache = timing_cache
        self.build_static_batch = False
        self.build_dynamic_shape = False

        self.max_batch_size = max_batch_size
        # TODO: Restrict batch size to 4 for larger image dimensions as a WAR for TensorRT limitation.
        if self.build_dynamic_shape or self.image_height > 512 or self.image_width > 512:
            self.max_batch_size = 4

        self.stream = None  # loaded in loadResources()
        self.models = {}  # loaded in __loadModels()
        self.engine = {}  # loaded in build_engines()

        # self.vae.forward = self.vae.decode
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        self.control_unet = UNet2DConditionControlNetModel(self.unet, self.controlnet)

        self.of_model = raft_small(pretrained=True, progress=False).eval().cuda()
        self.of_model = self.of_model.half()

    def __loadModels(self):
        # Load pipeline models
        self.embedding_dim = self.text_encoder.config.hidden_size
        models_args = {
            "device": self.torch_device,
            "max_batch_size": self.max_batch_size,
            "embedding_dim": self.embedding_dim,
            "inpaint": self.inpaint,
            "upscale": self.upscale,
        }
        if "UNet2DConditionControlNet" in self.stages:
            self.models["UNet2DConditionControlNet"] = make_UNet2DConditionControlNet(
                self.control_unet, **models_args
            )
        if "controlnet" in self.stages:
            self.models["controlnet"] = make_ControlNet(self.controlnet, **models_args)
        if "unet" in self.stages:
            self.models["unet"] = make_UNet(self.unet, **models_args)
        if "clip" in self.stages:
            self.models["clip"] = make_CLIP(self.text_encoder, **models_args)
        if "vae" in self.stages:
            self.models["vae"] = make_VAE(self.vae, **models_args)
        if "vae_encoder" in self.stages:
            self.models["vae_encoder"] = make_VAEEncoder(self.vae, **models_args)
        if "TemporalVAE" in self.stages:
            self.models["TemporalVAE"] = make_TemporalVAE(self.vae, **models_args)
        if "raft" in self.stages:
            self.models["raft"] = make_RAFT(self.of_model, **models_args)

    def get_flow_tensorrt(self, engine, image1, image2, stream):
        image1 = image1.cuda() # shape: (1, 3, H, W)
        image2 = image2.cuda()
        inputs = {"image1": image1, "image2": image2}
        flow = runEngine(engine, inputs, stream)["flow"] # shape: (1, 2, H, W)
        return flow


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(
        self, image: Union[torch.Tensor, PIL.Image.Image], device: torch.device, dtype: torch.dtype
    ) -> Tuple[Union[torch.Tensor, PIL.Image.Image], Optional[bool]]:
        r"""
        Runs the safety checker on the given image.
        Args:
            image (Union[torch.Tensor, PIL.Image.Image]): The input image to be checked.
            device (torch.device): The device to run the safety checker on.
            dtype (torch.dtype): The data type of the input image.
        Returns:
            (image, has_nsfw_concept) Tuple[Union[torch.Tensor, PIL.Image.Image], Optional[bool]]: A tuple containing the processed image and
            a boolean indicating whether the image has a NSFW (Not Safe for Work) concept.
        """
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    @classmethod
    @validate_hf_hub_args
    def set_cached_folder(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        cls.cached_folder = (
            pretrained_model_name_or_path
            if os.path.isdir(pretrained_model_name_or_path)
            else snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
        )

    def to(self, torch_device: Optional[Union[str, torch.device]] = None, silence_dtype_warnings: bool = False):
        super().to(torch_device, silence_dtype_warnings=silence_dtype_warnings)

        self.onnx_dir = os.path.join(self.cached_folder, self.onnx_dir)
        self.engine_dir = os.path.join(self.cached_folder, self.engine_dir)
        self.timing_cache = os.path.join(self.cached_folder, self.timing_cache)

        # set device
        self.torch_device = self._execution_device
        logger.warning(f"Running inference on device: {self.torch_device}")

        # load models
        self.__loadModels()

        # build engines
        self.engine = build_engines(
            self.models,
            self.engine_dir,
            self.onnx_dir,
            self.onnx_opset,
            opt_image_height=self.image_height,
            opt_image_width=self.image_width,
            force_engine_rebuild=self.force_engine_rebuild,
            static_batch=self.build_static_batch,
            static_shape=not self.build_dynamic_shape,
            timing_cache=self.timing_cache,
        )

        return self

    def __initialize_timesteps(self, timesteps, strength):
        self.scheduler.set_timesteps(timesteps)
        offset = self.scheduler.steps_offset if hasattr(self.scheduler, "steps_offset") else 0
        init_timestep = int(timesteps * strength) + offset
        init_timestep = min(init_timestep, timesteps)
        timesteps = self.scheduler.timesteps.to(self.torch_device)
        return timesteps

    def __preprocess_images(self, batch_size, images=()):
        init_images = []
        for image in images:
            image = image.to(self.torch_device).float()
            image = image.repeat(batch_size, 1, 1, 1)
            init_images.append(image)
        return tuple(init_images)

    def __encode_image(self, init_image):
        init_latents = runEngine(self.engine["vae_encoder"], {"images": init_image}, self.stream)["latent"]
        init_latents = 0.18215 * init_latents
        return init_latents

    def __encode_prompt(self, prompt, negative_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
        """
        # Tokenize prompt
        text_input_ids = (
            self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.type(torch.int32)
            .to(self.torch_device)
        )

        # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
        text_embeddings = runEngine(self.engine["clip"], {"input_ids": text_input_ids}, self.stream)[
            "text_embeddings"
        ].clone()

        # Tokenize negative prompt
        uncond_input_ids = (
            self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.type(torch.int32)
            .to(self.torch_device)
        )
        uncond_embeddings = runEngine(self.engine["clip"], {"input_ids": uncond_input_ids}, self.stream)[
            "text_embeddings"
        ]

        # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float32)

        return text_embeddings

    def __denoise_latent(
        self, image, latents, warped_prev_est, text_embeddings, timesteps=None, step_offset=0
    ):
        do_classifier_free_guidance = self._guidance_scale > 1
        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps

        for step_index, timestep in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
            latent_model_input = torch.cat([latent_model_input, image], dim=1)

            # Predict the noise residual
            timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep
            noise_pred = runEngine(
                    self.engine["UNet2DConditionControlNet"],
                    {
                        "sample": latent_model_input,
                        "timestep": timestep_float,
                        "encoder_hidden_states": text_embeddings,
                        "controlnet_cond": warped_prev_est,
                    },
                    self.stream,
                )["noise_pred"][0]

            # if warped_prev_est is not None:
            #     noise_pred = runEngine(
            #         self.engine["UNet2DConditionControlNetModel"],
            #         {
            #             "sample": latent_model_input,
            #             "timestep": timestep_float,
            #             "encoder_hidden_states": text_embeddings,
            #             "controlnet_cond": warped_prev_est,
            #         },
            #         self.stream,
            #     )["noise_pred"][0]

            # else:
            #     noise_pred = runEngine(
            #         self.engine["unet"],
            #         {"sample": latent_model_input, "timestep": timestep_float, "encoder_hidden_states": text_embeddings},
            #         self.stream,
            #     )["latent"][0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample

        return latents

    def __decode_latent(self, latents):
        images = runEngine(self.engine["vae"], {"latent": latents}, self.stream)["images"]
        # images = (images / 2 + 0.5).clamp(0, 1)
        # return images.permute(0, 2, 3, 1).float()
        return images

    def __loadResources(self, image_height, image_width, batch_size):
        self.stream = cudart.cudaStreamCreate()[1]

        # Allocate buffers for TensorRT engine bindings
        for model_name, obj in self.models.items():
            print(f"Allocating buffers for {model_name}")
            self.engine[model_name].allocate_buffers(
                shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=self.torch_device
            )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        # shape = (batch_size, num_channels_latents, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def compute_flows(self, of_model, images, rescale_factor=1):
        print('Computing forward flows...')
        forward_flows = []
        for i in range(1, len(images)):
            prev_image = images[i - 1]
            cur_image = images[i]
            fflow = of.get_flow(of_model, cur_image, prev_image, rescale_factor=rescale_factor)
            forward_flows.append(fflow)
        return forward_flows

    def compute_flows_tensorrt(self, engine, images, stream):
        print('Computing forward flows with tensorrt...')
        forward_flows = []
        for i in range(1, len(images)):
            prev_image = images[i - 1]
            cur_image = images[i]
            fflow = self.get_flow_tensorrt(engine, cur_image, prev_image, stream)
            forward_flows.append(fflow)
        return forward_flows

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        images: Union[torch.Tensor, PIL.Image.Image, List] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        of_model = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.

        """
        self.generator = generator
        self.denoising_steps = num_inference_steps
        self._guidance_scale = guidance_scale

        # Pre-compute latent input scales and linear multistep coefficients
        self.scheduler.set_timesteps(self.denoising_steps, device=self.torch_device)

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"Expected prompt to be of type list or str but got {type(prompt)}")

        if negative_prompt is None:
            negative_prompt = [""] * batch_size

        if negative_prompt is not None and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        assert len(prompt) == len(negative_prompt)

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(prompt)} is larger than allowed {self.max_batch_size}. If dynamic shape is used, then maximum batch size is 4"
            )

        # load resources
        self.__loadResources(self.image_height, self.image_width, batch_size)

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            # Initialize timesteps
            timesteps = self.__initialize_timesteps(self.denoising_steps, strength)
            latent_timestep = timesteps[:1].repeat(batch_size)

            # Pre-process input image
            if isinstance(images, PIL.Image.Image):
                images = preprocess_image(images)
            # init_image = self.__preprocess_images(batch_size, (image,))[0]
            init_images =  [self.image_processor.preprocess(image, height=self.image_height//4, width=self.image_width//4).to(dtype=torch.float32).to(self.torch_device) for image in images]
            upscaled_images = [F.interpolate(image, scale_factor=4, mode='bicubic') for image in init_images]
            height, width = upscaled_images[0].shape[-2:]

            # VAE encode init image
            # init_latents = self.__encode_image(init_images)


            # Add noise to latents using timesteps
            # noise = torch.randn(
            #     init_latents.shape, generator=self.generator, device=self.torch_device, dtype=torch.float32
            # )
            # latents = self.scheduler.add_noise(init_latents, noise, latent_timestep)

            # CLIP text encoder
            text_embeddings = self.__encode_prompt(prompt, negative_prompt)

            # Prepare latent variables
            num_channels_latents = self.vae.config.latent_channels
            latents = [self.prepare_latents(
                        batch_size,
                        num_channels_latents,
                        height,
                        width,
                        text_embeddings.dtype,
                        self.torch_device,
                        self.generator
                    ) for _ in range(len(init_images))]
            
            # compute flows
            forward_flows = self.compute_flows(of_model, upscaled_images, rescale_factor=1.0)
            # forward_flows = self.compute_flows_tensorrt(self.engine["raft"], upscaled_images, self.stream)
            
            for num_image, data in enumerate(init_images): 
                image = data

                if num_image > 0:
                    warped_prev_est = of.flow_warp(warp_to_next, forward_flows[num_image - 1], interp_mode="bilinear") # warp it to current frame
                else:
                    warped_prev_est = False

                # UNet denoiser
                latents[num_image] = self.__denoise_latent(image, latents[num_image], warped_prev_est, text_embeddings, timesteps=timesteps)

                # VAE decode latent
                # image = self.__decode_latent(latents[num_image])
                image = runEngine(
                    self.engine["TemporalVAE"],
                    {
                        "current_latent": latents[num_image],
                        "prev_rgb": warped_prev_est,
                    },
                    self.stream,
                )["current_rgb"]

                warp_to_next = image

                has_nsfw_concept = None
                if has_nsfw_concept is None:
                    do_denormalize = [True] * image[0].shape[0]
                else:
                    do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

                image = self.image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)
                images[num_image] = image

        # images, has_nsfw_concept = self.run_safety_checker(images, self.torch_device, text_embeddings.dtype)
        # images = self.numpy_to_pil(images)
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)