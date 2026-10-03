#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import math
import os
import sys
import random
import shutil
from pathlib import Path
from omegaconf import OmegaConf
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from dataset.reds_dataset import REDSRecurrentDataset
from dataset.vimeo90k_dataset import Vimeo90KRecurrentDataset

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from torchvision.transforms import ToTensor
tt = ToTensor()
from einops import rearrange
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from util.flow_utils import get_flow, flow_warp

import diffusers
from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    ControlNetModel,
    UNet2DConditionModel
)

from pipeline.stream_diffvsr_pipeline import StreamDiffVSRPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

# from diffusers import DDIMScheduler
from scheduler.ddpm_scheduler import DDPMScheduler
from scheduler.ddim_scheduler import DDIMScheduler

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from gan_loss_utils.diffusion_gan import Discriminator, NLayerDiscriminator, calculate_adaptive_weight, adopt_weight, weights_init

from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny, load_from_pretrained


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step, of_model):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StreamDiffVSRPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )

    pipeline.scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", beta_schedule="scaled_linear", timestep_spacing='trailing')
    pipeline = pipeline.to(accelerator.device)
    # pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_video_txt) == len(args.validation_prompt):
        train_images = args.train_video_txt
        train_images_GT = args.train_video_GT_txt
        validation_images = args.validation_video_txt
        validation_images_GT = args.validation_video_GT_txt
        validation_prompt = args.validation_prompt
    elif len(args.validation_video_txt) == 1:
        train_images = args.train_video_txt * len(args.validation_prompt)
        train_images_GT = args.train_video_GT_txt * len(args.validation_prompt)
        validation_images = args.validation_video_txt * len(args.validation_prompt)
        validation_images_GT = args.validation_video_GT_txt * len(args.validation_prompt)
        validation_prompt = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        train_images = args.train_video_txt
        train_images_GT = args.train_video_GT_txt
        validation_images = args.validation_video_txt
        validation_images_GT = args.validation_video_GT_txt
        validation_prompt = args.validation_prompt * len(args.validation_video_txt)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    train_image_logs = []
    validation_image_logs = []

    lpips = LPIPS(normalize=True).to(accelerator.device)
    psnr = PSNR(data_range=1).to(accelerator.device)
    ssim = SSIM(data_range=1).to(accelerator.device)
    
    def read_video_frames(file_path):
        with open(file_path, 'r') as f:
            return [[Image.open(os.path.join(video.split()[0], frame)).convert("RGB") for frame in sorted(os.listdir(video.split()[0])) if frame.endswith('.png')][:int(video.split()[1])] for video in f.read().splitlines()]

    validation_videos = read_video_frames(args.validation_video_txt)
    train_videos = read_video_frames(args.train_video_txt)
    validation_videos_GT = read_video_frames(args.validation_video_GT_txt)
    train_videos_GT = read_video_frames(args.train_video_GT_txt)

    train_psnr_avg, validation_psnr_avg = 0, 0
    train_ssim_avg, validation_ssim_avg = 0, 0
    train_lpips_avg, validation_lpips_avg = 0, 0

    for train_video, train_video_GT, validation_video, validation_video_GT in zip(train_videos, train_videos_GT, validation_videos, validation_videos_GT):
        if train_video is not None:
            train_frames = [frame for frame in train_video]
        if train_video_GT is not None:
            train_frames_GT = [frame for frame in train_video_GT]
        if validation_video is not None:
            validation_frames = [frame for frame in validation_video]
        if validation_video_GT is not None:
            validation_frames_GT = [frame for frame in validation_video_GT]

        with torch.no_grad():
            for _ in range(args.num_validation_images):
                # with torch.autocast("cuda"):
                train_video_out = pipeline(
                    validation_prompt[0], train_frames, num_inference_steps=4, generator=generator, guidance_scale=0, of_model=of_model
                ).images
                validation_video_out = pipeline(
                    validation_prompt[0], validation_frames, num_inference_steps=4, generator=generator, guidance_scale=0, of_model=of_model
                ).images

            train_images_out = [x[0] for x in train_video_out]
            validation_images_out = [x[0] for x in validation_video_out]

            # Calculate PSNR, SSIM, and LPIPS

            n = len(validation_images_out)
            for i in range(n):
                train_images_out_tensor = tt(train_images_out[i]).unsqueeze(0).to(accelerator.device)
                validation_images_out_tensor = tt(validation_images_out[i]).unsqueeze(0).to(accelerator.device)
                train_frames_GT_tensor = tt(train_frames_GT[i]).unsqueeze(0).to(accelerator.device)
                validation_frames_GT_tensor = tt(validation_frames_GT[i]).unsqueeze(0).to(accelerator.device)

                train_psnr_avg += psnr(train_images_out_tensor, train_frames_GT_tensor).item()
                validation_psnr_avg += psnr(validation_images_out_tensor, validation_frames_GT_tensor).item()

                train_ssim_avg += ssim(train_images_out_tensor, train_frames_GT_tensor).item() 
                validation_ssim_avg += ssim(validation_images_out_tensor, validation_frames_GT_tensor).item()

                train_lpips_avg += lpips(train_frames_GT_tensor, train_images_out_tensor).item()
                validation_lpips_avg += lpips(validation_frames_GT_tensor, validation_images_out_tensor).item()

    train_image_logs.append(
        {"train_images_out": train_images_out[:4]}
        )
    validation_image_logs.append(
        {"validation_images_out": validation_images_out[:4]}
    )

    n = len(validation_video) * len(validation_videos)
    train_psnr_avg /= n
    validation_psnr_avg /= n
    train_ssim_avg /= n
    validation_ssim_avg /= n
    train_lpips_avg /= n
    validation_lpips_avg /= n

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in train_image_logs:
                images = log["train_images_out"]

                formatted_images = []

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.concatenate(formatted_images, axis=1)
                formatted_images = np.expand_dims(formatted_images, 0)

                tracker.writer.add_images("Train", formatted_images, step, dataformats="NHWC")
                tracker.writer.add_scalars('PSNR',{'train':train_psnr_avg, 'validation':validation_psnr_avg}, step)
                tracker.writer.add_scalars('SSIM',{'train':train_ssim_avg, 'validation':validation_ssim_avg}, step)
                tracker.writer.add_scalars('LPIPS',{'train':train_lpips_avg, 'validation':validation_lpips_avg}, step)

            for log in validation_image_logs:
                images = log["validation_images_out"]

                formatted_images = []

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.concatenate(formatted_images, axis=1)
                formatted_images = np.expand_dims(formatted_images, 0)

                tracker.writer.add_images("Validation", formatted_images, step, dataformats="NHWC")

        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_video_txt = log["validation_video_txt"]

                formatted_images.append(wandb.Image(validation_video_txt, caption="Unet output"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return train_image_logs, validation_image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):

    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_video_txt = log["validation_video_txt"]
            validation_video_txt.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_video_txt] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='stabilityai/stable-diffusion-2-1',
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--temporal_vae_config_path",
        type=str,
        required=True,
        default='./pretrained-model/taesd-x4/config.json',
        help="The path to the temporal vae model config.",
    )
    parser.add_argument(
        "--temporal_vae_pretrained_weight_path",
        type=str,
        required=True,
        default=None,
        help="The path to the pretrained temporal vae model.",
    )
    parser.add_argument(
        "--pretrained_unet_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model or model identifier from huggingface.co/models."
        " If not specified unet weights are initialized from the pretrained model.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--latent_mse_loss",
        action="store_true",
        help="Enable lpips loss for training.",
    )
    parser.add_argument(
        "--latent_mse_loss_weight",
        type=float,
        default=1,
        help="Weight for latent mse loss.",
    )
    parser.add_argument(
        "--charbonnier_loss",
        action="store_true",
        help="Enable lpips loss for training.",
    )
    parser.add_argument(
        "--charbonnier_loss_weight",
        type=float,
        default=0.1,
        help="Weight for charbonnier loss.",
    )
    parser.add_argument(
        "--lpips_loss",
        action="store_true",
        help="Enable lpips loss for training.",
    )
    parser.add_argument(
        "--lpips_loss_weight",
        type=float,
        default=0.1,
        help="Weight for lpips loss.",
    )
    parser.add_argument(
        "--gan_loss",
        action="store_true",
        help="Enable GAN loss for training.",
    )
    parser.add_argument(
        "--gan_loss_weight",
        type=float,
        default=0.025,
        help="Weight for GAN loss.",
    )
    parser.add_argument(
        "--gan_loss_start_iter",
        type=float,
        default=20000,
        help="First iter to enable the gan_loss.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_path",
        type=str,
        default=None,
        help=(
            "The path to the config file related to the dataset."
        ),
    )    
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_video_txt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_video_GT_txt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train_video_txt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_train_images",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--train_video_GT_txt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None and args.dataset_config_path is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir` or `dataset_config_path`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_video_txt is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_video_txt is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_video_txt is not None
        and args.validation_prompt is not None
        and len(args.validation_video_txt) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_video_txt) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
        diffusers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_warning()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    ########################################
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, num_train_timesteps=1000, subfolder="scheduler", beta_schedule="scaled_linear", timestep_spacing="trailing", prediction_type="v_prediction")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    vae = load_from_pretrained(
        config_path=args.temporal_vae_config_path,
        pretrained_path=args.temporal_vae_pretrained_weight_path,
    )
    # vae_tae = AutoencoderTiny.from_pretrained("madebyollin/taesd-x4-upscaler", torch_dtype=torch.float32)
    if args.pretrained_unet_name_or_path:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_unet_name_or_path, subfolder="unet", revision=args.revision
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(64,128,256,)) 

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "controlnet"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    of_model.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(True)
    controlnet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    if args.gan_loss:
        sr_discriminator = NLayerDiscriminator(input_nc=6).apply(weights_init).to(accelerator.device)
        sr_discriminator.requires_grad_(False)
        params_to_optimize_d = sr_discriminator.parameters()
        d_optimizer = optimizer_class(
            params_to_optimize_d,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon
        )


    # train_dataset = make_train_dataset(args, tokenizer, accelerator)
    if args.dataset_name == "REDS":
        dataset_opts = OmegaConf.load(args.dataset_config_path)
        train_dataset = REDSRecurrentDataset(dataset_opts['dataset']['train'])
    elif args.dataset_name == "Vimeo90K":
        dataset_opts = OmegaConf.load(args.dataset_config_path)
        train_dataset = Vimeo90KRecurrentDataset(dataset_opts['dataset']['train'])
    else:
        raise ValueError(f"Dataset {args.dataset_name} is not supported. We only support REDS and Vimeo90K currently.")


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        # collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    of_model.to(accelerator.device, dtype=weight_dtype)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("train_video_txt")
        tracker_config.pop("train_video_GT_txt")
        tracker_config.pop("validation_video_txt")
        tracker_config.pop("validation_video_GT_txt")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    train_image_logs = None
    validation_image_logs = None

    noise_scheduler.set_timesteps(4)

    if args.lpips_loss:
        lpips = LPIPS(normalize=True).to(accelerator.device)

    # get here input condition since it is fixed
    with torch.no_grad():
        tokenization = tokenizer([''] * args.train_batch_size, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        encoder_hidden_states = text_encoder(tokenization.input_ids.to(accelerator.device))[0]
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Prepare images
                lq = batch['lq'] 
                gt = batch['gt']
                gt = 2 * gt - 1
                lq = 2 * lq - 1
                b, t, _, _, _ = lq.shape
                upscaled_lq = rearrange(lq, 'b t c h w -> (b t) c h w')
                upscaled_lq = F.interpolate(upscaled_lq, scale_factor=4, mode='bicubic')
                upscaled_lq = rearrange(upscaled_lq, '(b t) c h w -> b t c h w', b=b, t=t)

                random_t = [round(random.random()) * 2 for _ in range(b)] # <- decide t-1 or t+1
                gt_prev = torch.stack([gt[i, t] for i, t in enumerate(random_t)])
                upscaled_lq_prev = torch.stack([upscaled_lq[i, t] for i, t in enumerate(random_t)])
                lq_prev = torch.stack([lq[i, t] for i, t in enumerate(random_t)])

                gt = gt[:, t // 2, ...]
                lq = lq[:, t // 2, ...]
                upscaled_lq_cur = upscaled_lq[:, t // 2, ...]

                # Convert images to latent space
                vae.reset_temporal_condition()
                latents = vae.encode(gt.to(dtype=weight_dtype), return_layers_features=False).latents
                latents_prev = vae.encode(gt_prev.to(dtype=weight_dtype), return_layers_features=False).latents
                
                latents = latents * vae.config.scaling_factor
                latents_prev = latents_prev * vae.config.scaling_factor

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.full((bsz,), 999, device=latents.device, dtype=torch.long)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                noise_prev = torch.randn_like(latents_prev)
                noisy_latents_prev = noise_scheduler.add_noise(latents_prev, noise_prev, timesteps)

                # noisy_latents_cat = torch.cat([noisy_latents, lq], dim=1)

                # Get the text embedding for conditioning
                # tokenization = tokenizer('', max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                # encoder_hidden_states = text_encoder(tokenization)[0]


                # make prediction of the previous frame
                # noisy_latents_prev_cat = torch.cat([noisy_latents_prev, lq_prev], dim=1)
                # noise_level = torch.cat([torch.tensor([20], dtype=torch.long, device=accelerator.device)] * b)
                for t in [999, 749, 499, 249]:
                    timesteps = torch.full((bsz,), t, device=latents.device, dtype=torch.long)
                    noisy_latents_prev_cat = torch.cat([noisy_latents_prev, lq_prev], dim=1)

                    model_pred_prev = unet(
                        noisy_latents_prev_cat,
                        timesteps,
                        # class_labels = noise_level,
                        encoder_hidden_states=encoder_hidden_states.detach()
                    ).sample
                    out = noise_scheduler.step(model_pred_prev, t, noisy_latents_prev)
                    noisy_latents_prev, model_pred_x0_prev = out.prev_sample, out.pred_original_sample

                    # latents_prev_warped = compute_of_and_warp(of_model, upscaled_lq_cur, upscaled_lq_prev, latents_prev)
                    # controlnet_image = latents_prev_warped.to(dtype=weight_dtype)
                model_pred_x0_prev_rgb = vae.decode(model_pred_x0_prev / vae.config.scaling_factor).sample
                f_flow = get_flow(of_model, upscaled_lq_cur, upscaled_lq_prev)
                warped_approximated_x0 = flow_warp(model_pred_x0_prev_rgb, f_flow)
                controlnet_image = warped_approximated_x0.to(dtype=weight_dtype).detach()
                
                tem, enc_layer_features = vae.encode(warped_approximated_x0.to(dtype=weight_dtype), return_dict=False, return_layers_features=True)

                dec_temporal_features = enc_layer_features[::-1]

                for t in [999, 749, 499, 249]:
                    timesteps = torch.full((bsz,), t, device=latents.device, dtype=torch.long)
                    noisy_latents_cat = torch.cat([noisy_latents, lq], dim=1)

                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents_cat,
                        timesteps,
                        # class_labels = noise_level,
                        encoder_hidden_states=encoder_hidden_states.detach(),
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )
                    
                    # Predict the noise residual
                    model_pred = unet(
                        noisy_latents_cat,
                        timesteps,
                        # class_labels = noise_level,
                        encoder_hidden_states=encoder_hidden_states.detach(),
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    ).sample

                    # Get the target for loss depending on the prediction type and loss type

                    out = noise_scheduler.step(model_pred, t, noisy_latents)
                    noisy_latents, model_pred_x0 = out.prev_sample, out.pred_original_sample
                
                if args.latent_mse_loss:
                    target = {
                        "epsilon": noise,
                        "v_prediction": noise_scheduler.get_velocity(latents, noise, timesteps),
                        "x_prediction": latents
                    }.get(noise_scheduler.prediction_type, None)
                    
                    if target is None:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")
                    
                    if noise_scheduler.prediction_type == "v_prediction":
                        latent_mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        latent_mse_loss = F.mse_loss(model_pred_x0.float(), target.float(), reduction="mean")
                    w_latent_mse_loss = latent_mse_loss * args.latent_mse_loss_weight
                else:
                    latent_mse_loss = 0

                rgb_predicted = (vae.decode(model_pred_x0 / vae.config.scaling_factor, temporal_features=dec_temporal_features).sample).clamp(-1, 1)
                rgb_predicted_norm = (rgb_predicted + 1) / 2
                gt_norm = (gt + 1) / 2

                if args.charbonnier_loss:
                    pixel_loss = charbonnier_loss(rgb_predicted_norm, gt_norm)
                    w_pixel_loss = pixel_loss * args.charbonnier_loss_weight

                else:
                    w_pixel_loss = 0

                if args.lpips_loss:
                    lpips_loss = lpips(rgb_predicted_norm, gt_norm)
                    w_lpips_loss = lpips_loss * args.lpips_loss_weight
                else:
                    w_lpips_loss = 0
                
                reconstruction_loss = latent_mse_loss + w_pixel_loss + w_lpips_loss
                
                if args.gan_loss and step % 5 == 0:
                    # train the discriminator
                    d_optimizer.zero_grad()

                    noise_strength = 0.05
                    noisy_gt = (gt_norm + torch.randn_like(gt_norm) * noise_strength).clamp(0, 1)

                    # real_logits, real_features = sr_discriminator(noisy_gt, upscaled_lq_cur)
                    # fake_logits, fake_features = sr_discriminator(rgb_predicted_norm, upscaled_lq_cur)
                    
                    # adversieral loss for controlnet
                    logits_fake = sr_discriminator(torch.cat((rgb_predicted_norm, upscaled_lq_cur), dim=1))

                    gan_loss = F.softplus(-logits_fake).mean()

                    d_weight = calculate_adaptive_weight(reconstruction_loss, gan_loss, last_layer=controlnet.module.controlnet_mid_block.weight, discriminator_weight=args.gan_loss_weight)
                    disc_factor = adopt_weight(args.gan_loss_weight, global_step, threshold=args.gan_loss_start_iter)

                    w_gan_loss = gan_loss * d_weight * disc_factor
                    
                    controlnet.requires_grad_(False)
                    sr_discriminator.requires_grad_(True)
                    sr_discriminator.train()

                    logits_real = sr_discriminator(torch.cat((noisy_gt, upscaled_lq_cur), dim=1))
                    logits_fake = sr_discriminator(torch.cat((rgb_predicted_norm, upscaled_lq_cur), dim=1))


                    d_loss = F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean()
                    # accelerator.backward(d_loss)
                    # d_optimizer.step()

                    if d_loss.item() > 0.5:
                        accelerator.backward(d_loss, retain_graph=True)
                        d_optimizer.step()

                    controlnet.requires_grad_(True)
                    sr_discriminator.requires_grad_(False)
                    sr_discriminator.eval()

                else:
                    w_gan_loss = 0
                

                loss = w_gan_loss + reconstruction_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            of_model
                        )

            logs = {"loss": loss.detach().item(),
                    "d_loss": d_loss.detach().item(),
                    "lpips_loss": lpips_loss.detach().item(),
                    "latent_mse_loss": latent_mse_loss.detach().item(),
                    "gan_loss":gan_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
