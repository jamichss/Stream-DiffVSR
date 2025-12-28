# Copyright 2024 Ollin Boer Bohan and The HuggingFace Team. All rights reserved.
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
# limitations under the License.


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from safetensors.torch import load_file

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_utils import ModelMixin
from .vae import DecoderOutput, TemporalDecoderTiny, EncoderTiny
from .models.unets.unet_2d_blocks import TemporalAutoencoderTinyBlock


@dataclass
class TemporalAutoencoderTinyOutput(BaseOutput):
    """
    Output of TemporalAutoencoderTiny encoding method.

    Args:
        latents (`torch.Tensor`): Encoded outputs of the `Encoder`.

    """

    latents: torch.Tensor


class TemporalAutoencoderTiny(ModelMixin, ConfigMixin):
    r"""
    A tiny distilled VAE model for encoding images into latents and decoding latent representations into images.

    [`AutoencoderTiny`] is a wrapper around the original implementation of `TAESD`.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for its generic methods implemented for
    all models (such as downloading or saving).

    Parameters:
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (`int`,  *optional*, defaults to 3): Number of channels in the output.
        encoder_block_out_channels (`Tuple[int]`, *optional*, defaults to `(64, 64, 64, 64)`):
            Tuple of integers representing the number of output channels for each encoder block. The length of the
            tuple should be equal to the number of encoder blocks.
        decoder_block_out_channels (`Tuple[int]`, *optional*, defaults to `(64, 64, 64, 64)`):
            Tuple of integers representing the number of output channels for each decoder block. The length of the
            tuple should be equal to the number of decoder blocks.
        act_fn (`str`, *optional*, defaults to `"relu"`):
            Activation function to be used throughout the model.
        latent_channels (`int`, *optional*, defaults to 4):
            Number of channels in the latent representation. The latent space acts as a compressed representation of
            the input image.
        upsampling_scaling_factor (`int`, *optional*, defaults to 2):
            Scaling factor for upsampling in the decoder. It determines the size of the output image during the
            upsampling process.
        num_encoder_blocks (`Tuple[int]`, *optional*, defaults to `(1, 3, 3, 3)`):
            Tuple of integers representing the number of encoder blocks at each stage of the encoding process. The
            length of the tuple should be equal to the number of stages in the encoder. Each stage has a different
            number of encoder blocks.
        num_decoder_blocks (`Tuple[int]`, *optional*, defaults to `(3, 3, 3, 1)`):
            Tuple of integers representing the number of decoder blocks at each stage of the decoding process. The
            length of the tuple should be equal to the number of stages in the decoder. Each stage has a different
            number of decoder blocks.
        latent_magnitude (`float`, *optional*, defaults to 3.0):
            Magnitude of the latent representation. This parameter scales the latent representation values to control
            the extent of information preservation.
        latent_shift (float, *optional*, defaults to 0.5):
            Shift applied to the latent representation. This parameter controls the center of the latent space.
        scaling_factor (`float`, *optional*, defaults to 1.0):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper. For this Autoencoder,
            however, no such scaling factor was used, hence the value of 1.0 as the default.
        force_upcast (`bool`, *optional*, default to `False`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without losing too much precision, in which case
            `force_upcast` can be set to `False` (see this fp16-friendly
            [AutoEncoder](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        encoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        act_fn: str = "relu",
        upsample_fn: str = "nearest",
        latent_channels: int = 4,
        upsampling_scaling_factor: int = 2,
        num_encoder_blocks: Tuple[int, ...] = (1, 3, 3, 3),
        num_decoder_blocks: Tuple[int, ...] = (3, 3, 3, 1),
        latent_magnitude: int = 3,
        latent_shift: float = 0.5,
        force_upcast: bool = False,
        scaling_factor: float = 1.0,
        shift_factor: float = 0.0,
    ):
        super().__init__()

        if len(encoder_block_out_channels) != len(num_encoder_blocks):
            raise ValueError("`encoder_block_out_channels` should have the same length as `num_encoder_blocks`.")
        if len(decoder_block_out_channels) != len(num_decoder_blocks):
            raise ValueError("`decoder_block_out_channels` should have the same length as `num_decoder_blocks`.")

        self.encoder = EncoderTiny(
            in_channels=in_channels,
            out_channels=latent_channels,
            num_blocks=num_encoder_blocks,
            block_out_channels=encoder_block_out_channels,
            act_fn=act_fn,
        )

        self.encoder.requires_grad_(False)

        self.decoder = TemporalDecoderTiny(
            in_channels=latent_channels,
            out_channels=out_channels,
            num_blocks=num_decoder_blocks,
            block_out_channels=decoder_block_out_channels,
            upsampling_scaling_factor=upsampling_scaling_factor,
            act_fn=act_fn,
            upsample_fn=upsample_fn,
        )

        self.decoder.requires_grad_(False)

        for name, param in self.decoder.named_parameters():
            if "alpha" in name or "temporal_processor" in name:
                param.requires_grad_(True)

        self.latent_magnitude = latent_magnitude
        self.latent_shift = latent_shift
        self.scaling_factor = scaling_factor

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.spatial_scale_factor = 2**out_channels
        self.tile_overlap_factor = 0.125
        self.tile_sample_min_size = 512
        self.tile_latent_min_size = self.tile_sample_min_size // self.spatial_scale_factor

        self.register_to_config(block_out_channels=decoder_block_out_channels)
        self.register_to_config(force_upcast=False)

    def reset_temporal_condition(self):
        """reset temporal memory"""
        for module in self.encoder.layers:
            if isinstance(module, TemporalAutoencoderTinyBlock):
                module.reset_temporal()
        for module in self.decoder.layers:
            if isinstance(module, TemporalAutoencoderTinyBlock):
                module.reset_temporal()

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if isinstance(module, (EncoderTiny, TemporalDecoderTiny)):
            module.gradient_checkpointing = value

    def scale_latents(self, x: torch.Tensor) -> torch.Tensor:
        """raw latents -> [0, 1]"""
        return x.div(2 * self.latent_magnitude).add(self.latent_shift).clamp(0, 1)

    def unscale_latents(self, x: torch.Tensor) -> torch.Tensor:
        """[0, 1] -> raw latents"""
        return x.sub(self.latent_shift).mul(2 * self.latent_magnitude)

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def enable_tiling(self, use_tiling: bool = True) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = use_tiling

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.enable_tiling(False)

    def _tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output.

        Args:
            x (`torch.Tensor`): Input batch of images.

        Returns:
            `torch.Tensor`: Encoded batch of images.
        """
        # scale of encoder output relative to input
        sf = self.spatial_scale_factor
        tile_size = self.tile_sample_min_size

        # number of pixels to blend and to traverse between tile
        blend_size = int(tile_size * self.tile_overlap_factor)
        traverse_size = tile_size - blend_size

        # tiles index (up/left)
        ti = range(0, x.shape[-2], traverse_size)
        tj = range(0, x.shape[-1], traverse_size)

        # mask for blending
        blend_masks = torch.stack(
            torch.meshgrid([torch.arange(tile_size / sf) / (blend_size / sf - 1)] * 2, indexing="ij")
        )
        blend_masks = blend_masks.clamp(0, 1).to(x.device)

        # output array
        out = torch.zeros(x.shape[0], 4, x.shape[-2] // sf, x.shape[-1] // sf, device=x.device)
        for i in ti:
            for j in tj:
                tile_in = x[..., i : i + tile_size, j : j + tile_size]
                # tile result
                tile_out = out[..., i // sf : (i + tile_size) // sf, j // sf : (j + tile_size) // sf]
                tile = self.encoder(tile_in)
                h, w = tile.shape[-2], tile.shape[-1]
                # blend tile result into output
                blend_mask_i = torch.ones_like(blend_masks[0]) if i == 0 else blend_masks[0]
                blend_mask_j = torch.ones_like(blend_masks[1]) if j == 0 else blend_masks[1]
                blend_mask = blend_mask_i * blend_mask_j
                tile, blend_mask = tile[..., :h, :w], blend_mask[..., :h, :w]
                tile_out.copy_(blend_mask * tile + (1 - blend_mask) * tile_out)
        return out

    def _tiled_decode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output.

        Args:
            x (`torch.Tensor`): Input batch of images.

        Returns:
            `torch.Tensor`: Encoded batch of images.
        """
        # scale of decoder output relative to input
        sf = self.spatial_scale_factor
        tile_size = self.tile_latent_min_size

        # number of pixels to blend and to traverse between tiles
        blend_size = int(tile_size * self.tile_overlap_factor)
        traverse_size = tile_size - blend_size

        # tiles index (up/left)
        ti = range(0, x.shape[-2], traverse_size)
        tj = range(0, x.shape[-1], traverse_size)

        # mask for blending
        blend_masks = torch.stack(
            torch.meshgrid([torch.arange(tile_size * sf) / (blend_size * sf - 1)] * 2, indexing="ij")
        )
        blend_masks = blend_masks.clamp(0, 1).to(x.device)

        # output array
        out = torch.zeros(x.shape[0], 3, x.shape[-2] * sf, x.shape[-1] * sf, device=x.device)
        for i in ti:
            for j in tj:
                tile_in = x[..., i : i + tile_size, j : j + tile_size]
                # tile result
                tile_out = out[..., i * sf : (i + tile_size) * sf, j * sf : (j + tile_size) * sf]
                tile = self.decoder(tile_in)
                h, w = tile.shape[-2], tile.shape[-1]
                # blend tile result into output
                blend_mask_i = torch.ones_like(blend_masks[0]) if i == 0 else blend_masks[0]
                blend_mask_j = torch.ones_like(blend_masks[1]) if j == 0 else blend_masks[1]
                blend_mask = (blend_mask_i * blend_mask_j)[..., :h, :w]
                tile_out.copy_(blend_mask * tile + (1 - blend_mask) * tile_out)
        return out

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True, return_layers_features: bool = True, return_features_only: bool = False) -> Union[TemporalAutoencoderTinyOutput, Tuple[torch.Tensor]]:
        layer_features = [] if return_layers_features else None  # 只有啟用時才存儲特徵

        if self.use_slicing and x.shape[0] > 1:
            output = [
                self._tiled_encode(x_slice) if self.use_tiling else self.encoder(x_slice)
                for x_slice in x.split(1)
            ]
            output = torch.cat(output)
        else:
            if self.use_tiling:
                output = self._tiled_encode(x)
            elif return_layers_features:
                current_features = x
                for module in self.encoder.layers:
                    current_features = module(current_features)

                    if isinstance(module, TemporalAutoencoderTinyBlock):
                        layer_features.append(current_features)

                if return_features_only:
                    return layer_features

            output = self.encoder(x) # output = current_features will lead to error calculation, we will fix it later.

        # self.reset_temporal_condition()

        if not return_dict:
            return (output,), layer_features

        return TemporalAutoencoderTinyOutput(latents=output)


    @apply_forward_hook
    def decode(
        self, x: torch.Tensor, temporal_features = None, generator: Optional[torch.Generator] = None, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        if self.use_slicing and x.shape[0] > 1:
            output = [
                self._tiled_decode(x_slice) if self.use_tiling else self.decoder(x_slice) for x_slice in x.split(1)
            ]
            output = torch.cat(output)
        elif temporal_features is not None:
            block_idx = 0
            for module in self.decoder.layers:
                if isinstance(module, TemporalAutoencoderTinyBlock):
                    module.prev_features = temporal_features[block_idx]
                    block_idx += 1
            output = self.decoder(x)
        else:
            # self.reset_temporal_condition()
            output = self._tiled_decode(x) if self.use_tiling else self.decoder(x)

        # self.reset_temporal_condition()

        if not return_dict:
            return (output,)

        return DecoderOutput(sample=output)

    def forward(
        self,
        sample: torch.Tensor,
        previous_sample: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        layer_features = None

        if previous_sample is None:
            # enc, _ = self.encode(sample, return_dict=return_dict)
            pass
        else:
            prev_enc, layer_features = self.encode(previous_sample, return_dict=return_dict)

        # scale latents to be in [0, 1], then quantize latents to a byte tensor,
        # as if we were storing the latents in an RGBA uint8 image.
        # scaled_enc = self.scale_latents(enc).mul_(255).round_().byte()

        if layer_features is not None:
            temporal_features = layer_features[::-1]
        else:
            temporal_features = None

        # unquantize latents back into [0, 1], then unscale latents back to their original range,
        # as if we were loading the latents from an RGBA uint8 image.
        # unscaled_enc = self.unscale_latents(scaled_enc / 255.0, temporal_features=temporal_features)

        dec = self.decode(sample, temporal_features=temporal_features, return_dict=return_dict)[0]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

def load_from_pretrained(config_path, pretrained_path, **kwargs):
    # config_path = "/home_nfs/jamichsu/StableVSR/hf-model/taesd-x4/config.json"
    model = TemporalAutoencoderTiny.from_config(config_path)

    # load pretrained image AE weights
    if pretrained_path is not None:
        print("Loading VAE from", pretrained_path)
        missing_keys, unexpected_keys = model.load_state_dict(load_file(pretrained_path, device="cpu"), strict=False)

    # initialize missing weights
    for name in missing_keys:
        print(f"Initializing weights for missing layer: {name}")
        module_name = name.rsplit('.', 1)[0]
        module = dict(model.named_modules()).get(module_name)
        if module is not None:
            print(f"Initializing weights for missing layer: {module_name}")
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    print("Model loaded successfully.")
    return model