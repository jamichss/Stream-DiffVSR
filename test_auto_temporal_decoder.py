import os
import sys
import argparse
import time
from pathlib import Path
import torch
from accelerate.utils import set_seed
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from pipeline.stream_diffvsr_pipeline import StreamDiffVSRPipeline, ControlNetModel, UNet2DConditionModel
from diffusers import DDIMScheduler
from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny

torch.backends.cuda.matmul.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser(description="Test code for StableVSR.")
    parser.add_argument("--model_id", default='stabilityai/stable-diffusion-x4-upscaler', type=str, help="model_id of the model to be tested.")
    parser.add_argument("--unet_pretrained_weight", type=str, help="UNet pretrained weight.")
    parser.add_argument("--controlnet_pretrained_weight", type=str, help="ControlNet pretrained weight.")
    parser.add_argument("--temporal_vae_pretrained_weight", type=str, help="Path to Temporal VAE.")
    parser.add_argument("--out_path", default='./StreamDiffVSR_results/', type=str, help="Path to output folder.")
    parser.add_argument("--in_path", type=str, required=True, help="Path to input folder (containing sets of LR images).")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of sampling steps")
    parser.add_argument("--enable_tensorrt", action='store_true', help="Enable TensorRT. Note that the performance will drop if TensorRT is enabled.")
    parser.add_argument("--image_height", type=int, default=720, help="Height of the output images. Needed for TensorRT.")
    parser.add_argument("--image_width", type=int, default=1280, help="Width of the output images. Needed for TensorRT.")
    return parser.parse_args()

def load_component(cls, weight_path, model_id, subfolder):
    path = weight_path if weight_path else model_id
    sub = None if weight_path else subfolder
    return cls.from_pretrained(path, subfolder=sub)

def main():
    args = parse_args()

    print("Run with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    set_seed(42)
    device = torch.device('cuda')

    controlnet = load_component(ControlNetModel, args.controlnet_pretrained_weight, args.model_id, "controlnet")
    unet = load_component(UNet2DConditionModel, args.unet_pretrained_weight, args.model_id, "unet")
    vae = load_component(TemporalAutoencoderTiny, args.temporal_vae_pretrained_weight, args.model_id, "vae")
    scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    tensorrt_kwargs = {
        "custom_pipeline": "/acceleration/tensorrt/sd_with_controlnet_ST",
        "image_height": args.image_height,
        "image_width": args.image_width,
    } if args.enable_tensorrt else {"custom_pipeline": None}
    
    pipeline = StreamDiffVSRPipeline.from_pretrained(
        args.model_id,
        controlnet=controlnet, 
        vae=vae, 
        unet=unet, 
        scheduler=scheduler,
        **tensorrt_kwargs
    )

    if args.enable_tensorrt:
        pipeline.set_cached_folder("Jamichsu/Stream-DiffVSR")

    pipeline = pipeline.to(device)
    pipeline.enable_xformers_memory_efficient_attention()
    
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    of_model.requires_grad_(False) 
    
    seqs = sorted(os.listdir(args.in_path))
    for seq in seqs:
        seq_path = os.path.join(args.in_path, seq)
        frame_names = sorted(os.listdir(seq_path))
        frames = []
        for frame_name in frame_names:
            frame_path = os.path.join(seq_path, frame_name)
            frames.append(Image.open(frame_path))

        output = pipeline(
            '', frames, 
            num_inference_steps=args.num_inference_steps, 
            guidance_scale=0, 
            of_model=of_model
        )
        frames_hr = output.images
        frames_to_save = [frame[0] for frame in frames_hr]
        
        seq_path_obj = Path(seq_path)
        target_path = os.path.join(args.out_path, seq_path_obj.parent.name, seq_path_obj.name)
        os.makedirs(target_path, exist_ok=True)
        
        for frame, name in zip(frames_to_save, frame_names):
            frame.save(os.path.join(target_path, name))
        
        print(f"Upscaled {seq} and saved to {target_path}.")
        
        del frames
        del frames_to_save
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()