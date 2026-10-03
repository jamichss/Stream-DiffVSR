#!/bin/sh

MODEL_ID='Jamichsu/Stream-DiffVSR'
OUTPUT_DIR='./checkpoints/artg'
GPUS="0 1 2 3"

GPUS_STR=$(echo $GPUS | tr ' ' ',')

export CUDA_VISIBLE_DEVICES=$GPUS_STR

# Calculate the number of GPUs (i.e., the number of processes)
NUM_PROCESSES=$(echo $GPUS | wc -w)

accelerate launch --num_processes $NUM_PROCESSES --main_process_port 29502 ./train/train_artg.py \
 --pretrained_model_name_or_path=$MODEL_ID \
 --pretrained_unet_name_or_path="YOUR_PATH_TO_UNET" \
 --temporal_vae_config_path="pretrained-model/taesd-x4/config.json" \
 --temporal_vae_pretrained_weight_path="/YOUR_PATH_TO_TEMPORAL_VAE/diffusion_pytorch_model.safetensors" \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="REDS" \
 --dataset_config_path="dataset/config_reds.yaml" \
 --learning_rate=5e-5 \
 --lr_scheduler=constant_with_warmup \
 --lr_warmup_steps=1000 \
 --latent_mse_loss \
 --lpips_loss \
 --lpips_loss_weight=0.1 \
 --gan_loss \
 --gan_loss_weight=0.025 \
 --gan_loss_start_iter=2000 \
 --validation_steps=5000 \
 --checkpointing_steps=10000 \
 --train_batch_size=2 \
 --dataloader_num_workers=8 \
 --max_train_steps=60000 \
 --enable_xformers_memory_efficient_attention \
 --validation_prompt "" \
 --validation_video_txt "dataset/val_video_for_validation.txt" \
 --validation_video_GT_txt "dataset/val_video_GT_for_validation.txt" \
 --train_video_txt "dataset/train_video_for_validation.txt" \
 --train_video_GT_txt "dataset/train_video_GT_for_validation.txt" \

 