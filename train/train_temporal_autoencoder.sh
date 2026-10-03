#!/bin/sh

MODEL_ID='madebyollin/taesd-x4-upscaler'
OUTPUT_DIR='./checkpoints/temporalAE'
GPUS="0 1 2 3"

GPUS_STR=$(echo $GPUS | tr ' ' ',')

export CUDA_VISIBLE_DEVICES=$GPUS_STR

# Calculate the number of GPUs (i.e., the number of processes)
NUM_PROCESSES=$(echo $GPUS | wc -w)

accelerate launch --num_processes $NUM_PROCESSES --main_process_port 29502 train/train_temporal_autoencoder.py \
 --pretrained_vae_name_or_path=$MODEL_ID \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="REDS" \
 --temporal_vae_config_path="pretrained-model/taesd-x4/config.json" \
 --temporal_vae_pretrained_weight_path="pretrained-model/taesd-x4/diffusion_pytorch_model.safetensors" \
 --dataset_config_path="dataset/config_reds.yaml" \
 --learning_rate=5e-5 \
 --lr_scheduler=constant_with_warmup \
 --lr_warmup_steps=1000 \
 --pixel_loss \
 --lpips_loss \
 --flow_loss \
 --gan_loss \
 --lpips_loss_weight=0.1 \
 --flow_loss_weight=0.1 \
 --flow_loss_start_iter=20000 \
 --gan_loss_weight=0.025 \
 --gan_loss_start_iter=20000 \
 --validation_steps=10000 \
 --checkpointing_steps=10000 \
 --train_batch_size=8 \
 --dataloader_num_workers=8 \
 --max_train_steps=10000000 \
 --enable_xformers_memory_efficient_attention \
 --validation_prompt "" \
 --validation_video_txt "dataset/reds/val_video_GT_for_validation.txt" \
 --validation_video_GT_txt "dataset/reds/val_video_GT_for_validation.txt" \
 --train_video_txt "dataset/reds/train_video_GT_for_validation.txt" \
 --train_video_GT_txt "dataset/reds/train_video_GT_for_validation.txt" \

 