#!/bin/sh

MODEL_ID='claudiom4sir/StableVSR' # initialized from the weights of StableVSR, see https://huggingface.co/claudiom4sir/StableVSR for more details.
OUTPUT_DIR='./checkpoints/unet'
GPUS="0 1 2 3"

GPUS_STR=$(echo $GPUS | tr ' ' ',')

export CUDA_VISIBLE_DEVICES=$GPUS_STR
export TORCH_DISTRIBUTED_DEBUG=INFO

# Calculate the number of GPUs (i.e., the number of processes)
NUM_PROCESSES=$(echo $GPUS | wc -w)

accelerate launch --num_processes $NUM_PROCESSES --main_process_port 29502 train/train_4steps_unet_rollout.py \
 --pretrained_model_name_or_path=$MODEL_ID \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="REDS" \
 --dataset_config_path="dataset/config_reds.yaml" \
 --tiny_vae \
 --learning_rate=5e-5 \
 --lr_scheduler=constant_with_warmup \
 --lr_warmup_steps=1000 \
 --latent_mse_loss \
 --lpips_loss \
 --gan_loss \
 --lpips_loss_weight=0.1 \
 --gan_loss_weight=0.025 \
 --gan_loss_start_iter=1000 \
 --validation_steps=5000 \
 --checkpointing_steps=10 \
 --train_batch_size=2 \
 --dataloader_num_workers=8 \
 --max_train_steps=600000 \
 --enable_xformers_memory_efficient_attention \
 --validation_prompt "" \
 --validation_video_txt "dataset/reds/val_video_for_validation.txt" \
 --validation_video_GT_txt "dataset/reds/val_video_GT_for_validation.txt" \
 --train_video_txt "dataset/reds/train_video_for_validation.txt" \
 --train_video_GT_txt "dataset/reds/train_video_GT_for_validation.txt" \
