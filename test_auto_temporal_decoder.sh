#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python test_auto_temporal_decoder.py \
    --model_id 'Jamichsu/Stream-DiffVSR' \
    --out_path './result' \
    --in_path '/project/REDS4/sharp_bicubic' \
    --num_inference_steps 4
