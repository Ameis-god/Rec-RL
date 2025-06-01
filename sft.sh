#!/bin/bash
torchrun --nproc_per_node 4 --master_port=25642 sft.py \
        --model_name "Qwen2.5-7B-Instruct"  \
        --batch_size 8 \
        --gradient_accumulation_steps 8 \
        --dataset lastfm \
        --logging_dir '/your/path/to/log' \
        --output_dir '/your/path/to/output' \
        --learning_rate 1e-5 \
        --num_train_epochs 2 \
        --eval_step 0.2 \
        --wandb_project DeepRec-R1 \
        --wandb_name sft >sft.log
