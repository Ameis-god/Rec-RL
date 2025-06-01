#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node 1 --master_port=25589\
         src/generate.py \
        --dataset lastfm \
        --batch_size 128 \
        --base_model '/your/path/to/model' \
        --tokenizer '/your/path/to/model' \
        --resume_from_checkpoint "/your/checkpoint/path"\
        --sft_prompt_path "../prompt/lastfm/music.txt" \
        --explain_prompt_path "../prompt/lastfm/explain.txt" \
        --wandb_project DeepRec-R1\
        --wandb_name Generate_Dataset > generate_dataset.log
        
	