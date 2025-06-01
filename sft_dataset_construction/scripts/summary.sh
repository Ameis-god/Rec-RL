#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node 1 --master_port=25599\
         src/summary.py \
        --dataset lastfm \
        --batch_size 128 \
        --base_model 'Llama-3.1-8B-Instruct' \
        --tokenizer 'Llama-3.1-8B-Instruct' \
        --summary_prompt_path "../prompt/lastfm/summary.txt" \
		