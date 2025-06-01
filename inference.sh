
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node 1 --master_port=22095\
         inference.py \
        --dataset lastfm \
        --batch_size 64 \
        --base_model 'Qwen2.5-7B-Instruct' \
        --tokenizer 'Qwen2.5-7B-Instruct' \
        --external_prompt_path './prompt/lastfm/music.txt' \
        --resume_from_checkpoint "/your/path/to/checkpoint"\
        --wandb_project "DeepRec-R1" \
        --wandb_name inference > inference.log
	