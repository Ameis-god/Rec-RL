#!/bin/bash
# Set CUDA visible devices if needed
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run the training script with Fire parameters
python train.py \
  --dataset="./grpo_datasets/lastfm/train.json" \
  --model_name='Qwen2.5-7B-Instruct' \
  --tokenizer_name='Qwen2.5-7B-Instruct' \
  --resume_from_checkpoint="/your/path/to/checkpoint" \
  --checkpoint_path="/your/path/to/save" \
  --world_size=4 \
  --seed=42 \
  --checkpoint_interval=100 \
  --train_batch_size=8 \
  --lr=5e-6 \
  --kl_weight=0.01 \
  --clip_eps=0.2 \
  --group_size=8 \
  --rollouts_per_step=32 \
  --epochs_per_step=1 \
  --total_epochs=2 \
  --max_norm=1.0 \
  --max_length=1024 \
  --top_p=1.0 \
  --temperature=1.0\
  --wandb_project="DeepRec-R1" \
  --wandb_name='AR-GRPO' > grpo.log