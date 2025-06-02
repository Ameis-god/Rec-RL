# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


### for training SASRec model

HOME=/your/home/path
export CUDA_VISIBLE_DEVICES=0

export DISABLE_MLFLOW_INTEGRATION=true;
export WANDB_DIR=$HOME/.cache/
export WANDB_PROJECT="ExplanRec"

DATA_DIR="$HOME/ExplanRec/data/Books"

attn_implementation="flash_attention_2"
model_name_or_path=/your/base_model/path
rec_model_name_or_path=/your/rec_model/path
rec_model_type="SASRec"
model_max_length=1024
torch_dtype="bfloat16"

train_file=$DATA_DIR/both_train_onlynext.json
validation_file=$DATA_DIR/both_valid_onlynext.json
sequential_file=$DATA_DIR/user_history.csv
max_hist_len=9
task_type="both"
template_name="llama-3"

TASK_NAME="your_task_name"

output_dir=$DATA_DIR/$TASK_NAME
cd $HOME/ExplanRec
mkdir -p $HOME/ExplanRec/log

torchrun --nnodes=1 --nproc_per_node 1 --master_port=29501 ./src/sft_training.py \
    --seed 2024 \
    --do_train \
    --do_eval \
    --bf16 \
    --attn_implementation $attn_implementation \
    --rec_model_name_or_path $rec_model_name_or_path \
    --model_name_or_path $model_name_or_path \
    --model_max_length $model_max_length \
    --cache_dir $HOME/.cache \
    --torch_dtype $torch_dtype \
    --train_file $train_file \
    --validation_file $validation_file \
    --sequential_file $sequential_file \
    --max_hist_len $max_hist_len \
    --task_type $task_type \
    --template_name $template_name \
    --data_type_filter "uid2next,uid2next_t,sharegpt" \
    --rec_model_type $rec_model_type \
    --output_dir $output_dir \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --dataloader_drop_last False \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --report_to wandb \
    --run_name $TASK_NAME > $HOME/ExplanRec/log/$TASK_NAME.log 2>&1
    