# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
HOME=/your/home/path

DATA_DIR="$HOME/ExplanRec/data/Books"

output_dir=/your/output_dir

log_dir=/your/log_dir
mkdir -p $log_dir

model_name_or_path=/your/model_path
validation_file=$DATA_DIR/explan_both_valid.json
sequential_file=$DATA_DIR/user_history.csv
max_hist_len=9
model_max_length=1024
task_type="both"
template_name="llama-3"

metadata_file=$DATA_DIR/Books_name.json
test_top_file=$DATA_DIR/test_top.txt
torch_dtype="bfloat16"
attn_implementation="flash_attention_2"
rec_model_type="SASRec"

cd $HOME/ExplanRec

export CUDA_VISIBLE_DEVICES=1


accelerate launch --config_file ./shell/config/infer.yaml ./src/inference.py \
    --preprocessing_num_workers 1 \
    --output_dir $output_dir \
    --per_device_eval_batch_size 1 \
    --model_name_or_path $model_name_or_path \
    --validation_file $validation_file \
    --sequential_file $sequential_file \
    --cache_dir $HOME/.cache \
    --max_hist_len $max_hist_len \
    --model_max_length $model_max_length \
    --task_type $task_type \
    --template_name $template_name \
    --inference_mode "case study" \
    --metadata_file $metadata_file \
    --test_top_file $test_top_file \
    --max_new_tokens 500 \
    --min_new_tokens 150 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --torch_dtype $torch_dtype \
    --attn_implementation $attn_implementation \
    --rec_model_type $rec_model_type \
    --temperature 0.1 \
    --repetition_penalty 1.1 \
    --do_sample True > $log_dir/explain.log 2>&1 