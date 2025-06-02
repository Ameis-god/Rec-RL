# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
HOME=/home/xieao1
DATA_DIR=$HOME/ExplanRec/data/Books

MODEL_NAME=0524_noweight_llama3_8b_instruct_ori/checkpoint-28857
output_dir=$HOME/ExplanRec/output/Books/alignment/${MODEL_NAME}
mkdir -p $output_dir
model_name_or_path=/mnt/d1/xieao/.cache/${MODEL_NAME}
validation_file=$DATA_DIR/both_valid_onlynext.json
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

# ## infer for next item retrieval task
CUDA_VISIBLE_DEVICES=1 accelerate launch ./src/inference.py \
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
    --inference_mode "uid2next" \
    --metadata_file $metadata_file \
    --test_top_file $test_top_file \
    --max_new_tokens 150 \
    --num_beams 1 \
    --num_return_sequences 1 \
    --torch_dtype $torch_dtype \
    --attn_implementation $attn_implementation \
    --rec_model_type $rec_model_type


# # infer for item recovery task
# accelerate launch ./src/inference.py \
#     --preprocessing_num_workers 1 \
#     --output_dir $output_dir \
#     --per_device_eval_batch_size 2 \
#     --model_name_or_path $model_name_or_path \
#     --validation_file $validation_file \
#     --sequential_file $sequential_file \
#     --cache_dir $HOME/.cache \
#     --max_hist_len $max_hist_len \
#     --model_max_length $model_max_length \
#     --task_type $task_type \
#     --template_name $template_name \
#     --inference_mode "uid2hist" \
#     --metadata_file $metadata_file \
#     --test_top_file $test_top_file \
#     --max_new_tokens 200 \
#     --num_beams 1 \
#     --num_return_sequences 1 \
#     --torch_dtype $torch_dtype \
#     --attn_implementation $attn_implementation \
#     --rec_model_type $rec_model_type


# # infer for item recovery task
# accelerate launch ./src/inference.py \
#     --preprocessing_num_workers 1 \
#     --output_dir $output_dir \
#     --per_device_eval_batch_size 2 \
#     --model_name_or_path $model_name_or_path \
#     --validation_file $validation_file \
#     --sequential_file $sequential_file \
#     --cache_dir $HOME/.cache \
#     --max_hist_len $max_hist_len \
#     --model_max_length $model_max_length \
#     --task_type $task_type \
#     --template_name $template_name \
#     --inference_mode "iid2title" \
#     --metadata_file $metadata_file \
#     --test_top_file $test_top_file \
#     --max_new_tokens 200 \
#     --num_beams 1 \
#     --num_return_sequences 1 \
#     --torch_dtype $torch_dtype \
#     --attn_implementation $attn_implementation \
#     --rec_model_type $rec_model_type

# # infer for item ranking task
# accelerate launch ./src/inference.py \
#     --preprocessing_num_workers 1 \
#     --output_dir $output_dir \
#     --per_device_eval_batch_size 2 \
#     --model_name_or_path $model_name_or_path \
#     --validation_file $validation_file \
#     --sequential_file $sequential_file \
#     --cache_dir $HOME/.cache \
#     --max_hist_len $max_hist_len \
#     --model_max_length $model_max_length \
#     --task_type $task_type \
#     --template_name $template_name \
#     --inference_mode "uidiid2rank" \
#     --metadata_file $metadata_file \
#     --test_top_file $test_top_file \
#     --max_new_tokens 500 \
#     --num_beams 1 \
#     --num_return_sequences 1 \
#     --torch_dtype $torch_dtype \
#     --attn_implementation $attn_implementation \
#     --rec_model_type $rec_model_type


# # ## infer for interest classification task
# accelerate launch ./src/inference.py \
#     --preprocessing_num_workers 1 \
#     --output_dir $output_dir \
#     --per_device_eval_batch_size 2 \
#     --model_name_or_path $model_name_or_path \
#     --validation_file $validation_file \
#     --sequential_file $sequential_file \
#     --cache_dir $HOME/.cache \
#     --max_hist_len $max_hist_len \
#     --model_max_length $model_max_length \
#     --task_type $task_type \
#     --template_name $template_name \
#     --inference_mode "uidiid2binary" \
#     --metadata_file $metadata_file \
#     --test_top_file $test_top_file \
#     --max_new_tokens 100 \
#     --num_beams 1 \
#     --num_return_sequences 1 \
#     --torch_dtype $torch_dtype \
#     --attn_implementation $attn_implementation \
#     --rec_model_type $rec_model_type


# # ## infer for history reconstruction task
# accelerate launch ./src/inference.py \
#     --preprocessing_num_workers 1 \
#     --output_dir $output_dir \
#     --per_device_eval_batch_size 2 \
#     --model_name_or_path $model_name_or_path \
#     --validation_file $validation_file \
#     --sequential_file $sequential_file \
#     --cache_dir $HOME/.cache \
#     --max_hist_len $max_hist_len \
#     --model_max_length $model_max_length \
#     --task_type $task_type \
#     --template_name $template_name \
#     --inference_mode "uid2hist" \
#     --metadata_file $metadata_file \
#     --test_top_file $test_top_file \
#     --max_new_tokens 500 \
#     --num_beams 1 \
#     --num_return_sequences 1 \
#     --torch_dtype $torch_dtype \
#     --attn_implementation $attn_implementation \
#     --rec_model_type $rec_model_type