# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
HOME=/your/home/path
cd $HOME/ExplanRec

PEFT_MODEL_PATH=/your/peft_model/path # 0524_model/checkpoint-28857
BASE_MODEL=/your/base_model/path # Llama-3-8B-Instruct
rec_model_name_or_path=/your/rec_model/path
rec_model_type="SASRec"
output_dir=/your/output_dir
task_type="both"

python ./src/merge.py \
    --output_dir ${output_dir} \
    --cache_dir $HOME/.cache \
    --peft_model_name  ${PEFT_MODEL_PATH}/${MODEL_NAME} \
    --model_name_or_path  ${BASE_MODEL} \
    --rec_model_name_or_path ${rec_model_name_or_path} \
    --task_type ${task_type} \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --rec_model_type ${rec_model_type}