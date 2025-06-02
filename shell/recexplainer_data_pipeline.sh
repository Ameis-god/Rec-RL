# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
HOME=/your/home/path

PROCESS_DATA_DIR="$HOME/ExplanRec/data/Books"

gpt_response_file="$PROCESS_DATA_DIR/gemini_data/test_response.csv"
max_seq_len=9
model_name="/your/base_model/path" # "meta-llama/Meta-Llama-3-8B-Instruct"

model_max_length=1024  # Optimized for single GPU training
EXE_DIR="$HOME/ExplanRec/preprocess"
cd $EXE_DIR


# ## if use iid2summary task, generate user history summary query
# python amazon_generate_v3.py --seqdata_file $PROCESS_DATA_DIR/user_history.csv --metadata_file $PROCESS_DATA_DIR/Books_name.json \
#     --gpt_query_file $PROCESS_DATA_DIR/gemini_data/test_query.csv --max_seq_len $max_seq_len \
#     --model_name $model_name --model_max_length $model_max_length

# ### gemini data generation for user history summary
# if [[ -e $gpt_response_file ]]; then  
#     echo "All files exist. jump this step." 
# else
#     echo "generate gpt_response_file"
    
#     python gemini_api.py --input_file $PROCESS_DATA_DIR/gemini_data/test_query.csv --output_file $gpt_response_file

# fi

### For SASRec model:

### generate training and testing data for alignment tasks
python amazon_generate_v3.py --sharegpt_file $PROCESS_DATA_DIR/ShareGPT_V3_unfiltered_cleaned_split.json \
    --seqdata_file $PROCESS_DATA_DIR/user_history.csv --metadata_file $PROCESS_DATA_DIR/Books_name.json \
    --sim_item_file $PROCESS_DATA_DIR/sim_item.txt --train_top_file $PROCESS_DATA_DIR/train_top.txt --test_top_file $PROCESS_DATA_DIR/test_top.txt \
    --gpt_response_file $gpt_response_file \
    --save_intention_file $PROCESS_DATA_DIR/intention --save_behavior_file $PROCESS_DATA_DIR/behaviour --save_both_file $PROCESS_DATA_DIR/both \
    --max_seq_len $max_seq_len --model_name $model_name --model_max_length $model_max_length

### generate testing data for explanation task
python explan_data_gen.py --data_dir $PROCESS_DATA_DIR --seqdata_file $PROCESS_DATA_DIR/user_history.csv --metadata_file $PROCESS_DATA_DIR/Books_name.json \
    --test_top_file $PROCESS_DATA_DIR/test_top.txt --max_seq_len $max_seq_len --max_samples 500 --split "valid"

### generate training data for explanation task, used to train classifier and score predictor
python explan_data_gen.py --data_dir $PROCESS_DATA_DIR --seqdata_file $PROCESS_DATA_DIR/sequential_data.txt --metadata_file $PROCESS_DATA_DIR/metadata.json \
    --test_top_file $PROCESS_DATA_DIR/train_top.txt --max_seq_len $max_seq_len --max_samples 2000 --split "train"
