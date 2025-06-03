import csv
import os
import time
import argparse
import pandas as pd
import os.path as osp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, HarmCategory, HarmBlockThreshold

# 设置 Google Cloud 凭证
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gemini-oz-llm-01583c193c45.json"

# 初始化 Vertex AI
vertexai.init(project="gemini-oz-llm", location="us-central1")

def call_gemini(prompt):
    max_retry_cnt = 5
    result = "NULL"
    for i in range(max_retry_cnt):
        try:
            generation_config = {"temperature": 1.0}
            safety_config = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            model = GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=generation_config,
            )
            response = model.generate_content(prompt, safety_settings=safety_config)
            result = response.text
            break
        except Exception as e:
            error_msg = str(e)
            print(f"Gemini API Error: {error_msg}")            
            if "time" in error_msg:
                print("Rate limit reached. Waiting for 20 seconds...")
                time.sleep(20)
    if not result:
        result = "NULL"
    return result

def process_row(writer, sample, columns):
    question = sample['question'] 
    # 简化为字符计数而不是token计数
    input_token_num = len(question)
    output = call_gemini(question)
    all_writes = [sample[col] for col in columns]
    all_writes.append(output)
    writer.writerow(all_writes)
    output_token_num = len(output)
    return input_token_num, output_token_num

def process_hf_data(dataset, output_file, args):
    filename = os.path.basename(output_file)
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(args.input_columns.split(',')+['answer'])
        total_input_token_num, total_output_token_num = 0, 0

        try:
            with ThreadPoolExecutor(max_workers=args.num_process) as executor:
                futures = []
                for i, sample in enumerate(dataset):
                    futures.append(executor.submit(process_row, writer, sample, args.input_columns.split(',')))

                for future in tqdm(futures, desc=filename):
                    input_token_num, output_token_num = future.result()
                    total_input_token_num += input_token_num
                    total_output_token_num += output_token_num
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, exiting...")
            for future in futures:  
                future.cancel()  
        
            executor.shutdown(wait=False)  
        
            print("Results of completed tasks:")  
            for future in futures:
                if future.done() and not future.cancelled():
                    try:  
                        input_token_num, output_token_num = future.result()
                        total_input_token_num += input_token_num
                        total_output_token_num += output_token_num
                    except Exception as e:  
                        print(f"Task generated an exception: {e}")  
        
    return total_input_token_num, total_output_token_num

def load_jsonl_from_disk(file_path):
    df = pd.read_csv(file_path)    
    return df.to_dict(orient='records')

def main(args):
    total_token_num, total_cost = 0, 0

    infile = args.input_file
    outfile = args.output_file
    data_as_list = load_jsonl_from_disk(infile)

    total_input_token_num, total_output_token_num = process_hf_data(data_as_list, outfile, args)
    # 注意：这里的成本计算可能需要根据Gemini的实际定价进行调整
    cost = 10 * total_input_token_num / 1000000 + 30 * total_output_token_num / 1000000
    total_token_num = total_input_token_num + total_output_token_num
    print(">> Task done. Use {:d} tokens in total, and cost $ {:.4f}.".format(total_token_num, cost)) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--input_columns", type=str, default="question")
    args = parser.parse_args()
    main(args)