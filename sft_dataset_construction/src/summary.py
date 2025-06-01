from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoTokenizer,AutoModelForCausalLM
# from Prompt import *
import torch
from summary_batch import evaluate
from accelerate import Accelerator
import fire
from utils.Prompt_summary import Prompt_summary
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  

def inference( dataset="",
               batch_size: int = 0,
               base_model = "",
               summary_prompt_path = "",
               tokenizer="",
               ):
    compute_dtype = getattr(torch, "bfloat16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        # load_in_8bit=True,
    )
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,  
        quantization_config=bnb_config,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)     
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"
    

    
    def convert_dict_to_prompt(d:dict):  
        """Convert dictionary to Prompt object, generate prompt for model summary."""  
        # external_prompt_path = "../prompt/amazon_book/summary.txt"
        t = Prompt_summary(summary_prompt_path)  
        t.response = d["Response"]
        return t  

    def process_data(data_point):  
        """Process data points into the required format"""  
        t = convert_dict_to_prompt(data_point)   
        prompt = str(t)   
        target = data_point["Target"]  
        sft_prompt = data_point["Sft_prompt"]
        dic = {  
            "prompt": prompt, 
            "completion": target,  
            "sft_prompt":sft_prompt
        }  
        return dic
    

    #### Load dataset #############
    data_files = {
         "generated_data": "generate_dataset.json",
    }


    data = load_dataset("json", data_files=data_files)
    data.cleanup_cache_files()
    print(data)

    processed_data = data["generated_data"].map(process_data)
   
    evaluate(model, tokenizer, processed_data, batch_size=batch_size)
    


if __name__ == "__main__":
    fire.Fire(inference)
