from datasets import load_dataset
from transformers import  BitsAndBytesConfig, AutoTokenizer,AutoModelForCausalLM
from utils.Prompt_explain import Prompt_explain
from utils.Prompt import *
import torch
from torch.utils.data import DataLoader
from explain_batch import evaluate
from accelerate import Accelerator
import fire
from peft import PeftModel
import wandb 
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  
def inference( dataset="",
               batch_size: int = 0,
               base_model = "",
               tokenizer = "",
               sft_prompt_path = "",
               explain_prompt_path = "",
                wandb_project: str = "",  
                wandb_name: str = "",   
                resume_from_checkpoint: str = None,  
               ):
    os.environ['WANDB_PROJECT'] = wandb_project
    # initialize wandb 
    wandb.init(  
        project=wandb_project,  
        name=wandb_name,  
        config={  
            "base_model": base_model,  
            "batch_size": batch_size,  
            "dataset": dataset,  
            "sft_prompt_path": sft_prompt_path,
            "explain_prompt_path": explain_prompt_path,
            "resume_from_checkpoint": resume_from_checkpoint,
            "tokenizer": tokenizer
        }  
    )  
    base_model = base_model
    compute_dtype = getattr(torch, "bfloat16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        # load_in_8bit=True,
    )
    os.environ['WANDB_PROJECT'] = wandb_project
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,  
        quantization_config=bnb_config,
    )
    if resume_from_checkpoint != "":
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"
    
    def convert_dict_to_prompt(d:dict):
        """Convert dictionary to Prompt object, generate SFT prompt"""  
        # external_prompt_path = "../prompt/lastfm/music.txt"
        s = Prompt(sft_prompt_path)   
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        s.historyList = d["historyList"]
        s.itemList = d["itemList"]
        s.trueSelection = d["trueSelection"]
        return s
    
    def convert_dict_to_prompt_explain(d:dict):  
        """Convert dictionary to Prompt object, generate prompt for model explanation."""  
        # external_prompt_path = "../prompt/lastfm/explain.txt"
        t = Prompt_explain(explain_prompt_path)  
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]  
        t.historyList = d["historyList"]  
        t.trueSelection = d["trueSelection"]  
        return t  

    
    def process_data(data_point):  
        """Process data points into the required format"""  
        ####The following is to generate explain_prompt############
        t = convert_dict_to_prompt_explain(data_point)  
        prompt_explain = str(t)  
        ####The following is to generate sft_prompt############
        s = convert_dict_to_prompt(data_point)   
        prompt = str(s)  
        target = data_point["trueSelection"]  
        dic = {  
            "sft_prompt": prompt,  
            "prompt": prompt_explain, 
            "completion": target  
        }  
        return dic
    
    #### Load dataset #############
    data_files = {
        "original_data": "../grpo_datasets/lastfm/train.json",
    }


    data = load_dataset("json", data_files=data_files)
    data.cleanup_cache_files()
    print(data)

    processed_data = data["original_data"].map(process_data)
    print(f"found {len(processed_data)} matching prompts")  
    filtering_rate= evaluate(model, tokenizer, processed_data, batch_size=batch_size)
    print(filtering_rate)
    wandb.log({  
        "filtering_rate": filtering_rate
    })  
    wandb.run.summary.update({  
        "filtering_rate": filtering_rate,  
        "model_type": base_model.split("/")[-1],  
    })  
    wandb.finish()  

if __name__ == "__main__":
    fire.Fire(inference)
