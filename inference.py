from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoTokenizer,AutoModelForCausalLM
from Prompt import *
import torch
from evaluate_batch_grpo import evaluate
from accelerate import Accelerator
import fire
from peft import PeftModel
import wandb

def inference( dataset="",
               batch_size: int = 0,
               base_model = "",
               resume_from_checkpoint: str = "",
               tokenizer = "",
               external_prompt_path = "",
                wandb_project: str = "", 
                wandb_name: str = "",  
               ):
    os.environ['WANDB_PROJECT'] = wandb_project
    wandb.init(  
        project=wandb_project,  
        name=wandb_name,  
        config={  
            "base_model": base_model,  
            "batch_size": batch_size,  
            "dataset": dataset,  
            "prompt_path": external_prompt_path or "prompt.txt"  ,
            "resume_from_checkpoint": resume_from_checkpoint
        }  
    )  
    base_model = base_model
    compute_dtype = getattr(torch, "bfloat16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
        # load_in_8bit=True,
    )
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
        t  = Prompt(external_prompt_path)
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t
    
    def process_data(data_point):  
        t = convert_dict_to_prompt(data_point)  
        prompt = str(t)    
        target = data_point["trueSelection"]  
        dic = {  
            "prompt": prompt,  
            "trueSelection": target  
        }  
        return dic
    
    data_files = {
        "test": "./grpo_datasets/lastfm/test.json",
    }


    data = load_dataset("json", data_files=data_files)
    data.cleanup_cache_files()
    print(data)

    val_data = data["test"].map(process_data)
   
    print(f"found {len(val_data)} matching prompts")  
    accuracy, valid_ratio= evaluate(model, tokenizer, val_data,batch_size=batch_size)
    print(accuracy, valid_ratio)
    
    wandb.log({  
        "test_accuracy": accuracy,
        "test_valid_ratio": valid_ratio  
    })  
    wandb.run.summary.update({  
        "final_accuracy": accuracy, 
        "final_valid_ratio": valid_ratio,   
        "model_type": base_model.split("/")[-1],  
    })  
    wandb.finish()  

if __name__ == "__main__":
    fire.Fire(inference)
