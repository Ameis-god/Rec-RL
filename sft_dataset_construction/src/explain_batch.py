import torch
import json
from transformers import GenerationConfig
from tqdm import tqdm
import re
import json  
# DeepSeek Zero system prompt
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

device_map = "auto"
def evaluate(
    model,
    tokenizer,
    val_data,
    batch_size: int = 32
):
    
    def output_generate(
        prompts,
        temperature = 0,
    ):
        chat_messages_list = []
        for prompt in prompts:
            chat_messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            chat_prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            chat_messages_list.append(chat_prompt)
        
        #set tokenizer's padding_side  
        tokenizer.padding_side = "left"  
        model_device = model.device
        model_inputs = tokenizer(
            chat_messages_list,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        ).to(model_device)
        
        pad_token_id = tokenizer.eos_token_id
        generation_config = GenerationConfig(
            do_sample=True,
            top_p=1.0,
            temperature=1.0,
            max_length=1024,
            pad_token_id=pad_token_id,
        )
        sequence_ids = model.generate(
            **model_inputs,
            generation_config = generation_config
            )
        # cpmpute the input length
        input_lengths = [len(ids) for ids in model_inputs["input_ids"]]
        # extract completion
        completions = []
        for i, seq in enumerate(sequence_ids):
            input_length = input_lengths[i]
            completion = tokenizer.decode(seq[input_length:], skip_special_tokens=True)
            completions.append(completion)
       
        return completions
    
    targets = []
    inputs = []
    sft_prompts = []
    data =[]  
    for elm in val_data:
        prompt = elm["prompt"]
        target = elm["trueSelection"]
        sft_prompt= elm["sft_prompt"]
        targets.append(target)
        inputs.append(prompt)
        sft_prompts.append(sft_prompt)

    batch_num = (len(inputs)-1)// batch_size + 1
    score = 0
    count =0
    for i in tqdm(range(batch_num), desc="Generating..."):
        start = i*batch_size
        end = min(len(inputs), start+batch_size)
        batch_inputs = inputs[start:end]
        outputs = output_generate(batch_inputs)
        batch_targets = targets[start:end]
        batch_sft_prompts = sft_prompts[start:end]
        for input_text, output, target, sft_prompt in zip(batch_inputs, outputs, batch_targets,batch_sft_prompts):
            count+=1
            answer_match = re.search(r"<answer>(.*?)</answer>", output, flags=re.DOTALL)
            answer = answer_match.group(1) if answer_match else None
            if answer is not None:  
                answer = answer.strip()  
            else:  
                answer = ""  
            with open('generate_dataset.txt', 'a', encoding='utf-8') as file:
                file.write(f'#############################Question###########################\n{input_text}\n')
                file.write(f'#########################Output#####################\n{output}\n')   
                file.write(f'##############################Answer#################################\n{answer}\n')
                file.write(f'#############################Target############################\n{target}\n')
            if 'Yes' in output or 'yes' in output: 
                score += 1     
                with open('generate_dataset.txt', 'a', encoding='utf-8') as file:
                    file.write(f"score increased to {score}\n")
                data.append({  
                    "Sft_prompt" : sft_prompt,
                    "Prompt": input_text,  
                    "Response": output,
                    "Target":target  
                })  
            print('Current Score Ratio:',score/count)
            print("\n")
            with open('generate_dataset.json', 'w', encoding='utf-8') as f:  
                json.dump(data, f, ensure_ascii=False, indent=4)  
            torch.cuda.empty_cache()  # clear cache 
    with open('generate_dataset.json', 'w', encoding='utf-8') as f:  
        json.dump(data, f, ensure_ascii=False, indent=4)  
        print('data all saved!!!')
            
    return score/len(inputs)