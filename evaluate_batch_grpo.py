import torch
from transformers import GenerationConfig
from tqdm import tqdm
import re
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

# device_map = "auto"
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
        
        tokenizer.padding_side = "left"  
        model_inputs = tokenizer(
            chat_messages_list,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        ).to(model.device)
        
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

        input_lengths = [len(ids) for ids in model_inputs["input_ids"]]

        completions = []
        for i, seq in enumerate(sequence_ids):
            input_length = input_lengths[i]
            completion = tokenizer.decode(seq[input_length:], skip_special_tokens=True)
            completions.append(completion)
       
        return completions
    
    targets = []
    inputs = []
    cans = []
    for elm in val_data:
        prompt = elm["prompt"]
        target = elm["trueSelection"]
        targets.append(target)
        inputs.append(prompt)
        cans.append(elm["itemList"])

    batch_num = (len(inputs)-1)// batch_size + 1
    score = 0
    valid = 0
    for i in tqdm(range(batch_num), desc="Testing..."):
        start = i*batch_size
        end = min(len(inputs), start+batch_size)
        batch_inputs = inputs[start:end]
        outputs = output_generate(batch_inputs)
        batch_targets = targets[start:end]
        batch_cans = cans[start:end]
        for input_text, output, target,candidates in zip(batch_inputs, outputs, batch_targets, batch_cans):
            answer_match = re.search(r"<answer>(.*?)</answer>", output, flags=re.DOTALL)
             
            answer = answer_match.group(1) if answer_match else None
            num_cans=0

            if answer is not None:  
                answer = answer.strip()  
                num_cans = sum([1 for can in candidates if can in answer]) 
            else:  
                answer = ""   
                num_cans =0
            with open('GRPO_evaluated_reward.txt', 'a', encoding='utf-8') as file:
                file.write(f'#############################Question###########################\n{input_text}\n')
                file.write(f'#########################Output#####################\n{output}\n')   
                file.write(f'##############################Answer#################################\n{answer}\n')
                file.write(f'#############################Target############################\n{target}\n')
            if num_cans == 1:
                valid+=1    
                if target in answer:
                    score += 1
                    with open('GRPO_evaluated_reward.txt', 'a', encoding='utf-8') as file:
                        file.write(f"score increased to {score}\n")
            print("\n")
            
    return score/len(inputs),valid/len(inputs)