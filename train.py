from pathlib import Path
import random
import re
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm 
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch
from Prompt import Prompt
import os  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  
torch.backends.cudnn.benchmark = True 
from datetime import datetime  
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'    
from datasets import load_dataset
from peft import  PeftModel
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import fire
##############GLOBAL VARIABLE##############
recent_returns = []
# THRESHOLD
THRESHOLD_MEDIUM = 256 * 0.6 * (0.5 + 1.5)  
THRESHOLD_HIGH = 256 * 0.8 * (0.5 + 1.5)    
#Add this function for distributed environment initialization.
def setup_distributed(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12225'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set GPU device

# Function to load pre-trained model and tokenizer.
def load_model(
    model_name_or_path: str,
    tokenizer_name : str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    resume_from_checkpoint="",
    device_map=None,
    is_trainable=False
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training
    # Set the padding token as the end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    # compute_dtype = getattr(torch, "bfloat16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        attn_implementation="flash_attention_2",
        device_map=device_map,
        quantization_config=bnb_config
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, resume_from_checkpoint, 
                                        is_trainable=is_trainable)
    model.print_trainable_parameters()
    # print_trainable_parameters(base_model)
    # model.print_trainable_parameters()
    return model, tokenizer

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
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>?$"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# get avg_50 returns
def get_average_recent_returns():
    if not recent_returns:
        return 0.0
    # Only use the most recent 50 returns, or all available if less than 50
    recent = recent_returns[-50:]
    return sum(recent) / len(recent)


# Execute the gradient-free rollout process to generate answers and calculate rewards.
@torch.no_grad()
def rollout(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    tasks: list[str],  
    oracle_answers: list[str],  
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_title_length :int =0
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    
    model.eval()
    batch_size = len(tasks)
    chat_messages_list = []
    for task in tasks:
        chat_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": task,
            },
        ]
        chat_prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        chat_messages_list.append(chat_prompt)
    
    tokenizer.padding_side = "left"  
    
    model_device = model.device  # Get the model's actual device 
    model_inputs = tokenizer(
        chat_messages_list,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    ).to(model_device)

    expanded_inputs = {}
    for key, value in model_inputs.items():
        value = value.to(model_device)  
        expanded_inputs[key] = value.repeat_interleave(num_rollouts, dim=0)
    
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**expanded_inputs, generation_config=generation_config)
    input_lengths = [len(ids) for ids in expanded_inputs["input_ids"]]
    
    completions = []
    for i, seq in enumerate(sequence_ids):
        input_length = input_lengths[i]
        completion = tokenizer.decode(seq[input_length:], skip_special_tokens=True)
        completions.append(completion)
    
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    for i, length in enumerate(input_lengths):
        action_mask[i, length:] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]
    
    # get avg_return
    avg_returns = get_average_recent_returns()
    
   # determine reward
    returns = torch.zeros(batch_size * num_rollouts, 1, dtype=torch.float)

    oracle_answers = [elem for elem in oracle_answers for _ in range(8)]  
    # Calculate rewards for all completions in parallel.
    for i, (completion, oracle_answer) in enumerate(zip(completions, oracle_answers)):
        answer_match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        answer = answer_match.group(1) if answer_match else None
        reward = 0

        if answer is not None:  
            answer = answer.strip()  
        else:  
            answer = ""  
        # with open('completion_LORA_reward.txt', 'a', encoding='utf-8') as file:
        #     file.write(f'################Prompt########################:\n {tasks[i//num_rollouts]}\n')
        #     file.write(f'################completion########################:\n {completion}\n')
        #     file.write(f'#####################answer################: \n{answer}\n')
        #     file.write(f'#####################true_answer################: \n{oracle_answer}\n')
        if answer is not None:
           
            if avg_returns < THRESHOLD_MEDIUM:
                if answer == oracle_answer:
                    reward = 4
                elif oracle_answer in answer:
                    if len(answer)<=max_title_length+10:
                        reward = 1.5
                    else:
                        reward = 0
                else:
                    reward = 0
                completion_list = [completion]
                reward += strict_format_reward_func(completion_list)[0]
                returns[i] = reward
            elif avg_returns > THRESHOLD_HIGH:
                if answer == oracle_answer:
                    reward = 4
                else:
                    reward = 0
                returns[i] = reward
            else:
                if answer == oracle_answer:
                    reward = 4
                else:
                    reward = 0
                completion_list = [completion]
                reward += strict_format_reward_func(completion_list)[0]
                returns[i] = reward
            
    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions

# Initialize random number generator
def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)

#Normalize the advantage values. This part is for calculating the advantage function
def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)

def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: AutoModelForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        # logits=logits[:, :-1].to(torch.float32),
        logits=logits[:, :-1],  # remove .to(torch.float32) 
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def convert_dict_to_prompt(d:dict,external_prompt_path:str = "./prompt/lastfm/music.txt"):  
    """Convert to prompt"""  
    t = Prompt(external_prompt_path)  
    d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]  
    t.historyList = d["historyList"]  
    t.itemList = d["itemList"]  
    t.trueSelection = d["trueSelection"] 
    return t  

def process_data(data_point):  
    """Process data points to the format required"""  
    t = convert_dict_to_prompt(data_point)  
    prompt = str(t)  
    target = data_point["trueSelection"]  
    dic = {  
        "prompt": prompt,  
        "completion": target  
    }  
    return dic

def train_parallel(model, optimizer, experience_sampler, objective, epochs_per_step, model_device, max_norm,rank):
    
    scaler = torch.amp.GradScaler()
    for step_epoch in range(epochs_per_step):
        model.train()
        for exp in experience_sampler:
            exp: Experience
            
            exp = exp.to(model_device)
            optimizer.zero_grad()
            
            
            with torch.amp.autocast(device_type='cuda'):
                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )
                loss, kl = objective(log_probs=log_probs, experience=exp)
            if not loss.isfinite():
                if rank==0:
                    print(f"Loss not finite, skipping backward, loss={loss}")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
            loss_value = loss.item()
            if rank == 0:
                print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}, loss = {loss_value}")
                wandb.log({"kl": kl, "grad_norm": grad_norm, "loss": loss_value})
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            
    return model

def find_max_completion_length(prompts):

    completion_lengths = [len(item['completion']) for item in prompts]
    
    # If the list is not empty, return the maximum length; otherwise, return 0.
    if completion_lengths:
        return max(completion_lengths)
    else:
        return 0

def main_distributed(
    rank,
    world_size,
    dataset='',
    model_name='',
    tokenizer_name='',
    resume_from_checkpoint="",
    checkpoint_path="",
    seed=42,
    checkpoint_interval=100,
    train_batch_size=16,
    lr=5e-6,
    kl_weight=0.01,
    clip_eps=0.2,
    group_size=8,
    rollouts_per_step=32,
    epochs_per_step=1,
    total_epochs=5,
    max_norm=1.0,
    max_length=1024,
    top_p=1.0,
    temperature=1.0,
     wandb_project="",
    wandb_name=""
): 
    
    setup_distributed(rank,world_size)
    wandb_name = wandb_name  
    
    checkpoint_path = Path(checkpoint_path) 

    cpu_device = torch.device("cpu")  
    init_rng(seed+rank)  
   
    num_gpus = torch.cuda.device_count()  
    print(f"Total available GPUs: {num_gpus}")  
    
    model_device = rank
    reference_device = rank
    # load model  
    reference_model, _ = load_model(model_name, device_map=reference_device,tokenizer_name=tokenizer_name,resume_from_checkpoint=resume_from_checkpoint)  
    model, tokenizer = load_model(model_name, device_map=model_device,tokenizer_name=tokenizer_name,resume_from_checkpoint=resume_from_checkpoint,is_trainable=True)  
    
    model = DDP(model,device_ids=[rank],output_device = rank)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)  
    
    reference_model.eval()  
    model.module.gradient_checkpointing_enable(  
        gradient_checkpointing_kwargs={"use_reentrant": False}  
    )  

    pad_token_id = tokenizer.eos_token_id  
    data_files = {
        "train": dataset if dataset else "./grpo_datasets/lastfm/train.json"
    }
 
    data = load_dataset("json", data_files=data_files)

    replay_buffer = ReplayBuffer()  
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)  

    # Only initialize wandb on rank 0
    if rank==0 and wandb_project is not None:  
       
        wandb.init(  
            project=wandb_project,  
            name=wandb_name,  
            config={  
                'dataset': dataset,
                "base_model": model_name,  
                "checkpoint": checkpoint_path,  
                "batch_size": train_batch_size,  
                "learning_rate": lr,  
                "dataset": "amazon",  
                "seed": seed,  
                "group_size": group_size,  
                "rollouts_per_step": rollouts_per_step ,
                "num_workers": 4,
                "total_epochs":total_epochs,
                "world_size": world_size
            }  
        )  
    else: 
        wandb.init(mode="disabled")  
     
    # document global_step
    global_step = 0
    for epoch in range(total_epochs):    
        prompts = data["train"].shuffle(seed=42+epoch).map(process_data)
        prompts = prompts.remove_columns(data["train"].column_names)
        max_title_length = find_max_completion_length(prompts)
        if rank==0:
            print(f"found {len(prompts)} matching prompts")  
         # Create a distributed sampler
        train_sampler = DistributedSampler(
            prompts, 
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42+epoch
        ) 
        prompt_loader = DataLoader(  
            prompts,  
            batch_size=rollouts_per_step // world_size,  # Adjust batch size per GPU
            shuffle=False,   # Don't shuffle - sampler does it
            drop_last=True,  
            pin_memory=True,  
            num_workers=4,    
            sampler=train_sampler
        )  
        if rank==0:
            epoch_pbar = tqdm(total=len(prompt_loader), desc=f"Epoch {epoch+1}/{total_epochs}")
            print(f"Starting epoch {epoch+1}/{total_epochs}")
        for k, prompt_batch in enumerate(prompt_loader):  
            rollout_returns = []  

            replay_buffer.clear()  

            questions = prompt_batch["prompt"]  
            answers = prompt_batch["completion"]  

            with torch.no_grad():  
                # Process multiple questions in parallel
                batch_size = len(questions)
                for i in range(0, batch_size, batch_size//2): 
                    batch_questions = questions[i:i+batch_size//2]
                    batch_answers = answers[i:i+batch_size//2]
                    
                   
                    sequence_ids, returns, action_mask, completions = rollout(
                        model.module, # Access the underlying model
                        tokenizer,
                        batch_questions,
                        batch_answers,
                        num_rollouts=group_size,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        max_title_length = max_title_length
                    )
                    
                    # process results
                    for j in range(len(batch_questions)):
                        q = batch_questions[j]
                        a = batch_answers[j]
                        
                        
                        q_sequence_ids = sequence_ids[j*group_size:(j+1)*group_size]
                        q_returns = returns[j*group_size:(j+1)*group_size]
                        q_action_mask = action_mask[j*group_size:(j+1)*group_size]
                        q_completions = completions[j*group_size:(j+1)*group_size]
                        if rank == 0:
                            print(f"rollout q='{q}', a='{a}', returns={q_returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}")
                            
                            # with open('completion_LORA_reward.txt', 'a', encoding='utf-8') as file:  
                            #     file.write(f"##############rollout############### q='{q}', a='{a}', returns={q_returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}\n"  
                            # )  
                        rollout_returns.append(q_returns.cpu())
                        
                        # compute advantages
                        advantages = group_advantages(q_returns)
                        attention_mask = q_sequence_ids != pad_token_id
                        
                        # compute log_probs
                        log_probs = sequences_log_probs(
                            model=model.module,
                            sequence_ids=q_sequence_ids,
                            attention_mask=attention_mask,
                        )
                        
                        # compute log_probs
                        log_probs_ref = sequences_log_probs(
                            model=reference_model,
                            sequence_ids=q_sequence_ids,
                            attention_mask=attention_mask,
                        )
                        
                        # compute KL
                        kl = approx_kl_divergence(
                            log_probs=log_probs,
                            log_probs_ref=log_probs_ref,
                            action_mask=q_action_mask,
                        )
                        
                        # Create an experience object and add it to the playback buffer
                        experience = Experience(
                            sequences=q_sequence_ids,
                            action_log_probs=log_probs,
                            log_probs_ref=log_probs_ref,
                            returns=q_returns,
                            advantages=advantages,
                            attention_mask=attention_mask,
                            action_mask=q_action_mask,
                            kl=kl,
                        )
                        replay_buffer.append(experience.to(cpu_device))
            torch.cuda.empty_cache()  
            # Synchronizing results on multiple GPUs
            if len(rollout_returns)>0:
                episode_return_sum = torch.stack(rollout_returns).sum()  
                # All-reduce to get the global sum across processes
                if world_size>1:
                   
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    episode_return_sum = episode_return_sum.to(device)
                    
                    if device.type == "cpu":
                        dist.all_reduce(episode_return_sum, op=dist.ReduceOp.SUM)
                    else:
                        dist.all_reduce(episode_return_sum, op=dist.ReduceOp.SUM)
                        
                    # Add current returns to the list of recent returns
                recent_returns.append(episode_return_sum.item())
                # Calculate the average of recent returns
                avg_recent_returns = get_average_recent_returns()
                
                global_step += 1
                if rank ==0:
                    # print(f"Global step: {global_step}, Epoch: {epoch+1}")
                    # print(f"returns of step {global_step}: {episode_return_sum:.4f}")  
                   
                    epoch_pbar.update(1)
                    epoch_pbar.set_postfix({
                        'step': global_step,
                        'return': f"{episode_return_sum:.4f}",
                        'avg_return': f"{avg_recent_returns:.4f}",
                        'epoch': f"{epoch+1}/{total_epochs}"
                    })
                    wandb.log({  
                        "returns": episode_return_sum,
                         "avg_recent_returns": avg_recent_returns,
                        'global_step':global_step
                    })  

            # Create an experience sampler for training 
            experience_sampler = DataLoader(  
                replay_buffer,  
                batch_size=train_batch_size,  
                shuffle=True,  
                drop_last=True,  
                collate_fn=join_experience_batch,  
            )  
            
            model = train_parallel(
                model=model,
                optimizer=optimizer,
                experience_sampler=experience_sampler,
                objective=objective,
                epochs_per_step=epochs_per_step,
                model_device=model.device,
                max_norm=max_norm,
                rank = rank
            )

            # Save model checkpoints regularly
            if (  rank == 0 and 
                checkpoint_path is not None  
                and checkpoint_interval is not None  
                and (global_step) % checkpoint_interval == 0  
            ):  
                model.module.save_pretrained(checkpoint_path / f"step_{global_step}")  
       # Close the progress bar after each epoch.
        if rank == 0:
            epoch_pbar.close()
            print(f"Epoch {epoch+1} completed. Global step: {global_step}") 
    # Clean up distributed process group
    dist.destroy_process_group()
    if rank ==0 :
        wandb.finish()  
        torch.cuda.empty_cache()  
        # save final_checkpoint  
        if checkpoint_path is not None:  
            model.module.save_pretrained(checkpoint_path / f"step_{global_step}")  

# Replace the original main function with the launcher
def main():
    """
    Launch distributed training with configurable parameters.
    
    Args:
        dataset: Path to the training dataset
        model_name: Path to the base model
        tokenizer_name: Path to the tokenizer
        resume_from_checkpoint: Path to the checkpoint to resume from
        checkpoint_path: Path to save checkpoints
        world_size: Number of GPUs to use
        seed: Random seed
        wandb_project: W&B project name
        checkpoint_interval: Save checkpoint every N steps
        train_batch_size: Batch size for training
        lr: Learning rate
        kl_weight: KL divergence weight
        clip_eps: Clipping threshold
        group_size: Group size for advantage computation
        rollouts_per_step: Number of rollouts per step
        epochs_per_step: Number of epochs per step
        total_epochs: Total number of epochs
        max_norm: Gradient clipping threshold
        max_length: Maximum sequence length
        top_p: Top-p sampling parameter
        temperature: Temperature for generation,
        wandb_name:W&B project name
    """
    # Get arguments from Fire
    args = fire.Fire(lambda **kwargs: kwargs)
    # Extract world_size or use default
    world_size = args.pop('world_size', 2)
    mp.spawn(
        main_distributed,
        args=(world_size,) + tuple(args.values()),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()