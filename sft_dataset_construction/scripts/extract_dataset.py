import os  
import multiprocessing as mp  
from tqdm import tqdm  
import ujson as json  # Much faster than standard json  

def process_chunk(chunk):  
    """Process a chunk of data in parallel"""  
    results = []  
    for entry in chunk:  
        if 'Response' in entry and 'Target' in entry and 'Sft_prompt' in entry:  
            response = entry['Response']  
            target = entry['Target']  
            prompt = entry['Sft_prompt']  
            
            formatted_answer = f"<reasoning>\n{response}\n</reasoning>\n<answer>\n{target}\n</answer>"  
            
            results.append({  
                "Prompt": prompt,  
                "answer": formatted_answer  
            })  
    return results  

def reformat_json_data(input_path, output_path, chunk_size=500):  
    """  
    Process a large JSON file with parallel processing  
    """  
    # Read the input JSON file  
    print(f"Loading data from {input_path}...")  
    with open(input_path, 'r', encoding='utf-8') as file:  
        data = json.load(file)  
    
    total_entries = len(data)  
    print(f"Processing {total_entries} entries...")  
    
    # Split data into chunks for parallel processing  
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]  
    
    # Create a process pool  
    num_processes = min(mp.cpu_count(), 8)  # Use up to 8 cores  
    print(f"Using {num_processes} CPU cores for processing")  
    
    # Process chunks in parallel with progress bar  
    formatted_data = []  
    with mp.Pool(processes=num_processes) as pool:  
        results = list(tqdm(  
            pool.imap(process_chunk, chunks),  
            total=len(chunks),  
            desc="Processing data chunks"  
        ))  
        # Flatten results  
        for chunk_result in results:  
            formatted_data.extend(chunk_result)  
    
    # Save to the new JSON file  
    print(f"Writing {len(formatted_data)} entries to {output_path}...")  
    with open(output_path, 'w', encoding='utf-8') as file:  
        json.dump(formatted_data, file, ensure_ascii=False, indent=2)  
    
    print(f"Successfully processed {len(formatted_data)} entries")  

# Example usage  
if __name__ == "__main__":  
    input_path = "summary_dataset.json"  
    output_path = "formatted_summary_dataset.json"  
    
    reformat_json_data(input_path, output_path)