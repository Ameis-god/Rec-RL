import json  
from sklearn.model_selection import train_test_split  
import os  

# read JSON 
data_path = "formatted_summary_dataset.json"  
with open(data_path, 'r', encoding='utf-8') as f:  
    data = json.load(f)  

# divide train and val (80%, 20% )  
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)  

print(f"#train: {len(train_data)}")  
print(f"#val: {len(val_data)}")  

# create data_dir 
data_dir = "./data/lastfm"  
os.makedirs(data_dir, exist_ok=True)  

# save data
train_path = os.path.join(data_dir, "train_dataset.json")  
val_path = os.path.join(data_dir, "val_dataset.json")  

with open(train_path, 'w', encoding='utf-8') as f:  
    json.dump(train_data, f, ensure_ascii=False, indent=2)  

with open(val_path, 'w', encoding='utf-8') as f:  
    json.dump(val_data, f, ensure_ascii=False, indent=2)  

print(f"train saved: {train_path}")  
print(f"val saved: {val_path}")