import json
import random
from datasets import load_dataset

def prepare_sharegpt(output_file="sharegpt_data.json", limit=100):
    print("Downloading ShareGPT from HuggingFace")
    dataset = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered", 
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json", 
        split="train"
    )
    
    formatted_data = []
    
    print(f"Processing {limit} random samples...")
    shuffled_indices = list(range(len(dataset)))
    random.shuffle(shuffled_indices)
    
    count = 0
    for i in shuffled_indices:
        if count >= limit: break
        
        item = dataset[i]
        
        conversations = item.get("conversations", [])
        if not conversations: continue
        
        if conversations[0]["from"] == "human":
            prompt = conversations[0]["value"]
            
            if len(prompt) < 50 or len(prompt) > 8000:
                continue
                
            formatted_data.append({
                "id": f"sharegpt_{item['id']}",
                "category": "sharegpt_real",
                "prompt": prompt,
                "min_output_tokens": 10,     
                "max_output_tokens": 1024,              
                "ignore_eos": False
            })
            count += 1

    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Saved {len(formatted_data)} real prompts to {output_file}")

if __name__ == "__main__":
    prepare_sharegpt()