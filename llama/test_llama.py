import pandas as pd
import json
import os
import time
import torch
from transformers import pipeline
from tqdm import tqdm

# 強制優化 CPU
torch.set_num_threads(128)

CSV_FILE = "../BACE-1_dataset/bace.csv"  
OUTPUT_FILE = "output/llama_8b_results.json"
# 使用 Llama 3.1 8B (如果你空間夠，這絕對是首選)
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" 

os.makedirs("output", exist_ok=True)

print(f"[*] Loading {MODEL_NAME} (this may take a few minutes)...")
# Llama 8B 需要約 15GB RAM
generator = pipeline(
    "text-generation", 
    model=MODEL_NAME, 
    device_map="auto", # 自動偵測，在 CPU 節點會自動用 CPU
    torch_dtype=torch.bfloat16 # 使用 bfloat16 節省空間並加速
)

def main():
    df = pd.read_csv(CSV_FILE).head(10) # 先測 10 筆
    results = []
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # 回到 Llama 擅長的 Prompt 風格
        prompt = [
            {"role": "system", "content": "You are a medicinal chemist."},
            {"role": "user", "content": f"Analyze this BACE-1 inhibitor:\nSMILES: {row['mol']}\npIC50: {row['pIC50']:.2f}\nMW: {row['MW']:.2f}\nAlogP: {row['AlogP']:.2f}\nSummarize its drug-likeness and biological activity in 2 professional sentences."}
        ]

        # Llama 3.1 需要套用其專屬的 Chat Template
        templated_prompt = generator.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        output = generator(
            templated_prompt, 
            max_new_tokens=100, 
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        
        description = output[0]['generated_text'].split("<|assistant|>")[-1].strip()

        results.append({
            "CID": str(row['CID']),
            "description": description
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()