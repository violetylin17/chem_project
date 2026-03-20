import pandas as pd
import json
import os
import torch
from transformers import pipeline
from tqdm import tqdm
import time

# ==========================================
# 1. Main Settings
# ==========================================
torch.set_num_threads(128)

CSV_FILE = "../BACE-1_dataset/bace.csv"  
OUTPUT_FILE = "output/llama_full_results.json"
MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B-Instruct"

os.makedirs("output", exist_ok=True)

# ==========================================
# 2. Load Model (use bfloat16 to speed up process)
# ==========================================
print(f"[*] Loading {MODEL_NAME} to CPU Memory...")
generator = pipeline(
    "text-generation", 
    model=MODEL_NAME, 
    device=-1,        # Use CPU
    torch_dtype=torch.bfloat16 
)

def main():
    # Load Data
    df = pd.read_csv(CSV_FILE)
    
    # --- Continue Logic ---
    results = []
    processed_cids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed_cids = {str(r['CID']) for r in results}
            print(f"[*] Detected existing progress: {len(processed_cids)} entries completed. Continuing with the remaining data...")
        except Exception as e:
            print(f"[*] Loading failed, start from the beginning: {e}")

    # --- Start the Process ---
    print(f"[*] Total: {len(df)}  | Pending: {len(df) - len(processed_cids)} ")
    start_time = time.time()

    for index, row in tqdm(df.iterrows(), total=len(df)):
        cid = str(row['CID'])
        
        # Skip thoese that's processed
        if cid in processed_cids:
            continue

        # Llama 3.1 command
        messages = [
            {
                "role": "system", 
                "content": "You are a professional medicinal chemistry analyzer. Focus ONLY on molecular property analysis and protein-ligand binding potential. Do not discuss chemical synthesis."
            },
            {
                "role": "user", 
                "content": (
                    f"Analyze this candidate for BACE-1 inhibition research:\n"
                    f"SMILES: {row['mol']}\n"
                    f"Metrics: pIC50 {row['pIC50']:.2f}, MW {row['MW']:.2f}, AlogP {row['AlogP']:.2f}.\n"
                    f"Provide a 2-sentence summary of its pharmacological profile and potential as an Alzheimer's lead compound."
                )
            }
        ]
        # messages = [
        #     {"role": "system", "content": "You are a professional medicinal chemist."},
        #     {"role": "user", "content": (
        #         f"Analyze this BACE-1 inhibitor:\n"
        #         f"SMILES: {row['mol']}\n"
        #         f"Properties: pIC50 {row['pIC50']:.2f}, MW {row['MW']:.2f}, AlogP {row['AlogP']:.2f}.\n"
        #         f"Provide a 2-sentence summary of its drug-likeness and biological potential."
        #     )}
        # ]

        # Generate Description
        output = generator(
            messages,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        description = output[0]['generated_text'][-1]['content'].strip()

        #  Save Output
        results.append({
            "CID": cid,
            "smiles": row['mol'],
            "pIC50": row['pIC50'],
            "description": description
        })

        # --- Save file Regularly (every 20 results) ---
        if len(results) % 20 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    # --- final save ---
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    total_time = time.time() - start_time
    print(f"\n[OK] Mission Accomplished！Total Time Spent: {total_time/3600:.2f} hr")

if __name__ == "__main__":
    main()