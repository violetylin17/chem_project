import pandas as pd
import json
import os
import torch
import sys
from transformers import pipeline
from tqdm import tqdm

# 1. Retrieve parallel execution parameters (passed in by PBS/LSF)
# If the input is 1, 2, 3, 4, convert them to 0, 1, 2, 3
raw_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
part_id = raw_id - 1
total_parts = int(sys.argv[2]) if len(sys.argv) > 2 else 4

# 2. CPU optimization: four parallel tasks, each using 32 threads
torch.set_num_threads(32)

# Paths for this parallel run
CSV_FILE = "/home/yl9210a-hpc/chem_project/BACE-1_dataset/bace.csv"
OUTPUT_FILE = f"/home/yl9210a-hpc/chem_project/llama/output/llama_part_{part_id}.json"
MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B-Instruct"

os.makedirs("output", exist_ok=True)

# 3. Load the model
print(f"[*] Task {raw_id} loading model...")
generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device=-1,
    torch_dtype=torch.bfloat16
)

def main():
    df_all = pd.read_csv(CSV_FILE)

    # --- Test mode: uncomment the next line to process only the first 4 rows ---
    # df_all = df_all.head(4)

    # ⭐ Data slicing logic
    chunk_size = len(df_all) // total_parts
    start_idx = part_id * chunk_size
    # The last part takes everything until the end
    end_idx = (part_id + 1) * chunk_size if part_id < total_parts - 1 else len(df_all)

    df = df_all.iloc[start_idx:end_idx]
    print(f"[*] Part {part_id}: processing rows {start_idx} to {end_idx} (total {len(df)} entries)")

    results = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
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

        # Error handling to prevent a single failure from crashing the entire part
        try:
            output = generator(
                messages,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.6,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            description = output[0]["generated_text"][-1]["content"].strip()
        except Exception as e:
            description = f"Error processing: {str(e)}"

        results.append({
            "CID": str(row["CID"]),
            "pIC50": row["pIC50"],
            "description": description
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"[*] Part {part_id} completed.")

if __name__ == "__main__":
    main()
