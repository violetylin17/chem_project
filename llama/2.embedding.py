import json
import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# path
BASE_DIR = "/home/yl9210a-hpc/chem_project/llama"
INPUT_JSON = f"{BASE_DIR}/output/final_bace_descriptions.json" 
OUTPUT_JSON = f"{BASE_DIR}/output/final_bace_with_embeddings.json"

def run_embedding():
    # read combine files
    if not os.path.exists(INPUT_JSON):
        print(f"[!] Can't find file: {INPUT_JSON}")
        return

    print(f"[*] Loading data from JSON ...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. initialize BGE-M3 Model
    # Use GPU if possible, if not will be run on CPU 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing BGE-M3 Model (Device: {device})...")
    model = SentenceTransformer('BAAI/bge-m3', device=device)

    # If running on CPU set the core number
    if device == "cpu":
        torch.set_num_threads(128)

    # 3. extract all the descriptions at once an process in batch
    descriptions = [item['description'] for item in data]
    
    print(f"[*] Generating Embeddings (Total: {len(descriptions)} molecules)...")
    
    # batch_size set as 16-32 adjust based on memory size
    embeddings = model.encode(
        descriptions, 
        batch_size=16, 
        show_progress_bar=True, 
        normalize_embeddings=True
    )

    # 4. combine the embeddings with the original file
    print("[*] Combining Results...")
    for i, embedding in enumerate(embeddings):
        data[i]['embedding'] = embedding.tolist()

    # 5. Save final output
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"[OK] Embedding Process Success! Output saved in: {OUTPUT_JSON}")

if __name__ == "__main__":
    run_embedding()