import pandas as pd
import torch
import json
import os
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Newest opensource ChemBERTa-100M-MLM model Aug 2025
MODEL_NAME = "deepchem/ChemBERTa-100M-MLM"
INPUT_CSV = "/home/yl9210a-hpc/chem_project/BACE-1_dataset/bace.csv"
BASE_OUTPUT_DIR = "/home/yl9210a-hpc/chem_project/llama_cluster/Chemberta_output/"
FILE_NAME = "chemberta_embeddings.json"

OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, FILE_NAME)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

def get_chemberta_embeddings():
    df = pd.read_csv(INPUT_CSV)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    results = []
    print(f"[*] 正在提取 ChemBERTa 特徵 (總計 {len(df)} 筆)...")

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            inputs = tokenizer(row['mol'], return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            
            # 取得 [CLS] token 的向量作為全分子表示 (通常是 384 維)
            embeddings = outputs.last_hidden_state[:, 0, :].flatten().tolist()
            
            results.append({
                "CID": row['CID'],
                "chemberta_embedding": embeddings
            })

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f)
    print(f"[OK] ChemBERTa 特徵已存至: {OUTPUT_JSON}")

if __name__ == "__main__":
    get_chemberta_embeddings()