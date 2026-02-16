import json
import requests
import pandas as pd
import os
import time
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# ==========================================
# 0. 配置區
# ==========================================
INPUT_CSV = "BACE-1_dataset/bace.csv"
OUTPUT_DIR = "output/parquet_dataset"
CHECKPOINT_FILE = "output/processed_indices.txt"
TARGET_MODEL = "aiasistentworld/Llama-3.1-8B-Instruct-STO-Master"
SAVE_INTERVAL = 50
TIMEOUT_SECONDS = 120  # 保持 120 秒

os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. 功能函數
# ==========================================

def extract_facts(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "Nitro_Groups": smiles.count("N(=O)=O") + smiles.count("[N+](=O)[O-]")
    }


def generate_text_via_ollama(smiles, facts):
    prompt = (
        f"Expert Chemist: Analyze {smiles} "
        f"(MW:{facts['MW']}, LogP:{facts['LogP']}, Nitro:{facts['Nitro_Groups']}). "
        f"Summarize energetic traits in 2 sentences."
    )

    url = "http://localhost:11434/api/generate"
    payload = {"model": TARGET_MODEL, "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        text = response.json().get("response", "").strip()

        if not text:
            return None

        return text

    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return None


def save_batch_to_parquet(batch_records):
    if not batch_records:
        return

    df_batch = pd.DataFrame(batch_records)
    table = pa.Table.from_pandas(df_batch)

    pq.write_to_dataset(
        table,
        root_path=OUTPUT_DIR,
        compression="zstd"
    )


# ==========================================
# 2. 主程序
# ==========================================

if __name__ == "__main__":

    df = pd.read_csv(INPUT_CSV)
    smiles_col = df.columns[0]

    print("[*] 初始化 Embedding 模型 (BGE-M3)...")
    embed_model = SentenceTransformer("BAAI/bge-m3")

    # 讀取 checkpoint
    processed_indices = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            processed_indices = {
                int(line.strip())
                for line in f
                if line.strip().isdigit()
            }

    print(f"[*] 總數: {len(df)} 筆，已完成: {len(processed_indices)} 筆")

    batch_records = []
    batch_descriptions = []
    batch_metadata = []

    for index in tqdm(range(len(df)), total=len(df), desc="Molecular AI Pipeline"):

        if index in processed_indices:
            continue

        row = df.iloc[index]
        smiles = row[smiles_col]

        facts = extract_facts(smiles)
        if not facts:
            continue

        description = generate_text_via_ollama(smiles, facts)

        # 不 embed error 或空值
        if not description:
            continue

        batch_descriptions.append(description)
        batch_metadata.append({
            "index": index,
            "CID": row.get("CID", index),
            "pIC50": row.get("pIC50", 0),
            "description": description
        })

        # 到達 SAVE_INTERVAL 批次 embedding
        if len(batch_descriptions) >= SAVE_INTERVAL:

            vectors = embed_model.encode(
                batch_descriptions,
                batch_size=16,
                show_progress_bar=False
            )

            for meta, vec in zip(batch_metadata, vectors):
                meta["embedding"] = vec.tolist()
                batch_records.append(meta)

            save_batch_to_parquet(batch_records)

            # 寫入 checkpoint
            with open(CHECKPOINT_FILE, "a") as f:
                for meta in batch_metadata:
                    f.write(f"{meta['index']}\n")

            # 清空 batch
            batch_records = []
            batch_descriptions = []
            batch_metadata = []

        time.sleep(0.05)

    # 最後殘留 batch
    if batch_descriptions:

        vectors = embed_model.encode(
            batch_descriptions,
            batch_size=16,
            show_progress_bar=False
        )

        for meta, vec in zip(batch_metadata, vectors):
            meta["embedding"] = vec.tolist()
            batch_records.append(meta)

        save_batch_to_parquet(batch_records)

        with open(CHECKPOINT_FILE, "a") as f:
            for meta in batch_metadata:
                f.write(f"{meta['index']}\n")

    print(f"\n[✔] 任務完成！資料已安全寫入 {OUTPUT_DIR}")
