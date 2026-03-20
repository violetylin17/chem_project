import json
import requests
import pandas as pd
import os
import time
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # 如果沒安裝請跑 uv add tqdm

# ==========================================
# 0. 配置區
# ==========================================
INPUT_CSV = "BACE-1_dataset/bace.csv"
OUTPUT_JSON = "output/final_full_results.json"
CHECKPOINT_FILE = "output/processed_indices.md"
TARGET_MODEL = "aiasistentworld/Llama-3.1-8B-Instruct-STO-Master"
SAVE_INTERVAL = 50  # 每 50 筆強制備份一次 JSON

os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

# ==========================================
# 1. 功能函數
# ==========================================
def extract_facts(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "Nitro_Groups": smiles.count("N(=O)=O") + smiles.count("[N+](=O)[O-]")
    }

def generate_text_via_ollama(smiles, facts):
    prompt = f"Expert Chemist: Analyze {smiles} (MW:{facts['MW']}, LogP:{facts['LogP']}, Nitro:{facts['Nitro_Groups']}). Summarize energetic traits in 2 sentences."
    url = "http://localhost:11434/api/generate"
    payload = {"model": TARGET_MODEL, "prompt": prompt, "stream": False}
    try:
        # 預熱狀態下，timeout 120s 綽綽有餘
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# 2. 主程序
# ==========================================
if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    smiles_col = df.columns[0]
    
    print("[*] 正在初始化 Embedding 模型 (BGE-M3)...")
    embed_model = SentenceTransformer('BAAI/bge-m3')

    # 讀取續傳紀錄
    processed_indices = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            processed_indices = {int(line.strip()) for line in f}

    results = []
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            results = json.load(f)

    print(f"[*] 總數: {len(df)} 筆，已完成: {len(processed_indices)} 筆，準備開始...")

    # 使用 tqdm 顯示華麗進度條
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Molecular AI Pipeline"):
        if index in processed_indices:
            continue

        smiles = row[smiles_col]
        facts = extract_facts(smiles)
        
        if facts:
            # LLM 推論
            description = generate_text_via_ollama(smiles, facts)
            # 向量化
            vector = embed_model.encode(description).tolist()

            results.append({
                "index": index,
                "CID": row.get('CID', index),
                "pIC50": row.get('pIC50', 0),
                "description": description,
                "embedding": vector
            })

            # 寫入 Checkpoint
            with open(CHECKPOINT_FILE, "a") as f:
                f.write(f"{index}\n")

            # 每隔 SAVE_INTERVAL 存一次檔，防止意外
            if len(results) % SAVE_INTERVAL == 0:
                with open(OUTPUT_JSON, "w") as f:
                    json.dump(results, f, indent=2)
        
        # 極短暫休息，讓 CPU 喘息並處理 I/O
        time.sleep(0.05)

    # 最終儲存
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✔] 任務達成！總計儲存 {len(results)} 筆資料至 {OUTPUT_JSON}")