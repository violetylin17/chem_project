import pandas as pd
import json
import requests
import os
import time
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 設定區塊
# ==========================================
CSV_FILE = "BACE-1_dataset/bace.csv"  
OUTPUT_FILE = "output/batch_test_output.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "aiasistentworld/Llama-3.1-8B-Instruct-STO-Master"

os.makedirs("output", exist_ok=True)

# ==========================================
# 2. Ollama 呼叫函數 (增加 Timeout 至 180s)
# ==========================================
def ask_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# 3. 主程序
# ==========================================
def main():
    start_time = time.time()   # ⭐ 開始計時

    print(f"[*] 讀取檔案: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE).head(10)
    
    print("[*] 載入 BGE-M3 Embedding 模型...")
    embed_model = SentenceTransformer('BAAI/bge-m3')
    
    final_results = []

    for index, row in df.iterrows():
        smiles = row['mol']
        cid = row['CID']
        mw = row['MW']
        alogp = row['AlogP']
        pIC50 = row['pIC50']

        print(f"\n[{index+1}/10] 正在處理 CID: {cid}")

        prompt = f"""
        Analyze this molecule:
        - SMILES: {smiles}
        - Molecular Weight: {mw}
        - AlogP: {alogp}
        - pIC50 (Activity): {pIC50}
        
        Task: Based on these chemical descriptors, summarize the drug-likeness and potential biological activity.
        Provide a professional analysis in 2 sentences.
        """

        description = ask_ollama(prompt)
        print(f"  > LLM 分析完成")

        vector = embed_model.encode(description).tolist()
        print(f"  > Embedding 生成完成")

        final_results.append({
            "CID": cid,
            "smiles": smiles,
            "description": description,
            "embedding": vector
        })

    # ==========================================
    # 正確的 JSON 儲存方式
    # ==========================================
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    # ==========================================
    # 計算總執行時間
    # ==========================================
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n[OK] 測試完成！結果已儲存至 {OUTPUT_FILE}")
    print(f"[*] 總執行時間: {elapsed:.2f} 秒")

if __name__ == "__main__":
    main()
