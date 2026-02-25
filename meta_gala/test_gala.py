import pandas as pd
import json
import os
import time
import torch
from transformers import pipeline
from tqdm import tqdm

# ==========================================
# 1. 設定與核心優化
# ==========================================
torch.set_num_threads(128)

CSV_FILE = "../BACE-1_dataset/bace.csv"  
OUTPUT_FILE = "output/galactica_test_results.json"
MODEL_NAME = "facebook/galactica-1.3b" 
TEST_COUNT = 10 

os.makedirs("output", exist_ok=True)

# ==========================================
# 2. 載入模型
# ==========================================
print(f"[*] Loading {MODEL_NAME} to CPU...")
generator = pipeline(
    "text-generation", 
    model=MODEL_NAME, 
    device=-1, 
    torch_dtype=torch.float32
)

# ==========================================
# 3. 處理程序
# ==========================================
def main():
    df_all = pd.read_csv(CSV_FILE)
    df = df_all.head(TEST_COUNT)
    
    results = []
    print(f"[*] Starting TEST inference for {len(df)} molecules...")
    start_time = time.time()

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # 提取資訊
        smiles = row['mol']
        cid = str(row['CID'])
        pIC50 = row['pIC50']
        mw = row['MW']
        logp = row['AlogP']
        psa = row['PSA']
        
        # 定義引導句
        lead_in = "The molecule contains"
        
        prompt = (
            f"Structure: [START_I_SMILES]{smiles}[END_I_SMILES]\n"
            f"Data: MW {mw:.2f} Da, LogP {logp:.2f}, PSA {psa:.2f} Å², pIC50 {pIC50:.2f}.\n"
            f"Analyze the chemical structure and pharmacological profile in detail.\n"
            f"Analysis: {lead_in}" 
        )

        output = generator(
            prompt, 
            max_new_tokens=120, 
            do_sample=True,     
            temperature=0.7,    
            top_p=0.9,
            repetition_penalty=1.2, 
            clean_up_tokenization_spaces=True
        )
        
        full_text = output[0]['generated_text']
        
        # ⭐ 修正切分邏輯：取得 "Analysis:" 之後的所有內容
        generated_part = full_text.split("Analysis:")[-1].strip()
        
        # 確保描述包含完整句意
        description = generated_part if generated_part.startswith(lead_in) else f"{lead_in} {generated_part}"

        results.append({
            "index": index,
            "CID": cid,
            "smiles": smiles,
            "pIC50": pIC50,
            "description": description
        })

    # 儲存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\n[OK] 測試完成！平均每筆時間: {elapsed/len(results):.2f} 秒")

if __name__ == "__main__":
    main()