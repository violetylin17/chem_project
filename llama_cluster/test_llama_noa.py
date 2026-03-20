import pandas as pd
import json
import os
import time
import torch
from transformers import pipeline
from tqdm import tqdm

# ==========================================
# 1. 核心與環境優化
# ==========================================
torch.set_num_threads(128)

CSV_FILE = "../BACE-1_dataset/bace.csv"  
OUTPUT_FILE = "output/llama_8b_test_results.json"
# 使用無需申請權限的 Llama 3.1 8B 備份版本，或改回官方路徑
MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B-Instruct" 

os.makedirs("output", exist_ok=True)

# ==========================================
# 2. 載入 8B 模型 (針對 CPU 節點優化)
# ==========================================
print(f"[*] Loading {MODEL_NAME} (approx. 15GB)...")

generator = pipeline(
    "text-generation", 
    model=MODEL_NAME, 
    device=-1,        # 強制使用 CPU，避開 device_map 的複雜報錯
    torch_dtype=torch.bfloat16 # 依然建議用 bfloat16 加速
)

# ==========================================
# 3. 處理程序
# ==========================================
def main():
    df = pd.read_csv(CSV_FILE).head(10) # 測試前 10 筆
    results = []
    
    print(f"[*] Starting 8B Inference Test...")
    start_time = time.time()

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # 建立結構化指令
        messages = [
            {"role": "system", "content": "You are a professional medicinal chemist."},
            {"role": "user", "content": (
                f"Analyze this BACE-1 inhibitor:\n"
                f"SMILES: {row['mol']}\n"
                f"Properties: pIC50 {row['pIC50']:.2f}, MW {row['MW']:.2f}, AlogP {row['AlogP']:.2f}.\n"
                f"Provide a 2-sentence summary of its drug-likeness and biological potential."
            )}
        ]

        # 呼叫 Llama 3.1 的生成
        output = generator(
            messages,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        # 取得助理的回覆
        description = output[0]['generated_text'][-1]['content'].strip()

        results.append({
            "index": index,
            "CID": str(row['CID']),
            "smiles": row['mol'],
            "description": description
        })

    # 儲存測試結果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\n" + "="*30)
    print(f"[*] 8B 模型測試完成！")
    print(f"[*] 平均每筆時間: {elapsed/len(results):.2f} 秒")
    print(f"[*] 預計全量 (1516筆) 所需時間: {(elapsed/len(results) * 1516 / 3600):.2f} 小時")
    print("="*30)

if __name__ == "__main__":
    main()