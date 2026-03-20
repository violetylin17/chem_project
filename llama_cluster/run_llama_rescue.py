import pandas as pd
import json
import os
import torch
import sys
from transformers import pipeline
from tqdm import tqdm

# 1. 配置與參數
raw_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
part_id = raw_id - 1
total_parts = int(sys.argv[2]) if len(sys.argv) > 2 else 4

torch.set_num_threads(32) # 使用 32 核以匹配四路並行

# 路徑設定
BASE_DIR = "/home/yl9210a-hpc/chem_project/llama"
CSV_FILE = "/home/yl9210a-hpc/chem_project/BACE-1_dataset/bace.csv"
INPUT_JSON = f"{BASE_DIR}/output/llama_part_{part_id}.json"
RESCUE_OUTPUT = f"{BASE_DIR}/output/llama_part_{part_id}_rescued.json"
MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B-Instruct"

# 2. 篩選需要補跑的分子
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    current_data = json.load(f)

# 找出被拒絕回答的 CID
refused_cids = [item['CID'] for item in current_data if "I cannot " in item['description']]
print(f"[*] Task {raw_id}: 偵測到 {len(refused_cids)} 筆失敗，開始救援...")

if not refused_cids:
    print(f"[*] Task {raw_id}: 沒有偵測到失敗，跳過。")
    sys.exit(0)

# 3. 載入模型
generator = pipeline("text-generation", model=MODEL_NAME, device=-1, torch_dtype=torch.bfloat16)

def main():
    df_all = pd.read_csv(CSV_FILE)
    df_rescue = df_all[df_all['CID'].isin(refused_cids)]

    results = []
    # 讀取原本成功的資料，以便合併
    for item in current_data:
        if item['CID'] not in refused_cids:
            results.append(item)

    # 4. 補跑失敗的分子 (使用 Research-Only Prompt)
    for index, row in tqdm(df_rescue.iterrows(), total=len(df_rescue)):
        # 修改過的科學研究專用 Prompt
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

        try:
            output = generator(messages, max_new_tokens=80, do_sample=True, temperature=0.4, 
                               pad_token_id=generator.tokenizer.eos_token_id)
            description = output[0]['generated_text'][-1]['content'].strip()
        except Exception as e:
            description = f"Still failed: {str(e)}"

        results.append({
            "CID": str(row['CID']),
            "pIC50": row['pIC50'],
            "description": description
        })

    # 存檔 (覆蓋或另存)
    with open(RESCUE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"[*] Task {raw_id} rescue completed.")

if __name__ == "__main__":
    main()