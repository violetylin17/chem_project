import json
import glob
import subprocess
import os

def merge_json_parts():
    combined_results = []
    # 建議使用絕對路徑，確保 PBS 執行時不會找不到檔案
    files = sorted(glob.glob("/home/yl9210a-hpc/chem_project/llama/output/llama_part_*.json"))
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            combined_results.extend(data)
            print(f"[*] Merged {file} ({len(data)} records)")

    output_name = "/home/yl9210a-hpc/chem_project/llama/output/final_bace_descriptions.json"
    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n[OK] Success! Total {len(combined_results)} molecules saved to {output_name}")

if __name__ == "__main__":
    # 1. 先跑合併
    merge_json_parts()
    print("[*] All parts combined successfully into final_bace_descriptions.json")

    # 2. 合併完後才跑 Embedding
    embedding_script = "/home/yl9210a-hpc/chem_project/llama/2.embedding.py"

    if os.path.exists(embedding_script):
        print(f"[*] Starting Embedding process: {embedding_script}")
        # 使用 subprocess 呼叫下一個腳本
        result = subprocess.run(["python", embedding_script], capture_output=True, text=True)
        
        # 將 embedding 的 log 也印出來
        print(result.stdout)
        if result.stderr:
            print("[!] Embedding Error Log:", result.stderr)
    else:
        print(f"[!] Error: Cannot find {embedding_script}")