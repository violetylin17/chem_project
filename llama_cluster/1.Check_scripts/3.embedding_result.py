import json
import numpy as np

FILE_PATH = "/home/yl9210a-hpc/chem_project/llama/output/final_bace_with_embeddings.json"

def check_json():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    first_entry = data[0]
    
    print(f"[*] Total number of mol: {total}")
    print(f"[*] Headings: {list(first_entry.keys())}")
    
    if 'embedding' in first_entry:
        emb_shape = np.array(first_entry['embedding']).shape
        print(f"[*] Embedding Demention: {emb_shape}") # should be (1024,)
    
    # Check if there are any missing embedding
    missing = [item['CID'] for item in data if 'embedding' not in item]
    if not missing:
        print("[OK] All small molecule has successfully generated their description embedding！")
    else:
        print(f"[!] Warning：It's lacking {len(missing)} embeddings。")

if __name__ == "__main__":
    check_json()