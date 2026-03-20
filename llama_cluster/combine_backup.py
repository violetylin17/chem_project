import json
import glob

def merge_json_parts():
    combined_results = []
    files = sorted(glob.glob("output/llama_part_*.json"))
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            combined_results.extend(data)
            print(f"[*] Merged {file} ({len(data)} records)")

    output_name = "output/final_bace_descriptions.json"
    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n[OK] Success! Total {len(combined_results)} molecules saved to {output_name}")



if __name__ == "__main__":
    merge_json_parts()