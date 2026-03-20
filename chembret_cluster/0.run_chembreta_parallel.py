import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import sys

# -----------------------------
# 1. Read arguments
# -----------------------------
part_id = int(sys.argv[1])   # 1,2,3,4
total_parts = int(sys.argv[2])  # 4

# -----------------------------
# 2. Load dataset
# -----------------------------
input_path = "/home/yl9210a-hpc/chem_project/llama_cluster/0.output/final_bace_descriptions.json"
df = pd.read_json(input_path)

N = len(df)
chunk_size = N // total_parts
start = (part_id - 1) * chunk_size
end = N if part_id == total_parts else part_id * chunk_size

print(f"[INFO] Part {part_id}/{total_parts}: processing rows {start} to {end}")

# -----------------------------
# 3. Init ChemBERTa
# -----------------------------
model_path = "deepchem/ChemBERTa-100M-MLM"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# -----------------------------
# 4. Output folder
# -----------------------------
output_dir = "/home/yl9210a-hpc/chem_project/chembret_cluster/output_chembret"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"chembreta_part{part_id}.json")

results = []

# -----------------------------
# 5. Loop through assigned rows
# -----------------------------
for i in range(start, end):
    smiles = df.loc[i, "CID"] if "CID" in df.columns else df.loc[i, "smiles"]
    pIC50 = float(df.loc[i, "pIC50"])

    inputs = tokenizer(smiles, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()

    results.append({
        "smiles": smiles,
        "pIC50": pIC50,
        "structural_embedding": cls_embedding.tolist()
    })

    if (i - start) % 100 == 0:
        print(f"[INFO] Part {part_id}: processed {i - start} molecules")

# -----------------------------
# 6. Save output
# -----------------------------
with open(output_file, "w") as f:
    json.dump(results, f)

print(f"[DONE] Part {part_id} saved to {output_file}")
