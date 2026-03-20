import os
import json
import torch
from transformers import AutoTokenizer, AutoModel

# Hugging Face path (only download once)
model_path = "deepchem/ChemBERTa-100M-MLM"

# init
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# test on one SMILES
smiles = "C1CCCCC1"
inputs = tokenizer(smiles, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Getting Hidden State
last_hidden_state = outputs.last_hidden_state
cls_embedding = last_hidden_state[0, 0, :].numpy()

print(f"ChemBERTa vector dim: {cls_embedding.shape}")

# ---------------------------------------------------
# Save output
# ---------------------------------------------------
output_dir = "/home/yl9210a-hpc/chem_project/chembret_cluster/output_chembret"
os.makedirs(output_dir, exist_ok=True)

result = {
    "smiles": smiles,
    "structural_embedding": cls_embedding.tolist()
}

output_path = os.path.join(output_dir, "chemberta_test_output.json")

with open(output_path, "w") as f:
    json.dump([result], f, indent=2)

print(f"Saved to: {output_path}")
