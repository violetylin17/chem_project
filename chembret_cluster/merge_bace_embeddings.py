import json
import pandas as pd

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BACE_CSV = "/home/yl9210a-hpc/chem_project/BACE-1_dataset/bace.csv"
EMBEDDING_JSON = "/home/yl9210a-hpc/chem_project/chembret_cluster/output_chembret/chembret_bace_embeddings.json"
OUTPUT_JSON = "/home/yl9210a-hpc/chem_project/chembret_cluster/output_chembret/chembret_bace_embeddings_with_pIC50.json"

# ---------------------------------------------------
# 1. Read bace.csv: extract mol, CID, pIC50
# ---------------------------------------------------
df_bace = pd.read_csv(BACE_CSV, usecols=["mol", "CID", "pIC50"])
# Build a lookup dict: smiles -> {CID, pIC50}
bace_lookup = {}
for _, row in df_bace.iterrows():
    smi = str(row["mol"]).strip()
    bace_lookup[smi] = {
        "CID": row["CID"],
        "pIC50": float(row["pIC50"]),
    }

print(f"[INFO] Loaded {len(bace_lookup)} molecules from bace.csv", flush=True)

# ---------------------------------------------------
# 2. Read embedding JSON
# ---------------------------------------------------
with open(EMBEDDING_JSON, "r") as f:
    embeddings = json.load(f)

print(f"[INFO] Loaded {len(embeddings)} embeddings from JSON", flush=True)

# ---------------------------------------------------
# 3. Merge by matching SMILES
# ---------------------------------------------------
merged = []
not_found = []

for rec in embeddings:
    smi = rec["smiles"].strip()
    if smi in bace_lookup:
        merged.append({
            "smiles": smi,
            "CID": bace_lookup[smi]["CID"],
            "pIC50": bace_lookup[smi]["pIC50"],
            "embedding": rec["embedding"],
        })
    else:
        not_found.append(smi)

print(f"[INFO] Matched: {len(merged)} molecules", flush=True)
if not_found:
    print(f"[WARN] {len(not_found)} SMILES in embeddings not found in bace.csv:", flush=True)
    for s in not_found[:5]:
        print(f"       {s}", flush=True)

# ---------------------------------------------------
# 4. Write merged output
# ---------------------------------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(merged, f)

print(f"[DONE] Wrote {len(merged)} records to {OUTPUT_JSON}", flush=True)
