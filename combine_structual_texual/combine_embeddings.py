import json
import os

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
STRUCTURAL_JSON = "/home/yl9210a-hpc/chem_project/chembret_cluster/output_chembret/chembret_bace_embeddings_with_pIC50.json"
TEXTUAL_JSON = "/home/yl9210a-hpc/chem_project/llama_cluster/0.output/final_bace_with_embeddings.json"
OUTPUT_JSON = "/home/yl9210a-hpc/chem_project/combine_structual_texual/combined_bace_embeddings.json"

# ---------------------------------------------------
# 1. Load structural embeddings (keyed by CID)
# ---------------------------------------------------
with open(STRUCTURAL_JSON, "r") as f:
    structural_data = json.load(f)

structural_lookup = {}
for rec in structural_data:
    cid = rec["CID"]
    structural_lookup[cid] = {
        "smiles": rec["smiles"],
        "pIC50": rec["pIC50"],
        "structural_embedding": rec["embedding"],
    }

print(f"[INFO] Loaded {len(structural_lookup)} structural embeddings", flush=True)

# ---------------------------------------------------
# 2. Load textual embeddings (keyed by CID)
# ---------------------------------------------------
with open(TEXTUAL_JSON, "r") as f:
    textual_data = json.load(f)

textual_lookup = {}
for rec in textual_data:
    cid = rec["CID"]
    textual_lookup[cid] = {
        "textual_embedding": rec["embedding"],
    }

print(f"[INFO] Loaded {len(textual_lookup)} textual embeddings", flush=True)

# ---------------------------------------------------
# 3. Merge by CID
# ---------------------------------------------------
all_cids = set(structural_lookup.keys()) & set(textual_lookup.keys())
missing_structural = set(textual_lookup.keys()) - set(structural_lookup.keys())
missing_textual = set(structural_lookup.keys()) - set(textual_lookup.keys())

if missing_structural:
    print(f"[WARN] {len(missing_structural)} CIDs in textual but not in structural", flush=True)
if missing_textual:
    print(f"[WARN] {len(missing_textual)} CIDs in structural but not in textual", flush=True)

combined = []
for cid in sorted(all_cids):
    s = structural_lookup[cid]
    t = textual_lookup[cid]

    textual_emb = t["textual_embedding"]
    structural_emb = s["structural_embedding"]
    combined_emb = textual_emb + structural_emb  # concatenate lists

    combined.append({
        "smiles": s["smiles"],
        "CID": cid,
        "pIC50": s["pIC50"],
        "textual_embedding": textual_emb,
        "structural_embedding": structural_emb,
        "combined_embedding": combined_emb,
    })

print(f"[INFO] Matched {len(combined)} molecules by CID", flush=True)

# ---------------------------------------------------
# 4. Write output
# ---------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

with open(OUTPUT_JSON, "w") as f:
    json.dump(combined, f)

print(f"[DONE] Wrote {len(combined)} records to {OUTPUT_JSON}", flush=True)
print(f"       Textual embedding dim:    {len(combined[0]['textual_embedding'])}", flush=True)
print(f"       Structural embedding dim: {len(combined[0]['structural_embedding'])}", flush=True)
print(f"       Combined embedding dim:   {len(combined[0]['combined_embedding'])}", flush=True)
