import json
import os

output_dir = "/home/yl9210a-hpc/chem_project/chembret_cluster/output_chembret"

all_results = []

for part in range(1, 5):
    file_path = os.path.join(output_dir, f"chembreta_part{part}.json")
    with open(file_path, "r") as f:
        data = json.load(f)
        all_results.extend(data)

final_path = os.path.join(output_dir, "chemberta_embeddings.json")

with open(final_path, "w") as f:
    json.dump(all_results, f)

print(f"[DONE] Combined {len(all_results)} molecules → {final_path}")
