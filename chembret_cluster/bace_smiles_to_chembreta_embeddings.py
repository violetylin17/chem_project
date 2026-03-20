import os
import json
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# Hugging Face path (only download once)
MODEL_PATH = "deepchem/ChemBERTa-100M-MLM"

# Input / Output
INPUT_CSV = "/home/yl9210a-hpc/chem_project/BACE-1_dataset/bace.csv"
SMILES_COL = "mol"
OUTPUT_JSON = "/home/yl9210a-hpc/chem_project/chembret_cluster/output_chembret/chembret_bace_embeddings.json"


def _iter_smiles_from_csv(csv_path: str, smiles_col: str):
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise KeyError(
            f"Column '{smiles_col}' not found in {csv_path}. Available columns: {list(df.columns)}"
        )

    # Keep original order, drop NaNs
    for s in df[smiles_col].dropna().astype(str).tolist():
        s = s.strip()
        if s:
            yield s


def _get_shard_bounds(n_items: int, part_id: int, total_parts: int):
    if total_parts <= 0:
        raise ValueError("total_parts must be >= 1")
    if not (1 <= part_id <= total_parts):
        raise ValueError("part_id must be in [1, total_parts]")

    chunk_size = n_items // total_parts
    start = (part_id - 1) * chunk_size
    end = n_items if part_id == total_parts else part_id * chunk_size
    return start, end


def _part_output_path(final_output_json: str, part_id: int):
    base, ext = os.path.splitext(final_output_json)
    if not ext:
        ext = ".json"
    return f"{base}.part{part_id}{ext}"


def combine_parts(final_output_json: str, total_parts: int):
    all_results = []
    missing = []
    for part_id in range(1, total_parts + 1):
        part_path = _part_output_path(final_output_json, part_id)
        if not os.path.exists(part_path):
            missing.append(part_path)
            continue
        with open(part_path, "r") as f:
            data = json.load(f)
            all_results.extend(data)

    if missing:
        missing_str = "\n".join(missing)
        raise FileNotFoundError(
            "Missing part file(s). Combine aborted. Missing:\n" + missing_str
        )

    os.makedirs(os.path.dirname(final_output_json), exist_ok=True)
    with open(final_output_json, "w") as f:
        json.dump(all_results, f)

    print(f"[DONE] Combined {len(all_results)} molecules -> {final_output_json}", flush=True)


def main():
    # Usage:
    # - Full run: python bace_smiles_to_chembreta_embeddings.py
    # - Sharded:  python bace_smiles_to_chembreta_embeddings.py <part_id> <total_parts>
    # - Combine:  python bace_smiles_to_chembreta_embeddings.py --combine <total_parts>
    if len(sys.argv) >= 2 and sys.argv[1] == "--combine":
        if len(sys.argv) != 3:
            raise SystemExit("Usage: --combine <total_parts>")
        combine_parts(OUTPUT_JSON, int(sys.argv[2]))
        return

    part_id = None
    total_parts = None
    if len(sys.argv) == 3:
        part_id = int(sys.argv[1])
        total_parts = int(sys.argv[2])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    smiles_list = list(_iter_smiles_from_csv(INPUT_CSV, SMILES_COL))

    start = 0
    end = len(smiles_list)
    output_path = OUTPUT_JSON
    if part_id is not None and total_parts is not None:
        start, end = _get_shard_bounds(len(smiles_list), part_id, total_parts)
        output_path = _part_output_path(OUTPUT_JSON, part_id)
        print(f"[INFO] Part {part_id}/{total_parts}: processing rows {start} to {end}", flush=True)

    results = []
    for i in range(start, end):
        smiles = smiles_list[i]
        inputs = tokenizer(smiles, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[0, 0, :].cpu().numpy().tolist()

        results.append({
            "smiles": smiles,
            "embedding": cls_embedding,
        })

        if (i - start) % 100 == 0:
            print(f"[INFO] processed {i - start} molecules", flush=True)

    with open(output_path, "w") as f:
        json.dump(results, f)

    print(f"Wrote {len(results)} embeddings to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
