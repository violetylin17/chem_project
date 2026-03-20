import json, glob
merged = []
for f in glob.glob("output/llama_part_*.json"):
    with open(f, "r") as infile:
        merged.extend(json.load(infile))
with open("output/final_all_results.json", "w") as outfile:
    json.dump(merged, outfile, indent=4)