import json
import sys
from pathlib import Path

def preview_value(v, max_list=5, max_str=40):
    if isinstance(v, list):
        return v[:max_list]
    if isinstance(v, str) and len(v) > max_str:
        return v[:max_str] + "..."
    return v

def main():
    if len(sys.argv) < 2:
        print("Usage: python json.py <file.json>")
        return

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    # If JSON is a list, take the first item
    if isinstance(data, list):
        item = data[0]
    # If JSON is an object, take the first key
    elif isinstance(data, dict):
        first_key = next(iter(data))
        item = data[first_key]
    else:
        print("Unsupported JSON structure")
        return

    print("\n=== Keys and Preview ===")
    for key, value in item.items():
        print(f"{key}: {preview_value(value)}")

if __name__ == "__main__":
    main()
