import os
import json
from pathlib import Path
from collections import Counter

VEDIC_FIELDS = {
    "collection", "book", "hymn", "verse", "language",
    "translation", "commentary", "source", "is_vedic"
}

def normalize_dialogue(dialogue):
    """Convert dialogue list into flattened text string."""
    if isinstance(dialogue, list):
        lines = []
        for turn in dialogue:
            role = turn.get("role", "unknown")
            content = turn.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else None
    return None


def normalize_item(item):
    """Convert raw dict/string into SFT-compatible dict, preserving metadata."""
    if isinstance(item, str):
        return {"text": item.strip(), "_type": "plain"}
    
    if isinstance(item, dict):
        result, dtype = {}, "plain"

        # Instruction-response format
        if "instruction" in item and "response" in item:
            result["instruction"] = item["instruction"].strip()
            result["response"] = item["response"].strip()
            result["text"] = f"{result['instruction']}\n{result['response']}"
            dtype = "instruction"

        # Dialogue format
        elif "dialogue" in item or "conversation" in item:
            dialogue = item.get("dialogue", item.get("conversation"))
            flattened = normalize_dialogue(dialogue)
            if flattened:
                result["dialogue"] = dialogue
                result["text"] = flattened
                dtype = "dialogue"

        # Plain text format
        elif "text" in item:
            result["text"] = str(item["text"]).strip()
            dtype = "plain"

        # Copy Vedic metadata if present
        for key in VEDIC_FIELDS:
            if key in item:
                result[key] = item[key]

        # Copy other useful fields
        for key, value in item.items():
            if key not in result and value not in [None, ""]:
                result[key] = value

        if "text" in result:
            result["_type"] = dtype
            return result

    return None


def read_file(path: Path):
    """Read txt/json/jsonl and yield normalized examples."""
    examples = []
    try:
        if path.suffix == ".txt":
            text = Path(path).read_text(encoding="utf-8").strip()
            if text:
                examples.append({"text": text, "_type": "plain"})

        elif path.suffix == ".json":
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    ex = normalize_item(item)
                    if ex: examples.append(ex)
            elif isinstance(data, dict):
                ex = normalize_item(data)
                if ex: examples.append(ex)

        elif path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        ex = normalize_item(item)
                        if ex: examples.append(ex)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")
    return examples


def save_json_parts(dataset, output_dir, output_prefix, max_per_file=None):
    """Save dataset as one or more JSON files into output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not max_per_file or len(dataset) <= max_per_file:
        out_path = Path(output_dir) / f"{output_prefix}.json"
        out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Wrote {len(dataset)} examples to {out_path}")
    else:
        total = len(dataset)
        parts = (total + max_per_file - 1) // max_per_file
        for i in range(parts):
            start, end = i * max_per_file, min((i + 1) * max_per_file, total)
            part_data = dataset[start:end]
            out_path = Path(output_dir) / f"{output_prefix}_part{i+1}.json"
            out_path.write_text(json.dumps(part_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✅ Wrote {len(part_data)} examples to {out_path}")


def build_sft_dataset(input_dir: str, output_dir: str = "./processed", max_per_file: int = None):
    dataset_main, dataset_dialogue = [], []
    type_counts, vedic_count = Counter(), 0

    for file in Path(input_dir).rglob("*"):
        if file.is_file() and file.suffix.lower() in [".txt", ".json", ".jsonl"]:
            examples = read_file(file)
            for ex in examples:
                t = ex.get("_type", "plain")
                type_counts[t] += 1
                if ex.get("is_vedic"):
                    vedic_count += 1
                if t == "dialogue":
                    dataset_dialogue.append(ex)
                else:
                    dataset_main.append(ex)

    # Save outputs into ./processed
    save_json_parts(dataset_main, output_dir, "sft_dataset", max_per_file)
    if dataset_dialogue:
        save_json_parts(dataset_dialogue, output_dir, "sft_dataset_dialogues", max_per_file)

    # Print summary
    print("\n📊 Dataset Summary:")
    for t, c in type_counts.items():
        print(f" - {t}: {c}")
    print(f" - vedic: {vedic_count}")
    print(f" - total: {len(dataset_main) + len(dataset_dialogue)}")


if __name__ == "__main__":
    # Example: process ./data into ./processed
    build_sft_dataset("./data", "./processed", max_per_file=10000)
