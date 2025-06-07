import os
import json
import argparse

def extract_unique_labels(json_folder):
    """Extract all unique label names from LabelMe-style JSON files."""
    unique_labels = set()

    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(json_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                shapes = data.get("shapes", [])
                for shape in shapes:
                    label = shape.get("label", "").strip()
                    if label:
                        unique_labels.add(label)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return sorted(unique_labels)


def save_classes_json(labels, output_file="classes.json"):
    """Save class mapping to JSON file."""
    class_mapping = {label: idx for idx, label in enumerate(labels)}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=4)

    print(f"✅ Saved {len(class_mapping)} classes to '{output_file}'")
    return class_mapping


def main():
    parser = argparse.ArgumentParser(description="Generate classes.json from LabelMe JSON annotations.")
    parser.add_argument("--input", type=str, required=True,
                        help="Folder containing LabelMe-style JSON annotation files")
    parser.add_argument("--output", type=str, default="classes.json",
                        help="Path to save generated classes.json")

    args = parser.parse_args()

    print("🔍 Scanning JSON files for unique labels...")
    unique_labels = extract_unique_labels(args.input)

    if not unique_labels:
        print("❌ No labels found in JSON files.")
        return

    print("\n📋 Unique labels found:")
    for label in unique_labels:
        print(f"- {label}")

    save_classes_json(unique_labels, args.output)


if __name__ == "__main__":
    main()