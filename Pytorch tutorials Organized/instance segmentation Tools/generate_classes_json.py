import os
import json

def extract_unique_labels(json_folder):
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
    class_mapping = {label: idx for idx, label in enumerate(labels)}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=4)

    print(f"Saved {len(class_mapping)} classes to '{output_file}'")


if __name__ == "__main__":
    # Set input folder containing JSON annotation files
    json_folder = "./Strawberry Disease Detection Dataset/train"

    print("Extracting unique labels from JSON files...")
    unique_labels = extract_unique_labels(json_folder)

    if not unique_labels:
        print("No labels found in JSON files.")
    else:
        print("Unique labels found:")
        for label in unique_labels:
            print(f"- {label}")

        save_classes_json(unique_labels)