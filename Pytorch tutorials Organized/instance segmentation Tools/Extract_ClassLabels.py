import os
import json

# Set input folder with JSON files
json_dir = "./Strawberry Disease Detection Dataset/train"

unique_labels = set()

for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        with open(os.path.join(json_dir, filename), encoding="utf-8") as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            label = shape["label"].strip()
            unique_labels.add(label)

print("Unique labels found:")
for label in sorted(unique_labels):
    print(f"- {label}")