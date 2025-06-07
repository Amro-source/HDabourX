import os
import json
from PIL import Image, ImageDraw
import argparse


def create_mask_and_label(json_path, image_width, image_height, shapes, output_mask_folder, output_label_folder, class_mapping):
    base_name = os.path.splitext(os.path.basename(json_path))[0]

    # --- Create binary mask ---
    mask = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)

    # Draw all polygons on the mask
    for shape in shapes:
        points = [(round(x), round(y)) for x, y in shape['points']]
        if shape['shape_type'] == 'polygon':
            draw.polygon(points, fill=255)  # White filled mask

    mask.save(os.path.join(output_mask_folder, f"{base_name}_mask.png"))

    # --- Save label file (YOLO-style) ---
    label_file_content = []

    for shape in shapes:
        label = shape['label']
        if label not in class_mapping:
            print(f"Unknown label '{label}' in {json_path}, skipping...")
            continue

        class_id = class_mapping[label]
        points = shape['points']

        normalized_points = []
        for x, y in points:
            normalized_x = x / image_width
            normalized_y = image_height - y  # Optional: flip Y axis
            normalized_points.append(normalized_x)
            normalized_points.append(normalized_y)

        line = str(class_id) + " " + " ".join(f"{p:.6f}" for p in normalized_points)
        label_file_content.append(line)

    label_path = os.path.join(output_label_folder, f"{base_name}.txt")
    with open(label_path, 'w') as f:
        f.write("\n".join(label_file_content))

    print(f"Saved mask and label for {base_name}")


def main():
    parser = argparse.ArgumentParser(description="Convert LabelMe JSON annotations to binary masks and YOLO-style label files.")
    parser.add_argument("--input", type=str, required=True, help="Directory containing JSON files")
    parser.add_argument("--output_masks", type=str, default="masks", help="Folder to save binary masks")
    parser.add_argument("--output_labels", type=str, default="labels", help="Folder to save label txt files")
    parser.add_argument("--class_mapping", type=str, required=False, help="Path to class mapping JSON")

    args = parser.parse_args()

    # Default class mapping (if no file is provided)
    class_mapping = {"default": 0}

    if args.class_mapping:
        try:
            with open(args.class_mapping, 'r') as f:
                class_mapping = json.load(f)
        except Exception as e:
            print(f"Error loading class mapping: {e}")
            return

    # Create output folders
    os.makedirs(args.output_masks, exist_ok=True)
    os.makedirs(args.output_labels, exist_ok=True)

    # Process each JSON file
    for filename in os.listdir(args.input):
        if filename.lower().endswith(".json"):
            json_path = os.path.join(args.input, filename)

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                image_width = data.get("imageWidth")
                image_height = data.get("imageHeight")
                shapes = data.get("shapes", [])

                if None in [image_width, image_height]:
                    print(f"Missing image dimensions in {filename}, skipping...")
                    continue

                create_mask_and_label(json_path, image_width, image_height, shapes,
                                      args.output_masks, args.output_labels, class_mapping)

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    print("✅ Processing complete.")


if __name__ == "__main__":
    main()