"""
Prepare VisDrone dataset for YOLO training with PEOPLE class only.
This script:
1. Reads raw VisDrone annotations
2. Converts to YOLO format
3. Filters for only 'people' class (VisDrone class 2)
4. Creates data.yaml for training
"""
import os
import shutil
from pathlib import Path
import yaml
from PIL import Image
from tqdm import tqdm

# Configuration
DATA_ROOT = Path("data")
VISDRONE_ROOT = DATA_ROOT / "visdrone"
OUTPUT_DIR = DATA_ROOT / "visdrone_person"

# VisDrone class 2 = 'people' (we exclude class 1 = 'pedestrian')
TARGET_CLASS_ID = 2  # people only
OUTPUT_CLASS_ID = 0  # remap to single class 'person'


def convert_visdrone_box(img_size, box):
    """Convert VisDrone box (x,y,w,h) to YOLO format (x_center, y_center, w, h) normalized."""
    dw = 1.0 / img_size[0]
    dh = 1.0 / img_size[1]
    x, y, w, h = box
    x_center = (x + w / 2) * dw
    y_center = (y + h / 2) * dh
    w_norm = w * dw
    h_norm = h * dh
    # Clamp to [0, 1]
    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, w_norm)),
        max(0.0, min(1.0, h_norm)),
    )


def process_subset(subset_name, output_split):
    """
    Process a VisDrone subset (e.g., 'VisDrone2019-DET-train') → output split ('train' or 'val').
    """
    subset_dir = VISDRONE_ROOT / subset_name
    img_dir = subset_dir / "images"
    ann_dir = subset_dir / "annotations"

    if not img_dir.exists() or not ann_dir.exists():
        print(f"[SKIP] {subset_name} not found at {subset_dir}")
        return 0

    # Create output directories
    out_img_dir = OUTPUT_DIR / "images" / output_split
    out_lbl_dir = OUTPUT_DIR / "labels" / output_split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(img_dir.glob("*.jpg"))
    processed = 0

    for img_path in tqdm(image_files, desc=f"Processing {subset_name}"):
        ann_path = ann_dir / f"{img_path.stem}.txt"

        if not ann_path.exists():
            continue

        # Get image dimensions
        try:
            with Image.open(img_path) as img:
                img_size = img.size  # (width, height)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            continue

        # Parse annotations and filter for people class only
        yolo_labels = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue

                try:
                    # VisDrone format: x,y,w,h,score,category,truncation,occlusion
                    score = int(parts[4])
                    category = int(parts[5])

                    # Skip ignored regions (score=0) and non-target classes
                    if score == 0 or category != TARGET_CLASS_ID:
                        continue

                    box = tuple(map(int, parts[:4]))
                    yolo_box = convert_visdrone_box(img_size, box)
                    yolo_labels.append(f"{OUTPUT_CLASS_ID} {' '.join(f'{x:.6f}' for x in yolo_box)}")
                except (ValueError, IndexError):
                    continue

        # Only copy image if it has people annotations
        if yolo_labels:
            shutil.copy2(img_path, out_img_dir / img_path.name)
            with open(out_lbl_dir / f"{img_path.stem}.txt", "w") as f:
                f.write("\n".join(yolo_labels))
            processed += 1

    return processed


def create_data_yaml():
    """Create data.yaml for YOLO training."""
    yaml_content = {
        "path": str(OUTPUT_DIR.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["person"],
    }

    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"\n✅ Created: {yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("VisDrone → YOLO (People Only) Dataset Preparation")
    print("=" * 60)
    print(f"Target class: 'people' (VisDrone class {TARGET_CLASS_ID})")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print()

    # Process train and val subsets
    train_count = process_subset("VisDrone2019-DET-train", "train")
    val_count = process_subset("VisDrone2019-DET-val", "val")

    if train_count == 0 and val_count == 0:
        print("\n❌ No data processed!")
        print("Please download VisDrone first: python scripts/download_visdrone.py")
        return

    # Create data.yaml
    yaml_path = create_data_yaml()

    print(f"\n📊 Summary:")
    print(f"   Train images: {train_count}")
    print(f"   Val images:   {val_count}")
    print(f"\n🚀 Ready to train:")
    print(f"   python scripts/train_yolo.py")


if __name__ == "__main__":
    main()
