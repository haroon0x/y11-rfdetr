import os
import json
import shutil
from pathlib import Path
import yaml
import pandas as pd
from tqdm import tqdm

# Configuration
DATA_ROOT = Path("data")
VISDRONE_ROOT = DATA_ROOT / "visdrone"
OUTPUT_DIR = DATA_ROOT / "visdrone_person"
TARGET_CLASS_IDS = [1, 2] # 1: pedestrian, 2: people
OUTPUT_CLASS_ID = 0 # person
MIN_ALTITUDE = 100  # meters (Note: VisDrone doesn't have altitude in annotations, this might be for other datasets or metadata)

def convert_visdrone_bbox(bbox, img_width, img_height):
    # VisDrone: x, y, w, h
    # YOLO: x_center, y_center, w, h (normalized)
    x, y, w, h = bbox
    
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [x_center, y_center, w_norm, h_norm]

def process_visdrone_subset(subset_name, output_subset):
    """
    subset_name: 'VisDrone2019-DET-train' or 'VisDrone2019-DET-val'
    output_subset: 'train' or 'val'
    """
    img_dir = VISDRONE_ROOT / subset_name / "images"
    ann_dir = VISDRONE_ROOT / subset_name / "annotations"
    
    if not img_dir.exists() or not ann_dir.exists():
        print(f"Warning: VisDrone subset {subset_name} not found at {img_dir}")
        return

    print(f"Processing {subset_name} -> {output_subset}...")
    
    # Create output directories
    (OUTPUT_DIR / "images" / output_subset).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / output_subset).mkdir(parents=True, exist_ok=True)

    image_files = list(img_dir.glob("*.jpg"))
    
    for img_path in tqdm(image_files, desc=f"Processing {subset_name}"):
        ann_path = ann_dir / f"{img_path.stem}.txt"
        
        if not ann_path.exists():
            continue
            
        # Read image dimensions (needed for normalization)
        # We can use PIL or OpenCV, but let's try to avoid heavy deps if possible.
        # However, we need dimensions. Let's assume cv2 or PIL is available.
        # Since ultralytics is installed, we likely have opencv-python or PIL.
        from PIL import Image
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue

        yolo_annotations = []
        
        with open(ann_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                
                # VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                try:
                    category = int(parts[5])
                    score = int(parts[4]) # score is usually 0 or 1 (ignored regions) in VisDrone DET? 
                    # Actually in DET, score might be 1 or 0. 0 is ignored.
                    # Let's check VisDrone docs or assume standard DET format.
                    # Usually we filter by ignored regions (score == 0) if necessary, but VisDrone 'score' in GT is often just 1 or 0 (ignored).
                    # Let's assume we keep score == 1 or just filter by category.
                    # Standard VisDrone toolkit ignores score 0.
                    
                    if category in TARGET_CLASS_IDS:
                        bbox = [float(x) for x in parts[0:4]]
                        yolo_bbox = convert_visdrone_bbox(bbox, img_width, img_height)
                        
                        # Clamp values to [0, 1] just in case
                        yolo_bbox = [max(0.0, min(1.0, x)) for x in yolo_bbox]
                        
                        yolo_annotations.append(f"{OUTPUT_CLASS_ID} {' '.join(map(str, yolo_bbox))}")
                except ValueError:
                    continue
        
        if yolo_annotations:
            # Copy image
            shutil.copy2(img_path, OUTPUT_DIR / "images" / output_subset / img_path.name)
            
            # Write label file
            label_path = OUTPUT_DIR / "labels" / output_subset / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_annotations))

def create_dataset_yaml():
    yaml_content = {
        "path": str(OUTPUT_DIR.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["person"]
    }
    
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Created dataset YAML at {OUTPUT_DIR / 'data.yaml'}")

def filter_and_merge():
    print(f"Preparing VisDrone dataset: filtering for 'person' (VisDrone class ids 1, 2)")
    
    # VisDrone
    # Assuming standard VisDrone folder structure: VisDrone2019-DET-train, VisDrone2019-DET-val
    process_visdrone_subset("VisDrone2019-DET-train", "train")
    process_visdrone_subset("VisDrone2019-DET-val", "val")
    
    # Only create YAML if output directory exists with data
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        create_dataset_yaml()
        print("Preparation complete.")
    else:
        print("\nNo data was processed. Please ensure VisDrone data exists at:")
        print(f"  {VISDRONE_ROOT / 'VisDrone2019-DET-train'}")
        print(f"  {VISDRONE_ROOT / 'VisDrone2019-DET-val'}")
        print("\nDownload VisDrone DET dataset from: http://aiskyeye.com/")

if __name__ == "__main__":
    filter_and_merge()
