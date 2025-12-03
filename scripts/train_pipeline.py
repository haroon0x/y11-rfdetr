import os
import subprocess
from ultralytics import YOLO

def run_training_step(dataset_yaml, model_weights, project_name, epochs=50, imgsz=1280):
    print(f"Starting training on {dataset_yaml} with weights {model_weights}...")
    
    model = YOLO(model_weights)
    
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=960,        # Reduced from 1280 to 960 to fix CUDA OOM (1280 is too big for T4)
        batch=8,          # Further reduced batch size to fix CUDA OOM on T4
        project="yolo11m_training_runs",
        name=project_name,
        exist_ok=True,
        save=True,
        cache=False,      # Disable RAM caching to prevent system OOM
        mixup=0.1,        # Context recommendation: 50% mixup
        mosaic=0.8,       # Strong augmentation for context
        degrees=10.0,     # Slight rotation
        box=7.5,          # Box loss gain
        cls=0.5,          # Class loss gain
        dfl=1.5,          # DFL loss gain
        workers=2,        # Reduced workers to prevent OOM/Crash on Colab
        
        augment=True,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    
    # Return path to best weights
    return f"yolo11m_training_runs/{project_name}/weights/best.pt"

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 1-epoch test")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Model weights to start with (e.g., yolo11n.pt, yolo11m.pt)")
    args = parser.parse_args()

    # Define the sequence
    # 1. VisDrone (Base)
    # 2. NOMAD (Occlusions)
    # 3. Merged/VTSaR (Target)
    
    # Configuration based on smoke-test
    epochs = 1 if args.smoke_test else 100 # Reduced to 50 for faster iteration on Colab
    imgsz = 640 if args.smoke_test else 1280
    print(f"Configuration: epochs={epochs}, imgsz={imgsz}, model={args.model}")
    
    # Ensure YAMLs exist (placeholders)
    visdrone_yaml = "data/visdrone.yaml"
    nomad_yaml = "data/nomad.yaml"
    merged_yaml = "data/merged_dataset.yaml"
    
    # Step 1: VisDrone
    print("=== Step 1: Training on VisDrone ===")
    # Ultralytics has built-in support for VisDrone.yaml, so we can use it directly
    # It will auto-download the dataset if not present
    try:
        weights_step1 = run_training_step("VisDrone.yaml", args.model, "step1_visdrone", epochs=epochs, imgsz=imgsz) # Use dynamic epochs and imgsz
    except Exception as e:
        print(f"VisDrone training failed: {e}")
        print("Falling back to base weights.")
        weights_step1 = args.model
        
    # Step 2: NOMAD
    print(f"\n=== Step 2: Fine-tuning on NOMAD using {weights_step1} ===")
    if os.path.exists(nomad_yaml):
        weights_step2 = run_training_step(nomad_yaml, weights_step1, "step2_nomad", epochs=50 if not args.smoke_test else 1, imgsz=imgsz)
    else:
        print("NOMAD YAML not found, skipping Step 2.")
        weights_step2 = weights_step1
        
    # Step 3: Merged
    print(f"\n=== Step 3: Final training on Merged Dataset using {weights_step2} ===")
    if os.path.exists(merged_yaml):
        final_weights = run_training_step(merged_yaml, weights_step2, "step3_merged", epochs=epochs, imgsz=imgsz)
        print(f"Pipeline complete. Final weights at: {final_weights}")
    else:
        print("Merged YAML not found. Cannot complete pipeline.")

if __name__ == "__main__":
    main()
