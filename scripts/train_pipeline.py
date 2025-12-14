import os
os.environ["OPENCV_HEADLESS"] = "1"
import subprocess
from ultralytics import YOLO

def run_training_step(dataset_yaml, model_weights, project_name, epochs=50, imgsz=960, batch=8, workers=2, patience=15):
    print(f"Starting training on {dataset_yaml} with weights {model_weights}...")
    
    model = YOLO(model_weights)
    
    # Dynamic project naming
    project_dir = f"{model_weights.replace('.pt', '')}_training_runs"

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project_dir,
        name=project_name,
        exist_ok=True,
        amp=True,         # Enable Mixed Precision (Faster & Less Memory)
        save=True,
        patience=patience,
        cache=False,      # Disable RAM caching to prevent system OOM
        # device=0,       # Removed to allow auto-detection (CPU/GPU)
        workers=workers,
        
        # Long-range/Small object optimizations
        mixup=0.1,        # Context recommendation: 50% mixup
        mosaic=0.8,       # Strong augmentation for context
        degrees=10.0,     # Slight rotation
        box=7.5,          # Box loss gain
        cls=0.5,          # Class loss gain
        dfl=1.5,          # DFL loss gain
        
        augment=True,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    
    # Return path to best weights
    return f"{project_dir}/{project_name}/weights/best.pt"

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 1-epoch test")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Model weights to start with (e.g., yolo11n.pt, yolo11s.pt, yolo11m.pt)")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (physical)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    
    args = parser.parse_args()

    # Define the sequence
    # 1. VisDrone (Base)
    # 2. NOMAD (Occlusions)
    # 3. Merged/VTSaR (Target)
    
    # Configuration override based on smoke-test
    if args.smoke_test:
        args.epochs = 1
        args.imgsz = 640
        args.batch = 2
        
    print(f"Configuration: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, workers={args.workers}, model={args.model}")
    
    # Ensure YAMLs exist (placeholders)
    visdrone_yaml = "data/visdrone.yaml"
    nomad_yaml = "data/nomad.yaml"
    merged_yaml = "data/visdrone_person/data.yaml"
    
    # Helper to pass args cleanly
    def train_step(dataset, weights, name, epochs_override=None):
        return run_training_step(
            dataset, 
            weights, 
            name, 
            epochs=epochs_override if epochs_override else args.epochs, 
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            patience=args.patience
        )
    
    # Step 1: VisDrone (Filtered Person Only)
    print("=== Step 1: Training on VisDrone (People Only) ===")
    # Check if prepared dataset exists (created by prepare_visdrone.py)
    visdrone_data = "data/visdrone_person/data.yaml"
    if not os.path.exists(visdrone_data):
        print("ERROR: Person-only dataset not found at data/visdrone_person/data.yaml")
        print("Please run 'python scripts/prepare_visdrone.py' first.")
        print("Exiting to prevent training on wrong dataset.")
        exit(1)

    try:
        weights_step1 = train_step(visdrone_data, args.model, "step1_visdrone_person")
    except Exception as e:
        print(f"Training failed: {e}")
        exit(1)
        
    # Step 2: NOMAD (Skipped for now)
    # print(f"\n=== Step 2: Fine-tuning on NOMAD using {weights_step1} ===")
    # if os.path.exists(nomad_yaml):
    #     weights_step2 = train_step(nomad_yaml, weights_step1, "step2_nomad")
    # else:
    #     print("NOMAD YAML not found, skipping Step 2.")
    #     weights_step2 = weights_step1
        
    # Step 3: Merged (Skipped for now)
    # print(f"\n=== Step 3: Final training on Merged Dataset using {weights_step2} ===")
    # if os.path.exists(merged_yaml):
    #     final_weights = train_step(merged_yaml, weights_step2, "step3_merged")
    #     print(f"Pipeline complete. Final weights at: {final_weights}")
    # else:
    #     print("Merged YAML not found. Cannot complete pipeline.")
    
    print(f"Step 1 complete. Best weights: {weights_step1}")

if __name__ == "__main__":
    main()
