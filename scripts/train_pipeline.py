import os
import sys

# Fix matplotlib backend BEFORE any imports - this is critical for Colab
os.environ["MPLBACKEND"] = "Agg"
os.environ["OPENCV_HEADLESS"] = "1"

# Force matplotlib to use Agg backend even if already partially initialized
if 'matplotlib' in sys.modules:
    del sys.modules['matplotlib']
if 'matplotlib.pyplot' in sys.modules:
    del sys.modules['matplotlib.pyplot']

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import subprocess
from ultralytics import YOLO

def run_training_step(dataset_yaml, model_weights, project_name, epochs=50, imgsz=960, batch=8, workers=2, patience=50, model_yaml=None):
    """
    Run a training step with optional custom model architecture.
    
    Args:
        dataset_yaml: Path to dataset YAML
        model_weights: Path to pretrained weights (.pt) or model architecture (.yaml)
        project_name: Name for this training run
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        workers: Number of dataloader workers
        patience: Early stopping patience
        model_yaml: Optional path to custom model YAML (e.g., P2 architecture)
    """
    if model_yaml:
        # Load custom architecture from YAML, then optionally load pretrained weights
        print(f"Loading custom architecture from {model_yaml}...")
        model = YOLO(model_yaml)
        if model_weights and model_weights.endswith('.pt'):
            print(f"Transferring pretrained weights from {model_weights}...")
            model.load(model_weights)
        
        # Absolute verification: print the model summary and check detection heads
        print("\n" + "="*50)
        print(f"VERIFICATION: Loaded model from {model_yaml}")
        print(f"Detection Layers: {len(model.model.names)} classes at {len(model.model.yaml.get('head', []))} head components")
        # In YOLO11, the 'Detect' module is usually the last component in 'head'
        # We can also check the number of layers in the model
        print(f"Model Layers: {len(model.model.model)}")
        print("="*50 + "\n")
        
        project_dir = f"{os.path.basename(model_yaml).replace('.yaml', '')}_training_runs"
    else:
        print(f"Starting training on {dataset_yaml} with weights {model_weights}...")
        model = YOLO(model_weights)
        project_dir = f"{model_weights.replace('.pt', '')}_training_runs"

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project_dir,
        name=project_name,
        exist_ok=True,
        amp=True,              # Enable Mixed Precision (Faster & Less Memory)
        save=True,
        patience=patience,
        cache=False,           # Disable RAM caching to prevent system OOM
        workers=workers,
        
        # High-precision optimizations for 50m altitude UAV person detection
        single_cls=True,       # Single class (person only) - optimized training
        deterministic=True,    # Reproducible results
        cos_lr=True,           # Cosine LR schedule for smoother convergence
        close_mosaic=30,       # Disable mosaic for final 30 epochs (stabilize training)
        
        # Reduced augmentation for cleaner, high-precision detection (research-backed)
        mosaic=0.5,            # Reduced mosaic - preserves small object details
        mixup=0.0,             # Disabled - cleaner training
        copy_paste=0.1,        # Helps with small object detection
        degrees=8.0,           # Moderate rotation
        scale=0.3,             # Reduced scaling to preserve small objects
        translate=0.1,         # Standard translation
        fliplr=0.5,            # Standard horizontal flip
        hsv_h=0.015,           # Standard hue variation
        hsv_s=0.4,             # Reduced saturation variation
        hsv_v=0.3,             # Reduced brightness variation
        
        # Loss function tuning for small objects (research-backed)
        box=10.0,              # Increased box loss for better small object localization
        cls=0.5,               # Class loss gain
        dfl=1.5,               # DFL loss gain
        
        augment=True,
    )
    
    # Return path to best weights
    return f"{project_dir}/{project_name}/weights/best.pt"

import argparse

def main():
    parser = argparse.ArgumentParser(description="YOLO Training Pipeline with P2 support for small object detection")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 1-epoch test")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained weights (e.g., yolo11n.pt, yolo11s.pt)")
    
    # Model Architecture Options
    parser.add_argument("--p2", action="store_true", help="Use P2 detection layer for small objects (loads configs/yolo11n-p2.yaml)")
    parser.add_argument("--model-yaml", type=str, default=None, help="Custom model architecture YAML (overrides --p2)")
    
    # Training Hyperparameters (research-backed for 50m altitude high-precision detection)
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs (200 recommended for small objects)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (physical)")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size (960 for small objects)")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    
    args = parser.parse_args()

    
    # Configuration override based on smoke-test
    if args.smoke_test:
        args.epochs = 1
        args.imgsz = 640
        args.batch = 4
    
    # Determine model architecture
    model_yaml = args.model_yaml
    if args.p2 and not model_yaml:
        model_yaml = "configs/yolo11n-p2.yaml"
        print("🎯 Using P2 detection layer for enhanced small object detection")
    
    arch_info = f"architecture={model_yaml}" if model_yaml else f"weights={args.model}"
    print(f"Configuration: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, workers={args.workers}, {arch_info}")
    
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
            patience=args.patience,
            model_yaml=model_yaml
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
