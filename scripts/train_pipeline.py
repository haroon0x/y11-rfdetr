import os
import subprocess
from ultralytics import YOLO

def run_training_step(dataset_yaml, model_weights, project_name, epochs=50):
    print(f"Starting training on {dataset_yaml} with weights {model_weights}...")
    
    model = YOLO(model_weights)
    
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=1280,
        batch=16,
        project="long_range_pipeline",
        name=project_name,
        exist_ok=True,
        save=True
    )
    
    # Return path to best weights
    return f"long_range_pipeline/{project_name}/weights/best.pt"

def main():
    # Define the sequence
    # 1. VisDrone (Base)
    # 2. NOMAD (Occlusions)
    # 3. Merged/VTSaR (Target)
    
    # Ensure YAMLs exist (placeholders)
    visdrone_yaml = "data/visdrone.yaml"
    nomad_yaml = "data/nomad.yaml"
    merged_yaml = "data/merged_dataset.yaml"
    
    # Step 1: VisDrone
    print("=== Step 1: Training on VisDrone ===")
    # Ultralytics has built-in support for VisDrone.yaml, so we can use it directly
    # It will auto-download the dataset if not present
    try:
        weights_step1 = run_training_step("VisDrone.yaml", "yolo11m.pt", "step1_visdrone", epochs=100)
    except Exception as e:
        print(f"VisDrone training failed: {e}")
        print("Falling back to base weights.")
        weights_step1 = "yolo11m.pt"
        
    # Step 2: NOMAD
    print(f"\n=== Step 2: Fine-tuning on NOMAD using {weights_step1} ===")
    if os.path.exists(nomad_yaml):
        weights_step2 = run_training_step(nomad_yaml, weights_step1, "step2_nomad", epochs=50)
    else:
        print("NOMAD YAML not found, skipping Step 2.")
        weights_step2 = weights_step1
        
    # Step 3: Merged
    print(f"\n=== Step 3: Final training on Merged Dataset using {weights_step2} ===")
    if os.path.exists(merged_yaml):
        final_weights = run_training_step(merged_yaml, weights_step2, "step3_merged", epochs=100)
        print(f"Pipeline complete. Final weights at: {final_weights}")
    else:
        print("Merged YAML not found. Cannot complete pipeline.")

if __name__ == "__main__":
    main()
