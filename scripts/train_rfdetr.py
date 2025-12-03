from rfdetr import RFDETRBase
import torch
import os

def train_rfdetr():
    print("Starting RF-DETR training setup...")
    
    # Configuration
    dataset_dir = "data/merged_dataset_coco"
    output_dir = "runs/rfdetr_exp"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Model
    try:
        # Using RFDETRBase as per documentation examples
        model = RFDETRBase()
        print("Model initialized: RFDETRBase")
        
        # Training
        print(f"Starting training with data at {dataset_dir}...")
        model.train(
            dataset_dir=dataset_dir,
            epochs=100,
            batch_size=4,
            grad_accum_steps=4, # Effective batch size 16
            lr=1e-4,
            output_dir=output_dir,
            early_stopping=True
        )
        print("Training complete.")
        
    except Exception as e:
        print(f"RF-DETR training failed: {e}")
        print("Ensure the dataset is in the correct COCO format with train/valid/test subdirectories.")

if __name__ == "__main__":
    train_rfdetr()
