import argparse
from ultralytics import YOLO
import sys
import os

def export_model(model_path, format="engine", half=False, int8=False, workspace=4):
    """
    Exports a YOLOv11 model to TensorRT format.
    
    Args:
        model_path (str): Path to the .pt model file.
        format (str): Export format, default is 'engine' (TensorRT).
        half (bool): Use FP16 half-precision export.
        int8 (bool): Use INT8 quantization.
        workspace (int): Workspace size (GB).
    """
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Starting export to {format} (half={half}, int8={int8})...")
    try:
        # Export the model
        # Note: device=0 is usually required for TensorRT export to use the GPU
        export_args = {
            "format": format,
            "half": half,
            "int8": int8,
            "workspace": workspace,
            "device": 0 
        }
        
        # Filter out None or False values if necessary, but Ultralytics handles flags well.
        # For int8, data might be needed for calibration, but standard export often uses default calibration or none.
        
        path = model.export(**export_args)
        print(f"Export completed successfully!")
        print(f"Exported model saved to: {path}")
        return path
    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv11 model to TensorRT")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--half", action="store_true", help="Use FP16 half-precision (recommended for Jetson)")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantization (requires calibration data usually)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)
        
    export_model(args.model, half=args.half, int8=args.int8)
