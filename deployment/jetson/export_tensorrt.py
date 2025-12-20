import argparse
from ultralytics import YOLO
from pathlib import Path
import sys
import os

def export_model(model_path, format="engine", half=False, int8=False, imgsz=960, workspace=4, data=None, dynamic=False):
    """
    Exports a YOLOv11 model to TensorRT format.
    
    Args:
        model_path (str): Path to the .pt model file.
        format (str): Export format, default is 'engine' (TensorRT).
        half (bool): Use FP16 half-precision export.
        int8 (bool): Use INT8 quantization.
        imgsz (int): Input image size (TensorRT engines are fixed to this size).
        workspace (int): Workspace size (GB).
        data (str): Path to data.yaml for INT8 calibration (recommended for INT8).
        dynamic (bool): Enable dynamic input shapes.
    """
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Starting export to {format}...")
    print(f"  - Precision: {'INT8' if int8 else 'FP16' if half else 'FP32'}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Dynamic shapes: {dynamic}")
    
    try:
        export_args = {
            "format": format,
            "half": half,
            "int8": int8,
            "imgsz": imgsz,
            "workspace": workspace,
            "dynamic": dynamic,
            "device": 0,
        }
        
        # Add calibration data for INT8 quantization
        if int8 and data:
            export_args["data"] = data
            print(f"  - Calibration data: {data}")
        elif int8 and not data:
            print("  - WARNING: INT8 without calibration data may reduce accuracy!")
            print("    Consider using --data path/to/data.yaml for better results.")
        
        path = model.export(**export_args)
        print(f"\n✅ Export completed successfully!")
        print(f"   Exported model saved to: {path}")
        
        # Print file size
        engine_path = Path(path)
        if engine_path.exists():
            size_mb = engine_path.stat().st_size / (1024 * 1024)
            print(f"   Model size: {size_mb:.1f} MB")
        
        return path
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv11 model to TensorRT")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--half", action="store_true", help="Use FP16 half-precision (recommended for Jetson)")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantization (fastest, use --data for calibration)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml for INT8 calibration")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace size in GB (default: 4)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)
    
    # Validate INT8 + data recommendation
    if args.int8 and not args.data:
        print("\n💡 Tip: For best INT8 accuracy, provide calibration data:")
        print(f"   python export_tensorrt.py --model {args.model} --int8 --data data/visdrone_person/data.yaml\n")
        
    export_model(
        args.model, 
        half=args.half, 
        int8=args.int8, 
        imgsz=args.imgsz,
        workspace=args.workspace,
        data=args.data,
        dynamic=args.dynamic
    )

