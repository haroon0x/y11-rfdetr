from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
from sahi.utils.cv import read_image
import cv2
import argparse
import os

def run_sahi_inference(model_path, source_image, output_dir="runs/sahi_inference"):
    print(f"Running SAHI inference on {source_image} using {model_path}...")
    
    # 1. Initialize the model
    # SAHI supports YOLOv8/v11 via the 'yolov8' model_type (compatible with v11)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.3,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    # 2. Read Image
    image = read_image(source_image)

    # 3. Get Sliced Prediction
    # slice_height/width: Size of the patches (e.g., 640x640)
    # overlap_height_ratio: How much patches overlap (0.2 = 20%)
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # 4. Save Result
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(source_image)
    save_path = os.path.join(output_dir, f"sahi_{filename}")
    
    # Export visualization
    result.export_visuals(export_dir=output_dir, file_name=f"sahi_{filename}")
    
    print(f"Saved result to {save_path}")
    return result

if __name__ == "__main__":
    import torch # Import here to check cuda
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Path to trained model weights")
    parser.add_argument("--source", type=str, required=True, help="Path to image file")
    args = parser.parse_args()
    
    run_sahi_inference(args.model, args.source)
