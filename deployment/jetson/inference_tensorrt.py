import argparse
from ultralytics import YOLO
import time
import os
import sys
import cv2

def run_inference(engine_path, source, conf=0.25, iou=0.45):
    """
    Runs inference on a source using a TensorRT engine.
    
    Args:
        engine_path (str): Path to the .engine file.
        source (str): Path to image/video or '0' for webcam.
        conf (float): Confidence threshold.
        iou (float): NMS IOU threshold.
    """
    print(f"Loading TensorRT engine from {engine_path}...")
    try:
        model = YOLO(engine_path)
    except Exception as e:
        print(f"Error loading engine: {e}")
        sys.exit(1)

    print(f"Running inference on {source}...")
    
    # Warmup
    print("Warming up...")
    model(source, imgsz=640, conf=conf, iou=iou, verbose=False, stream=True) # Run once to load
    
    # Run inference
    start_time = time.time()
    results = model(source, imgsz=640, conf=conf, iou=iou, stream=True)
    
    frame_count = 0
    
    for result in results:
        frame_count += 1
        # You can process results here (e.g., save images, print detections)
        # For benchmark, we just iterate
        pass
        
    end_time = time.time()
    duration = end_time - start_time
    fps = frame_count / duration if duration > 0 else 0
    
    print(f"\nInference Completed.")
    print(f"Processed {frame_count} frames in {duration:.2f} seconds.")
    print(f"Average FPS: {fps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TensorRT Inference on Jetson")
    parser.add_argument("--engine", type=str, required=True, help="Path to .engine file")
    parser.add_argument("--source", type=str, default="https://ultralytics.com/images/bus.jpg", help="Input source (image, video, 0 for cam)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.engine):
        print(f"Error: Engine file not found at {args.engine}")
        sys.exit(1)
        
    run_inference(args.engine, args.source, conf=args.conf)
