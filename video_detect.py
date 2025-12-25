import cv2
from ultralytics import YOLO
from pathlib import Path
import os

# Check if GUI is available (for headless environments like servers or headless OpenCV)
HEADLESS = os.environ.get("OPENCV_HEADLESS", "0") == "1"

model_path = Path(r"/home/cs-ai-01/Downloads/100ep,960,--p2&no--p5.pt")
if not model_path.exists():
    model_path = Path("runs/detect/train/weights/best.pt")

model = YOLO(str(model_path))

video_path = Path(r"/home/cs-ai-01/Downloads/TestSetVideos/Drone1/Morning/1.1.9.mp4")
if not video_path.exists():
    video_path = Path("2.mp4")

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"Error: Couldn't open video at {video_path}.")
    exit()

# Get video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = Path("output/output[okutama]_960_p2_nop5.1.9.mp4")
output_path.parent.mkdir(parents=True, exist_ok=True)
out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
show_gui = not HEADLESS

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO object detection on the frame
    results = model(frame, stream=True, conf=0.5)

    # Draw the results on the frame
    for r in results:
        for box in r.boxes:
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    frame_count += 1

    # Try to show frame (may fail on headless systems)
    if show_gui:
        try:
            display_frame = cv2.resize(frame, (800, 600))
            cv2.imshow("YOLO Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            print("GUI display not available, running in headless mode...")
            show_gui = False

# Cleanup
print(f"Processed {frame_count} frames. Output saved to: {output_path}")
cap.release()
out.release()
if show_gui:
    cv2.destroyAllWindows()


