# YOLOv11 Person Detection 🎯

Train **YOLOv11** for **person-only detection** using **VisDrone** dataset. Optimized for edge deployment on **NVIDIA Jetson Nano**.

## 📂 Project Structure
```text
y11-rfdetr/
├── data/                   # Dataset storage
│   └── visdrone/           # Raw VisDrone data
├── deployment/jetson/      # TensorRT export & inference
├── docs/                   # Documentation
├── notebooks/              # Colab training notebooks
├── scripts/
│   ├── prepare_visdrone.py # Filter VisDrone for person class
│   ├── train_yolo.py       # Simple training script
│   ├── train_pipeline.py   # Advanced training with args
│   ├── inference_sahi.py   # SAHI sliced inference
│   └── verify_setup.py     # Check environment
└── README.md
```

## 🛠️ Setup

### 1. Install Dependencies
```bash
pip install uv  # Optional but recommended
uv venv .venv && .venv\Scripts\activate  # Windows
uv sync
```

### 2. Download VisDrone Dataset
Download VisDrone DET from [aiskyeye.com](http://aiskyeye.com/) and extract to:
```
data/visdrone/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-val/
    ├── images/
    └── annotations/
```

### 3. Prepare Person-Only Dataset
```bash
python scripts/prepare_visdrone.py
```
This filters for pedestrian/people classes and creates `data/visdrone_person/`.

## 🏃‍♂️ Training

### Quick Start
```bash
python scripts/train_yolo.py
```

### With Custom Args
```bash
python scripts/train_pipeline.py --epochs 100 --batch 16 --imgsz 640 --model yolo11s.pt
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Training epochs |
| `--batch` | 8 | Batch size |
| `--imgsz` | 640 | Image resolution (1280 for small objects) |
| `--model` | yolo11s.pt | Base weights (yolo11n/s/m/l/x.pt) |
| `--smoke-test` | - | Quick 1-epoch test |

## 🔍 SAHI Inference (Small Objects)
```bash
python scripts/inference_sahi.py --model best.pt --source image.jpg
```

## 🤖 Jetson Nano Deployment
```bash
# Export to TensorRT
python deployment/jetson/export_tensorrt.py --weights best.pt

# Run inference
python deployment/jetson/inference_tensorrt.py --engine best.engine --source video.mp4
```