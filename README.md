# YOLOv11 Long-Range Person Detection 🎯

This project implements a robust training pipeline for **Long-Range Person Detection** using **YOLOv11** (specifically `yolo11m`). It is designed for local training and deployment on **NVIDIA Jetson Nano**.

## 📂 Project Structure
```text
y11-rfdetr/
├── data/                   # Dataset storage (VisDrone, NOMAD, VTSaR)
├── deployment/             # Deployment scripts
│   └── jetson/             # TensorRT export & inference
├── docs/                   # Documentation
├── notebooks/              # Exploratory Notebooks
├── scripts/                # Helper scripts
│   ├── download_data.py    # Data downloader
│   ├── merge_datasets.py   # Data merger (YOLO format)
│   ├── train_pipeline.py   # Main training script
│   └── export_requirements.py # Dependency management
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🛠️ Setup

We recommend using **uv** for fast dependency management, but standard pip works too.

### 1. Install uv (Optional but Recommended)
```bash
pip install uv
```

### 2. Create & Activate Virtual Environment

**Windows (PowerShell):**
```powershell
uv venv .venv
.\.venv\Scripts\activate
```

**Linux / macOS:**
```bash
uv venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
uv sync
```

## 🏃‍♂️ Training

The training pipeline is defined in `scripts/train_pipeline.py`. It supports command-line arguments for flexible configuration.

### **How to Run**
Make sure your venv is activated, then run:

```bash
python scripts/train_pipeline.py [ARGS]
```

### **Recommended Command (Local Training)**
This configuration balances speed and memory usage for typical local GPUs (e.g., RTX 3060/4070).
```bash
python scripts/train_pipeline.py --epochs 100 --batch 16 --imgsz 1280 --workers 4
```

### **Argument Guide 📚**

| Argument | Default | Usage Guide |
| :--- | :--- | :--- |
| `--epochs` | `50` | **How long to train.** Increase to `100` or `200` for final production models. Decrease to `10` for quick tests. |
| `--patience` | `15` | **Early Stopping.** If the model doesn't improve for this many epochs, training stops. Increase if you think the model needs more time to learn complex patterns. |
| `--batch` | `8` | **Physical Batch Size.** Controls GPU memory usage. <br>• **8-16** for mid-range GPUs (8GB-12GB VRAM). <br>• **32+** for high-end GPUs (24GB+ VRAM). |
| `--imgsz` | `960` | **Image Resolution.** <br>• `1280` is recommended for long-range detection (small objects). <br>• `960` or `640` if you are running out of VRAM. |
| `--workers` | `2` | **Data Loading Speed.** Set to the number of CPU cores you want to dedicate to data loading (e.g., `4` or `8`). |
| `--model` | `yolo11m.pt` | **Starting Weights.** You can switch to `yolo11n.pt` (faster, less accurate) or `yolo11l.pt` (slower, more accurate). |
| `--smoke-test`| `False` | **Debug Mode.** Runs 1 epoch just to make sure the code doesn't crash. |

## 🔍 Inference & SAHI
For detecting small objects, use the SAHI notebook or script:
```bash
# Example SAHI usage (in Python)
from sahi import AutoDetectionModel
from sahi.predict import get_prediction
# ... see notebooks/train_colab_sahi.ipynb for full example
```

## 🤖 Deployment (Jetson Nano)
1.  **Export to TensorRT**:
    ```bash
    python deployment/jetson/export_tensorrt.py --weights best.pt
    ```
2.  **Run Inference**:
    ```bash
    python deployment/jetson/inference_tensorrt.py --engine best.engine --source video.mp4
    ```