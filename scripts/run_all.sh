#!/bin/bash
set -e  # Exit on error

# Fix matplotlib backend for Colab/headless environments
export MPLBACKEND=Agg

echo "============================================================"
echo "🚀 Starting Complete YOLOv11 VisDrone Pipeline"
echo "============================================================"

# 1. Install Dependencies & Setup Environment with uv
echo -e "\n📦 Setting up environment with uv..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    python3 -m pip install uv
fi

echo "Syncing dependencies..."
uv sync

# 2. Download Dataset
echo -e "\n⬇️  Step 1: Downloading VisDrone Dataset..."
uv run python scripts/download_visdrone.py

# 3. Prepare Dataset (Filter & Merge)
echo -e "\nRg️  Step 2: Preparing Dataset (Merging Person + Pedestrian)..."
uv run python scripts/prepare_visdrone_.py

# 4. Train Model
echo -e "\n🔥 Step 3: Starting Training..."
# Using pipeline script with flexible arguments
# Adjust --batch and --imgsz based on your GPU memory
uv run python scripts/train_pipeline.py --epochs 100 --p2 --batch 4 --imgsz 960 --model yolo11n.pt 

echo -e "\n✅ Pipeline execution complete!"
