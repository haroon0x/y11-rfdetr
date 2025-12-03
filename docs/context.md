# Long-Range Person Detection Training Context

## Overview
This document summarizes the key context for training computer vision models on long-range person detection from elevated or aerial viewpoints (e.g., drones/UAVs at 60–300m distances). The focus is on detecting small/tiny humans (e.g., <50px bounding boxes) in challenging conditions like occlusions, low visibility, and varied terrains. This setup is ideal for search-and-rescue (SaR) or surveillance applications requiring high precision to minimize false positives.

The training pipeline involves:
- **Datasets**: Merging three complementary public datasets (NOMAD, VisDrone, VTSaR) with transfer learning and sequential fine-tuning.
- **Models**: Primary focus on YOLOv11m and RF-DETR, with emphasis on precision optimizations.
- **Strategy**: Sequential training on individual datasets before merging, using transfer learning from COCO pretraining to enhance generalization and small-object handling.

This context is derived from academic benchmarks, surveys, and best practices as of December 2025.

## Datasets
The selected datasets provide diverse aerial views, annotations (bounding boxes), and metadata (e.g., altitudes) for filtering to ≥100m subsets. Total merged size: ~57,000 samples. Standardize to COCO format via tools like Roboflow, with augmentations (e.g., scaling, mosaic) for 100m+ simulation. Filter for person class only.

### 1. NOMAD (Natural, Occluded, Multi-scale Aerial Dataset)
- **Description**: Multi-scale aerial dataset for emergency human detection, simulating SaR with activities (walking, laying, hiding) and occlusions. Captured at 5.4K resolution (5472x3078, 30fps) from small UAS.
- **Size**: 42,825 annotated frames from 500 videos.
- **Annotations**: Bounding boxes, 10-level visibility (0–100% occlusion), activity labels, metadata (demographics, weather).
- **Altitude/Distance**: Tiered 10–90m (focus on 70–90m horizontal/vertical; subset for 60–100m).
- **Modalities**: RGB.
- **Strengths for Task**: Occlusion handling critical for long-range foliage/partial views; precise tiers for targeted training.
- **Download**: [GitHub](https://github.com/ArtRuss/NOMAD).
- **Why Selected**: Complements urban density with SaR-specific challenges.

### 2. VisDrone (Vision Meets Drones)
- **Description**: Benchmark for UAV object detection in 14 Chinese cities, including pedestrians in crowds/traffic. Diverse urban/rural scenes from 288 videos (~10 hours).
- **Size**: 10,209 frames with 2,806,154 objects (heavy on persons).
- **Annotations**: Bounding boxes, attributes (occlusion: heavy/light/none; truncation, speed).
- **Altitude/Distance**: 5–200m (high coverage ≥100m; filterable for distant crowds).
- **Modalities**: RGB.
- **Strengths for Task**: Scale variety and attributes for tiny-person generalization; dense annotations reduce imbalance.
- **Download**: [Aiskyeye](http://aiskyeye.com/) (registration required).
- **Why Selected**: Largest and most diverse base for initial transfer learning.

### 3. VTSaR (Visual-Thermal Search and Rescue)
- **Description**: Multimodal benchmark for aerial person detection in SaR, with aligned RGB-IR images in coastal/suburban scenes and varied poses/lighting.
- **Size**: 4,801 samples (A-VTSaR variant: 19,956 instances); AS-VTSaR synthetic: 54,749 via mosaicking.
- **Annotations**: Horizontal bounding boxes; potential segmentation.
- **Altitude/Distance**: 50–300m (strong ≥100m coverage; angles 45°–75° for oblique views).
- **Modalities**: RGB + IR (synthetic IR via cropping/stitching).
- **Strengths for Task**: Thermal aids low-visibility at range (fog/night); augmentation for rare high-altitude data.
- **Download**: [GitHub](https://github.com/zxq309/VTSaR) or Baidu Pan (code: qqru).
- **Why Selected**: Extends to extreme distances/modalities, bridging gaps in NOMAD/VisDrone.

### Dataset Comparison Table
| Dataset   | Altitude Range (≥100m Focus) | Size (Samples/Instances) | Annotations/Attributes          | Modalities | Key Strengths for Long-Range |
|-----------|-----------------------------|--------------------------|---------------------------------|------------|------------------------------|
| **NOMAD** | 10–90m (70–90m tiers)      | 42,825 / N/A            | Boxes + visibility (10 levels) | RGB       | Occlusions, activities      |
| **VisDrone** | 5–200m                   | 10,209 / 2.8M           | Boxes + occlusion/truncation   | RGB       | Density, urban/rural variety|
| **VTSaR** | 50–300m                    | 4,801 / 19,956          | Boxes + segmentation potential | RGB + IR  | IR low-vis, synthetic aug   |

**Merging Notes**: Concatenate after filtering ≥100m via metadata (e.g., pandas on JSON/CSV). Use 70/15/15 train/val/test splits. Address domain shifts (e.g., urban vs. coastal) with CycleGAN adaptation.

## Models
Train on YOLOv11m (CNN-based for speed) and RF-DETR (transformer-based for precision), using sequential transfer learning: Pretrain on COCO, fine-tune on VisDrone → NOMAD → merged set. Emphasize precision (>0.85 target) via focal loss and Soft-NMS.

### 1. YOLOv11m
- **Description**: Ultralytics' 2025 real-time detector with C3k2 blocks and attention for multi-scale fusion. Medium variant balances accuracy/speed.
- **Strengths**: +8–12% AP_small on aerial benchmarks (e.g., 29.6% mAP@0.5 on VisDrone proxies); 25 FPS on mid-GPUs.
- **Precision Optimizations**: Focal loss (α=0.25, γ=2); input 1280x1280; confidence 0.6–0.8.
- **Hyperparams**: Epochs 100–300, batch 16, lr 0.01 (cosine decay); augment mixup 50%.
- **Implementation**: Ultralytics CLI: `yolo train model=yolov11m.pt data=merged.yaml epochs=200 imgsz=1280`.
- **Expected Metrics**: mAP@0.5 ~0.45–0.55; precision ~0.50–0.60 on tiny persons.

### 2. RF-DETR
- **Description**: Roboflow's 2025 end-to-end transformer detector (no anchors/NMS), optimized via NAS for real-time (58ms inference).
- **Strengths**: 60+ mAP on COCO; 51.3% AP_small; excels in diverse scenes (+32% over RT-DETR baselines).
- **Precision Optimizations**: Varifocal loss; ensemble with YOLOv11m (weight 0.6); Soft-NMS sigma=0.5.
- **Hyperparams**: Similar to YOLO; use vision-language pretraining for IR in VTSaR.
- **Implementation**: Roboflow Universe or PyTorch: Fine-tune on merged data with LoRA for efficiency.
- **Expected Metrics**: mAP@0.5 ~0.51–0.65; precision ~0.55–0.70 on small objects.

### Model Comparison Table
| Model     | mAP@0.5 (Aerial Est.) | AP_small (Tiny Persons) | Inference (ms) | Params (M) | Precision Focus |
|-----------|-----------------------|--------------------------|----------------|------------|-----------------|
| **YOLOv11m** | 29.6                | 0.40                    | 6             | 20.1      | Speed + fusion |
| **RF-DETR** | ~51.0                | 0.513                   | 58            | 28.6      | Transformer accuracy |

**Ensemble**: Weighted fusion (RF-DETR 0.6, YOLOv11m 0.4) for +10–15% precision uplift.

## Training Pipeline
1. **Preprocessing**: Download/filter datasets; merge with Roboflow (COCO export).
2. **Transfer Learning**: Start from COCO weights; sequential: VisDrone (base) → NOMAD (occlusions) → VTSaR/merged (multimodal).
3. **Precision Enhancements**: Higher resolution; hard negative mining; focal loss; post-process with Soft-NMS.
4. **Evaluation**: PR-curves on ≥100m holdouts; target precision >0.85 at 0.5 recall. Use MLflow for tracking.
5. **Deployment**: Edge-optimized (e.g., TensorRT for drones); validate on real UAV footage.

## Risks and Mitigations
- **Domain Shifts**: Augment with weather/angle variations; monitor val drops.
- **Data Bias**: Diverse scenes covered; audit demographics.
- **Compute**: Use GPUs (e.g., RTX 40-series); LoRA for RF-DETR efficiency.

This setup promises robust, high-precision models for long-range detection. Update as new benchmarks emerge.

---

*Last Updated: December 03, 2025*