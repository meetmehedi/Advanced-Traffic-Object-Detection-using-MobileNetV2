# Advanced Traffic Object Detection using MobileNetV2

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Towards Smart City Automation** — A two-stage deep learning pipeline combining **YOLOv8** for object localisation and **MobileNetV2** for fine-grained classification of 21 urban traffic classes in Dhaka, Bangladesh.

---

## Overview

The rapid urbanisation of modern cities demands intelligent traffic management systems capable of real-time, autonomous vehicle classification. This project bridges the gap between high accuracy and computational efficiency by proposing a **MobileNetV2-based** transfer-learning framework evaluated on a diverse dataset of over **19,000 traffic bounding-box instances** spanning 21 distinct classes — including region-specific vehicles such as auto-rickshaws, CNGs, and human haulers.

The fine-tuned pipeline achieves a **sustained classification accuracy of 66.34%** on the validation set, with up to **90% recall** on dominant classes like rickshaws and buses.

---

## Pipeline Architecture

```
Raw Image → YOLOv8 (Detection/Localisation) → Cropped ROI → MobileNetV2 (Classification) → Labelled Output
```

| Stage | Model | Role |
|-------|-------|------|
| Localisation | YOLOv8n | Bounding-box detection |
| Classification | MobileNetV2 (PyTorch) | 21-class fine-grained recognition |
| Explainability | Grad-CAM | Visual attribution heatmaps |
| Evaluation | mAP@0.5 | Full pipeline metric |

---

## Features

- **Transfer Learning with MobileNetV2** — Pre-trained on ImageNet, fine-tuned for a 21-class Bangladeshi urban traffic dataset.
- **Two-Stage Fine-Tuning**
  - *Phase 1*: Head training with frozen backbone.
  - *Phase 2*: Top-50 layer unfreezing at `lr = 1e-5` for spatial refinement.
- **Rich Data Augmentation** — Random horizontal flip, rotation (±15°), and colour jitter.
- **Grad-CAM Explainability** — Visual attribution maps for every prediction via `generate_gradcam.py`.
- **mAP@0.5 Evaluation** — End-to-end mean Average Precision across all 21 classes via `calculate_map.py`.
- **Lightweight & Edge-Ready** — MobileNetV2 backbone ensures real-time viability on edge cameras.
- **Cross-Platform** — All scripts use `os.path` for fully portable, machine-independent paths.

---

## Repository Structure

```
.
├── train_torch.py              # PyTorch MobileNetV2 training script (MPS / CUDA / CPU)
├── validate_model.py           # Classification report on held-out validation split
├── detect_final.py             # Two-stage inference: YOLOv8 + MobileNetV2 classifier
├── calculate_map.py            # End-to-end mAP@0.5 evaluation
├── generate_gradcam.py         # Grad-CAM XAI visualisation
├── improved_cnn_pipeline.py    # TensorFlow/Keras MobileNetV2 training pipeline
├── run_cnn.py                  # Baseline CNN training (TensorFlow)
├── run_rf.py                   # Random Forest baseline
├── consolidated_pipeline_run.py# Single-script full-pipeline runner
├── consolidate_figures.py      # Utility to merge output figures
├── class_names.json            # Ordered list of 21 traffic class names
├── validation_report.txt       # Per-class precision / recall / F1 report
├── requirements.txt            # Python dependencies
└── train/                      # Training data (Pascal VOC XML annotations)
    └── Final Train Dataset/
```

---

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_torch.py
```
Automatically selects **MPS** (Apple Silicon) → **CUDA** → **CPU**.

### 3. Validate
```bash
python validate_model.py
```

### 4. Run Full Detection Pipeline
```bash
python detect_final.py
```
Annotated outputs are saved to `detection_outputs/final/`.

### 5. Evaluate mAP@0.5
```bash
python calculate_map.py
```

### 6. Generate Grad-CAM Explanations
```bash
python generate_gradcam.py
```
Heatmap overlays are saved to `detection_outputs/xai/`.

---

## Performance

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Bus | 0.70 | 0.59 | 0.64 |
| Car | 0.51 | 0.80 | 0.62 |
| Motorbike | 0.63 | 0.71 | 0.67 |
| Rickshaw | 0.76 | 0.62 | 0.68 |
| Three-wheelers (CNG) | 0.47 | 0.72 | 0.57 |
| Truck | 0.53 | 0.59 | 0.56 |
| **Overall Accuracy** | — | — | **0.57** |

> Full per-class breakdown is available in [`validation_report.txt`](validation_report.txt).

---

## Traffic Classes (21)

`ambulance` · `army vehicle` · `auto rickshaw` · `bicycle` · `bus` · `car` · `garbagevan` · `human hauler` · `minibus` · `minivan` · `motorbike` · `pickup` · `policecar` · `rickshaw` · `scooter` · `suv` · `taxi` · `three wheelers (CNG)` · `truck` · `van` · `wheelbarrow`

---

## Future Work

- Integration of YOLOv8 region-proposal fine-tuning on the Dhaka-specific dataset.
- Synthetic data generation via GANs for underrepresented emergency vehicles (ambulance, policecar).
- Real-time video stream inference with FPS benchmarking.
- SHAP / LIME integration as a complementary XAI approach alongside Grad-CAM.

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{hasan2026trafficsmart,
  author       = {Mehedi Hasan},
  title        = {Advanced Traffic Object Detection using MobileNetV2},
  year         = {2026},
  howpublished = {\url{https://github.com/meetmehedi/Advanced-Traffic-Object-Detection-using-MobileNetV2}},
}
```
