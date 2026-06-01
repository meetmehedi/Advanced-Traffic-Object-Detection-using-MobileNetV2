# 🚦 Lightweight Deep Learning for Urban Traffic Intelligence
### A MobileNetV2-YOLOv8 Pipeline with Adaptive Resolution Processing

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MobileNetV2-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Paper-Accepted%20ECCT%202026-9B59B6?style=for-the-badge"/>
</p>

<p align="center">
  <b>Accepted at ECCT 2026 (ACSR, Atlantis Press, Springer Nature)</b><br/>
  Biswas, S., Hasan, M. M.,Rakib, R., Mahin, A. A., Afrose, M., Based, M. A.
</p>

---

## 📌 Overview

This repository presents a **production-ready, lightweight deep learning pipeline** for real-time urban traffic object detection. By fusing **MobileNetV2** (for efficient feature extraction) with **YOLOv8** (for high-speed detection), and introducing an **Adaptive Resolution Processing (ARP)** module, this system achieves state-of-the-art accuracy on resource-constrained hardware — making it ideal for smart city deployments in low-resource environments.

> 💡 **Why this matters**: Most traffic detection systems require high-end GPUs. This pipeline runs efficiently on edge devices while maintaining competitive accuracy — a critical requirement for developing countries and smart city infrastructure.

---

## 🏗️ Architecture

```
Input Frame
    │
    ▼
Adaptive Resolution Processing (ARP)
    │  └── Dynamic scaling based on scene complexity
    ▼
MobileNetV2 Backbone
    │  └── Depthwise separable convolutions (lightweight)
    ▼
YOLOv8 Detection Head
    │  └── Multi-scale object detection
    ▼
Post-Processing (NMS + Tracking)
    │
    ▼
Detected Objects + Risk Score
```

---

## ✨ Key Features

- 🔴 **Adaptive Resolution Processing (ARP)** — dynamically adjusts input resolution based on scene complexity, reducing compute by up to 40% with negligible accuracy loss
- 🟡 **MobileNetV2 + YOLOv8 Fusion** — combines efficient feature extraction with fast detection in a single unified pipeline
- 🟢 **Edge-Device Optimized** — runs on Raspberry Pi, Jetson Nano, and similar hardware
- 🔵 **Smart City Ready** — integrates with traffic management APIs and supports multi-camera inputs
- 🟣 **Explainable Outputs** — bounding boxes with class confidence, speed estimation, and congestion scoring

---

## 📊 Results

| Model | mAP@0.5 | FPS (GPU) | FPS (CPU) | Parameters |
|-------|---------|-----------|-----------|------------|
| YOLOv8n (baseline) | 0.71 | 142 | 23 | 3.2M |
| MobileNetV2 (baseline) | 0.68 | 98 | 31 | 3.4M |
| **Ours (MobileNetV2-YOLOv8 + ARP)** | **0.79** | **136** | **28** | **4.1M** |

> Evaluated on a custom urban traffic dataset collected from Dhaka, Bangladesh — one of the world's most congested cities.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/meetmehedi/Advanced-Traffic-Object-Detection-using-MobileNetV2
cd Advanced-Traffic-Object-Detection-using-MobileNetV2
pip install -r requirements.txt
```

### Run Inference on an Image

```bash
python detect.py --source path/to/image.jpg --weights weights/best.pt --conf 0.4
```

### Run Inference on Video / Webcam

```bash
# Video file
python detect.py --source path/to/video.mp4 --weights weights/best.pt

# Webcam (real-time)
python detect.py --source 0 --weights weights/best.pt --real-time
```

### Train from Scratch

```bash
python train.py --data config/traffic.yaml --epochs 100 --img 640 --batch 16
```

---

## 📁 Repository Structure

```
├── data/
│   ├── sample_images/          # Sample test images
│   └── traffic.yaml            # Dataset configuration
├── models/
│   ├── mobilenetv2_backbone.py # MobileNetV2 feature extractor
│   ├── yolov8_head.py          # YOLOv8 detection head
│   └── arp_module.py           # Adaptive Resolution Processing
├── weights/
│   └── best.pt                 # Pretrained weights
├── utils/
│   ├── preprocessing.py
│   └── postprocessing.py
├── notebooks/
│   └── EDA_and_Training.ipynb  # Original research notebook
├── detect.py                   # Inference script
├── train.py                    # Training script
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

The model was trained and evaluated on a custom dataset of urban traffic scenes collected from **Dhaka, Bangladesh** — representing one of the world's densest traffic environments.

| Class | Instances |
|-------|-----------|
| Car | 12,480 |
| Rickshaw | 8,230 |
| Bus | 4,110 |
| Motorcycle | 9,870 |
| Pedestrian | 11,340 |
| CNG/Auto | 5,620 |

> Dataset available upon request for academic use. Contact: meetmehedi1@gmail.com

---

## 🧪 Requirements

```
torch>=2.0.0
ultralytics>=8.0.0
tensorflow>=2.10.0
opencv-python>=4.7.0
numpy>=1.24.0
pandas>=1.5.0
matplotlib>=3.6.0
```

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@inproceedings{biswas2026lightweight,
  title     = {Lightweight Deep Learning for Urban Traffic Intelligence: 
               A MobileNetV2-YOLOv8 Pipeline with Adaptive Resolution Processing},
  author    = {Biswas, S. and Hasan, M. M., and Rakib, R., and Mahin, A. A. and Afrose, M. and Based, M. A.},
  booktitle = {Proceedings of ECCT 2026},
  publisher = {Atlantis Press, Springer Nature (ACSR)},
  year      = {2026}
}
```

---

## 👨‍💻 Authors

**Md. Mehedi Hasan**
Lead Researcher  
📧 meetmehedi1@gmail.com | 🌐 [mdmehedihasan.us](https://www.mdmehedihasan.us) | [LinkedIn](https://linkedin.com/in/meetmehedi)

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  ⭐ If this work helped you, please consider starring the repository!<br/>
</p>
