# Advanced Traffic Object Detection using MobileNetV2

This repository contains the implementation of a deep learning framework designed to classify diverse urban traffic entities using **MobileNetV2**. It was developed to evaluate the network's capacity to discriminate between 21 distinct traffic classes commonly found in complex urban environments, including region-specific vehicles like auto-rickshaws and human haulers.

## Overview

The rapid urbanization of modern cities has created a pressing need for intelligent traffic management systems that are capable of autonomous, real-time vehicle classification. This study bridges the gap between high accuracy and computational efficiency by proposing a MobileNetV2-based framework. The model leverages robust data augmentation, standardized high-resolution inputs (128x128 pixels), and strategic layer unfreezing to extract critical spatial features efficiently.

Evaluated on a diverse dataset of over 19,000 traffic bounding box instances, the fine-tuned MobileNetV2 pipeline achieved a sustained accuracy of 66.34%.

## Features

- **Transfer Learning with MobileNetV2**: Pre-trained on ImageNet, fine-tuned specifically for a 21-class urban traffic dataset.
- **Data Augmentation Strategies**: Employs random horizontal flipping and random rotation ($\pm10\%$) to account for differences in camera angles and bilateral vehicle symmetries.
- **Two-Phase Fine-Tuning**: 
  - **Phase 1**: Head training with a frozen backbone.
  - **Phase 2**: Unfreezing the top 50 layers with a highly reduced learning rate ($1 \times 10^{-5}$) to refine spatial representations.
- **Lightweight & Efficient**: The MobileNetV2 backbone ensures parameter efficiency, making the architecture viable for real-time edge computing on urban traffic cameras.

## Key Files
- `improved_cnn_pipeline.py`: Enhanced CNN training and data augmentation pipeline.
- `run_cnn.py`: Standard script to compile and evaluate the CNN model.
- `run_rf.py`: Random Forest baseline/comparison implementation.
- `research_paper.md`: The full manuscript detailing methodologies, metrics, and analyses.

## Performance
The network achieves robust performance metrics, overcoming class imbalances. It reliably recognizes common distinct profiles like cars, motorcycles, and rickshaws, achieving up to 90% recall in key classes. A comprehensive review of performance metrics per class and analysis of misclassification via confusion matrices are fully documented in the provided research manuscript.

## Future Work
- Integration with region-proposal networks (like YOLOv8) for holistic bounding-box scene detection.
- Generation of synthetic data using GANs for underrepresented emergency vehicles.
- Integration of Explainable AI (XAI) workflows (SHAP/LIME) for enhanced algorithmic transparency.
