# Dlib Model Training Toolkit

## Universal Dlib Training Pipelines

This repository provides modular training c++ programs for:
- ✅ Any Dlib-compatible model architecture
- ✅ Any image dataset (including custom datasets)
- ✅ Multiple training scenarios (from scratch/fine-tuning)

## `dnn_Vision_Transformer_SSL_ex.cpp`

**Description**:  
Implements a Vision Transformer (ViT) trained using Barlow Twins self-supervised learning on CIFAR-10. This example demonstrates:

- 🧠 Pure transformer architecture for computer vision
- 🔍 Self-supervised pretraining (no labels needed)

**Key Features**:
- Modular ViT architecture configurable for different:
  - Patch sizes
  - Attention heads
  - Transformer layers
- Barlow Twins loss for redundancy reduction
- Integrated evaluation with multiclass SVM
