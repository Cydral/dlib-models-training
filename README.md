# Dlib Model Training Toolkit

## Universal Dlib Training Pipelines

This repository provides modular training c++ programs for:
- ✅ Any Dlib-compatible model architecture
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

## `dnn_Vision_Transformer_Stable_ImageNet_1K.cpp`

**Description**:  
Comparative training of Vision Transformer (ViT) vs ResNet-34 on Stable ImageNet-1K with full evaluation pipeline. This implementation demonstrates:

- 🖼️ End-to-end ImageNet-1K classification
- ⚖️ Direct comparison between transformer and CNN architectures
- 🏆 Comprehensive Top-1/Top-5 accuracy evaluation

**Key Features**:
- Modular architecture supporting both:
  - **Vision Transformer** with learned positional embeddings
  - **ResNet-34** baseline
- Production-grade training features:
  - Multi-GPU support
  - Graceful interrupt handling
  - Automatic recovery/resumption
- Advanced evaluation:
  - Test-time augmentation (16 crops per image)
  - Both Top-1 and Top-5 accuracy metrics
 
## `slm_advanced_train_ex.cpp`

**Description**:  
Advanced transformer language model with text reconstruction capabilities. Implements:

- 🧠 Transformer architecture with Rotary Positional Embeddings (RoPE)
- 🧩 Mixture-of-Experts (MoE) layers
- 🔤 BPE tokenization with custom vocabulary

**Key Features**:
- Text memorization/reconstruction
- Three operational modes:
  - 🏋️ Training with RoPE-enhanced attention
  - 🖨️ Autoregressive text generation
  - 🔍 Byte-level verification
- Memory-efficient training:
  - Sliding window sequences
  - Adaptive batch striding
- Production-ready:
  - Model checkpointing
  - Token caching
  - GPU acceleration
 
## `slm_advanced_train_ex2.cpp`

**Description**:
Transformer language model featuring dynamic network-in-layer architecture for Mixture-of-Experts (MoE) implementation.

**Core Components**:
Core components:
- Transformer backbone with Rotary Positional Embeddings
- True MoE implementation (experts as subnetworks)
- Dynamic routing (top-k experts per token)

**Key Features**:
- 🏗️ Layer-as-network design:
  - Each expert is a complete feed-forward subnetwork
  - Gating network controls expert selection
  - Native backpropagation through routing
- 🏋️ Training modes:
  * Expert balancing via auxiliary loss
  * Noise injection during training
  * Usage-based expert selection

**Operational Modes**:
1. --train : Full MoE training
2. --generate : Expert-conditional text generation
3. --verify : Reconstruction validation
