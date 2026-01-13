# Geometry-Preserving Contrastive Learning for Cross-Geographic Aerial Object Detection

[![ICPR 2026](https://img.shields.io/badge/ICPR-2026-blue)](https://www.icpr2026.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **"Geometry-Preserving Contrastive Learning with Dual-Level Style Augmentation for Cross-Geographic Domain Generalization in Aerial Object Detection"** (ICPR 2026).

## 🎯 Overview

Cross-geographic deployment of aerial object detection systems faces severe performance degradation due to visual and environmental variations. This work introduces **Geometry-Preserving Contrastive Learning (GPCL)** with dual-level style augmentation to maintain spatial consistency across domains while maximizing appearance diversity.

### Key Contributions

- **Geometry-Preserving Contrastive Learning (GPCL)**: Novel IoU-weighted contrastive learning on RoI features that maintains spatial relationships across style variations
- **Dual-Level Style Augmentation**: Combines image-level AdaIN and feature-level prototype hallucination for comprehensive appearance diversity
- **Detection-Specific Consistency**: Variable-length output handling through IoU-based geometric and KL-divergence-based semantic consistency
- **Cross-Geographic Benchmarks**: Rigorous evaluation on VisDrone (China) → IndraEye (India) with novel DG metrics (DG-GAP, SI-Score, SSDG)

### Results mAP%

| Training → Testing | Baseline | GPCL-Full | Improvement |
|-------------------|----------|-----------|-------------|
| Indraeye → IndraEye | 64.5% | **68.2%** | +3.7% (+5.7%) |
| VisDrone → IndreEye | 34.2% | **46.5%** | +12.3% (+36.0%) |
| VisDrone → Visdrone | 55.3% | **58.7%** | +3.4% (+6.1%) |
| IndraEye → VisDrone | 28.9% | **37.6%** | +8.7% (+30.0%) |


## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/gpcl-aerial-detection.git
cd gpcl-aerial-detection

# Create environment
conda create -n gpcl python=3.9
conda activate gpcl

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Dataset Preparation

1. **VisDrone Dataset**: Download from [VisDrone](http://aiskyeye.com/)
2. **IndraEye Dataset**: 

Organize datasets as:
```
data/
├── visdrone_dataset/
│   ├── annotations/
│   ├── train/
│   └── val/
└── indraEye_dataset/
    ├── annotations/
    ├── train/
    └── val/
```

### Training

```bash
# Single-source training (VisDrone → IndraEye)
python scripts/train.py \
    --config configs/train_config.yaml \
    --train-dataset visdrone \
    --output-dir checkpoints/visdrone_to_indra

# Multi-source training
python scripts/train.py \
    --config configs/train_config.yaml \
    --train-dataset both \
    --output-dir checkpoints/combined
```

### Evaluation

```bash
# Evaluate on target domain
python scripts/evaluate.py \
    --config configs/eval_config.yaml \
    --checkpoint checkpoints/visdrone_to_indra/checkpoint_best.pth \
    --test-dataset indraEye
```

### Inference

```bash
# Run predictions with visualization
python scripts/predict.py \
    --config configs/predict_config.yaml \
    --checkpoint checkpoints/visdrone_to_indra/checkpoint_best.pth \
    --num-images 10 \
    --output-dir outputs/predictions
```

## 📊 Model Architecture

```
Input Image
    ↓
Dual-Level Style Augmentation
    ├── Image-Level: AdaIN (α=0.5, p=0.85)
    └── Feature-Level: Prototype Hallucination (K=48)
    ↓
ResNet-50-FPN Backbone
    ↓
Region Proposal Network (RPN)
    ↓
RoI Align → Box Head
    ├── Detection Heads (Classification + Regression)
    └── GPCL Module (IoU-weighted Contrastive Loss)
    ↓
Consistency Regularization (IoU + KL-divergence)
```

## 🔬 Key Components

### 1. GPCL Loss (`src/losses/gpcl_loss.py`)
```python
# Geometry-preserving contrastive learning
gpcl_loss = GeometryPreservingContrastiveLoss(
    temperature=0.07,
    feature_dim=1024,
    projection_dim=128
)
```

### 2. Dual-Level Augmentation (`src/augmentation/`)
- **Image-level**: AdaIN style transfer
- **Feature-level**: Learnable style prototypes with Dirichlet mixing

### 3. Consistency Regularization (`src/losses/consistency_loss.py`)
- **Geometric**: IoU-based box consistency
- **Semantic**: KL-divergence class consistency

## 📈 Configuration

Key hyperparameters in `configs/train_config.yaml`:

```yaml
# Loss weights
lambda_gpcl: 0.3          # GPCL weight
lambda_consistency: 0.7    # Consistency weight

# GPCL settings
gpcl_temperature: 0.07     # Contrastive temperature
projection_dim: 128        # Projection space dimension

# Style augmentation
adain_alpha: 0.5          # AdaIN mixing ratio
style_prob: 0.85          # Augmentation probability
num_prototypes: 48        # Number of style prototypes
```


## 🙏 Acknowledgements

This work is supported by Anusandhan National Research Foundation (ANRF), India, Project No. SRDP 1191 G.

- VisDrone dataset from [VisDrone Team](http://aiskyeye.com/)
- Based on [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/)
- Faster R-CNN implementation adapted from torchvision

## 📧 Contact

For questions or collaborations:
- **Biswajit Bera**: 25dr0304@iitism.ac.in, biswabera0382@gmail.com
- **Sudhakar Kumawat**: sudhakar@iitism.ac.in
- **Manisha Verma**: manisha@iitism.ac.in

**Affiliation**: IIT (ISM) Dhanbad, India

---

**Paper Status**: Submitted to ICPR 2026 (January 2026)
**Code Release**: January 2026
