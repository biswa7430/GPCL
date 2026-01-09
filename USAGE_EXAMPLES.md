# Usage Examples

This document provides practical examples for using the detection framework with multi-dataset support.

## Table of Contents
1. [Multi-Dataset Setup](#multi-dataset-setup)
2. [Basic Training](#basic-training)
3. [Advanced Training](#advanced-training)
4. [Cross-Dataset Training & Testing](#cross-dataset-training--testing)
5. [Evaluation](#evaluation)
6. [Prediction](#prediction)
7. [Common Workflows](#common-workflows)

## Multi-Dataset Setup

The framework now supports training and testing with multiple datasets simultaneously. Both **indraEye** and **visdrone** datasets share the same 5 object classes:
- people
- bus
- car
- motorcycle
- truck

### Configuration Files

Edit `configs/train_config.yaml`, `configs/eval_config.yaml`, or `configs/predict_config.yaml`:

```yaml
# Enable/disable datasets by setting enabled: true/false
dataset:
  multi_datasets:
    - name: "indraEye"
      root: "./data/indraEye_dataset"
      enabled: true    # Set to true to include this dataset
    - name: "visdrone"
      root: "./data/visdrone_dataset"
      enabled: false   # Set to true to include this dataset
```

### Cross-Combination Scenarios

1. **Train on indraEye, Test on visdrone**: Train with indraEye, evaluate/predict on visdrone
2. **Train on visdrone, Test on indraEye**: Train with visdrone, evaluate/predict on indraEye
3. **Train on both, Test on both**: Combine both datasets for training and testing
4. **Train on both, Test on one**: Combined training, evaluate on specific dataset

## Basic Training

### Single Dataset Training

#### Train on indraEye Dataset Only

```bash
# Using config file (set indraEye enabled: true, visdrone enabled: false)
python scripts/train.py --config configs/train_config.yaml

# Or using command line arguments (legacy single dataset mode)
python scripts/train.py \
    --data-path ./data/indraEye_dataset \
    --model fasterrcnn_resnet50_fpn \
    --num-classes 6 \
    --epochs 30 \
    --batch-size 4 \
    --lr 0.02 \
    --output-dir ./checkpoints/indra_only
```

#### Train on VisDrone Dataset Only

Edit `configs/train_config.yaml`:
```yaml
multi_datasets:
  - name: "indraEye"
    root: "./data/indraEye_dataset"
    enabled: false
  - name: "visdrone"
    root: "./data/visdrone_dataset"
    enabled: true
```

Then run:
```bash
python scripts/train.py --config configs/train_config.yaml \
    --output-dir ./checkpoints/indraeye_shm
```

### Combined Multi-Dataset Training

#### Train on Both Datasets Simultaneously

Edit `configs/train_config.yaml`:
```yaml
multi_datasets:
  - name: "indraEye"
    root: "./data/indraEye_dataset"
    enabled: true
  - name: "visdrone"
    root: "./data/visdrone_dataset"
    enabled: true
```

Then run:
```bash
python scripts/train.py --config configs/train_config.yaml \
    --output-dir ./checkpoints/combined_training
```

This will combine all images from both datasets for training and validation.

### Multi-GPU Training (4 GPUs)

```bash
# Distributed training with torchrun
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/train_config.yaml

# Adjust learning rate for multiple GPUs
python scripts/train.py \
    --config configs/train_config.yaml \
    --lr 0.01  # 0.02/8*4 = 0.01 for 4 GPUs
```

## Advanced Training

### Resume from Checkpoint

```bash
python scripts/train.py \
    --config configs/train_config.yaml \
    --resume ./checkpoints/model_10.pth
```

### Mixed Precision Training

```bash
python scripts/train.py \
    --config configs/train_config.yaml \
    --amp
```

### Custom Data Augmentation

Edit `configs/train_config.yaml`:

```yaml
training:
  data_augmentation: "lsj"  # Large Scale Jittering
```

### Different Backbone

```bash
python scripts/train.py \
    --model fasterrcnn_mobilenet_v3_large_fpn \
    --config configs/train_config.yaml \
    --num-classes 6 \
    --epochs 26
```

## Cross-Dataset Training & Testing

This is the key feature - train on one dataset and test on another!

### Scenario 1: Train on indraEye, Test on VisDrone

**Step 1: Train on indraEye**

Edit `configs/train_config.yaml`:
```yaml
multi_datasets:
  - name: "indraEye"
    enabled: true
  - name: "visdrone"
    enabled: false
```

```bash
python scripts/train.py --config configs/train_config.yaml \
    --output-dir ./checkpoints/train_visdrone
```

**Step 2: Evaluate on VisDrone**

Edit `configs/eval_config.yaml`:
```yaml
model:
  checkpoint: "./checkpoints/train_indra_test_visdrone/checkpoint.pth"
  
multi_datasets:
  - name: "indraEye"
    enabled: false
  - name: "visdrone"
    enabled: true
```

```bash
python scripts/evaluate.py --config configs/eval_config.yaml \
    --output-dir ./outputs/evaluation/shm_vis_ind
```

### Scenario 2: Train on VisDrone, Test on indraEye

**Step 1: Train on VisDrone**

Edit `configs/train_config.yaml`:
```yaml
multi_datasets:
  - name: "indraEye"
    enabled: false
  - name: "visdrone"
    enabled: true
```

```bash
python scripts/train.py --config configs/train_config.yaml \
    --output-dir ./checkpoints/train_visdrone_test_indra
```

**Step 2: Evaluate on indraEye**

Edit `configs/eval_config.yaml`:
```yaml
model:
  checkpoint: "./checkpoints/train_visdrone_test_indra/checkpoint.pth"
  
multi_datasets:
  - name: "indraEye"
    enabled: true
  - name: "visdrone"
    enabled: false
```

```bash
python scripts/evaluate.py --config configs/eval_config.yaml \
    --output-dir ./outputs/evaluation/vis_ind
```

### Scenario 3: Train on Both, Test on Each Separately

**Step 1: Train on Combined Dataset**

Edit `configs/train_config.yaml`:
```yaml
multi_datasets:
  - name: "indraEye"
    enabled: true
  - name: "visdrone"
    enabled: true
```

```bash
python scripts/train.py --config configs/train_config.yaml \
    --output-dir ./checkpoints/train_both
```

**Step 2a: Test on indraEye Only**

Edit `configs/eval_config.yaml`:
```yaml
model:
  checkpoint: "./checkpoints/train_both/checkpoint.pth"
  
multi_datasets:
  - name: "indraEye"
    enabled: true
  - name: "visdrone"
    enabled: false
```

```bash
python scripts/evaluate.py --config configs/eval_config.yaml \
    --output-dir ./outputs/evaluation/indra_trained_vis_tested
```

**Step 2b: Test on VisDrone Only**

Edit `configs/eval_config.yaml`:
```yaml
multi_datasets:
  - name: "indraEye"
    enabled: false
  - name: "visdrone"
    enabled: true
```

```bash
python scripts/evaluate.py --config configs/eval_config.yaml \
    --output-dir ./outputs/evaluation/both_trained_visdrone_tested
```

**Step 2c: Test on Both Combined**

Edit `configs/eval_config.yaml`:
```yaml
multi_datasets:
  - name: "indraEye"
    enabled: true
  - name: "visdrone"
    enabled: true
```

```bash
python scripts/evaluate.py --config configs/eval_config.yaml \
    --output-dir ./outputs/evaluation/both_trained_both_tested
```

## Evaluation

### Evaluate Single Dataset

#### Evaluate on indraEye Validation Set

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/checkpoint.pth \
    --config configs/eval_config.yaml \
    --eval-set val \
    --output-dir ./outputs/evaluation/indra_val
```

Make sure `configs/eval_config.yaml` has:
```yaml
multi_datasets:
  - name: "indraEye"
    enabled: true
  - name: "visdrone"
    enabled: false
```

#### Evaluate on VisDrone Test Set

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/checkpoint.pth \
    --config configs/eval_config.yaml \
    --eval-set test \
    --output-dir ./outputs/evaluation/visdrone_test
```

Make sure `configs/eval_config.yaml` has:
```yaml
dataset:
  eval_set: "test"
  multi_datasets:
    - name: "indraEye"
      enabled: false
    - name: "visdrone"
      enabled: true
```

### Evaluate on Combined Datasets

```bash
# Enable both datasets in configs/eval_config.yaml
python scripts/evaluate.py \
    --checkpoint ./checkpoints/checkpoint.pth \
    --config configs/eval_config.yaml \
    --eval-set val \
    --output-dir ./outputs/evaluation/combined_val
```

### Evaluate Multiple Checkpoints

```bash
# Create a loop to evaluate all checkpoints on visdrone
for checkpoint in checkpoints/model_*.pth; do
    echo "Evaluating $checkpoint on VisDrone"
    python scripts/evaluate.py \
        --checkpoint $checkpoint \
        --config configs/eval_config.yaml \
        --eval-set val \
        --output-dir ./outputs/evaluation/visdrone_$(basename $checkpoint .pth)
done
```

## Prediction

### Single Image Prediction

```bash
python scripts/predict.py \
    --checkpoint ./checkpoints/checkpoint.pth \
    --image path/to/your/image.jpg \
    --score-threshold 0.5 \
    --output-dir ./outputs/predictions
```

### Batch Prediction on Single Dataset

```bash
python scripts/predict.py \
    --config configs/predict_config.yaml \
    --num-images 10 \
    --score-threshold 0.5 \
    --output-dir ./outputs/predictions/indra
```

Make sure `configs/predict_config.yaml` has the correct dataset enabled:
```yaml
multi_datasets:
  - name: "indraEye"
    enabled: true
  - name: "visdrone"
    enabled: false
```

### Cross-Dataset Prediction

Predict on different dataset than training:

```bash
# Trained on indraEye, predict on visdrone
python scripts/predict.py \
    --checkpoint ./checkpoints/train_indra/checkpoint.pth \
    --config configs/predict_config.yaml \
    --num-images 10 \
    --output-dir ./outputs/predictions/cross_dataset
```

With `configs/predict_config.yaml`:
```yaml
multi_datasets:
  - name: "indraEye"
    enabled: false
  - name: "visdrone"
    enabled: true
dataset:
  image_set: "val"
```

### Predict on Multiple Datasets Simultaneously

```yaml
multi_datasets:
  - name: "indraEye"
    enabled: true
  - name: "visdrone"
    enabled: true
```

```bash
python scripts/predict.py \
    --config configs/predict_config.yaml \
    --num-images 10 \
    --output-dir ./outputs/predictions/multi
```

This creates separate subdirectories:
- `./outputs/predictions/multi/indraEye/`
- `./outputs/predictions/multi/visdrone/`

### Process All Images

```bash
python scripts/predict.py \
    --checkpoint ./checkpoints/checkpoint.pth \
    --config configs/predict_config.yaml \
    --num-images 0 \  # 0 means all images
    --score-threshold 0.5 \
    --output-dir ./outputs/predictions/all
```

### Higher Confidence Threshold

```bash
python scripts/predict.py \
    --checkpoint ./checkpoints/checkpoint.pth \
    --config configs/predict_config.yaml \
    --num-images 10 \
    --score-threshold 0.8 \  # Fewer but more confident detections
    --output-dir ./outputs/predictions/high_conf
```

### Predict on Test Set

```bash
python scripts/predict.py \
    --checkpoint ./checkpoints/checkpoint.pth \
    --config configs/predict_config.yaml \
    --image-set test \  # Use test set instead of val
    --num-images 10 \
    --output-dir ./outputs/predictions/test
```

## Common Workflows

### Complete Training Pipeline

```bash
#!/bin/bash

# 1. Train the model
echo "Starting training..."
python scripts/train.py \
    --config configs/train_config.yaml \
    --output-dir ./checkpoints/experiment_1

# 2. Evaluate on validation set
echo "Evaluating..."
python scripts/evaluate.py \
    --checkpoint ./checkpoints/experiment_1/checkpoint.pth \
    --data-path ./indraEye_dataset \
    --eval-set val \
    --output-dir ./outputs/evaluation/experiment_1

# 3. Generate predictions
echo "Generating predictions..."
python scripts/predict.py \
    --checkpoint ./checkpoints/experiment_1/checkpoint.pth \
    --data-path ./indraEye_dataset \
    --image-set val \
    --num-images 20 \
    --output-dir ./outputs/predictions/experiment_1
```

### Quick Test Run

```bash
# Test training for 1 epoch
python scripts/train.py \
    --data-path ./indraEye_dataset \
    --model fasterrcnn_resnet50_fpn \
    --num-classes 14 \
    --epochs 1 \
    --batch-size 1 \
    --output-dir ./checkpoints/test
```

### Compare Different Models

```bash
# Train Faster R-CNN with ResNet-50
python scripts/train.py \
    --model fasterrcnn_resnet50_fpn \
    --data-path ./indraEye_dataset \
    --num-classes 14 \
    --output-dir ./checkpoints/resnet50

# Train Faster R-CNN with MobileNet
python scripts/train.py \
    --model fasterrcnn_mobilenet_v3_large_fpn \
    --data-path ./indraEye_dataset \
    --num-classes 14 \
    --output-dir ./checkpoints/mobilenet

# Compare evaluations
python scripts/evaluate.py \
    --checkpoint ./checkpoints/resnet50/checkpoint.pth \
    --data-path ./indraEye_dataset \
    --output-dir ./outputs/evaluation/resnet50

python scripts/evaluate.py \
    --checkpoint ./checkpoints/mobilenet/checkpoint.pth \
    --data-path ./indraEye_dataset \
    --output-dir ./outputs/evaluation/mobilenet
```

### Hyperparameter Tuning

```bash
# Try different learning rates
for lr in 0.005 0.01 0.02 0.04; do
    python scripts/train.py \
        --config configs/train_config.yaml \
        --lr $lr \
        --output-dir ./checkpoints/lr_${lr}
done
```

## Tips and Best Practices

1. **Start Small**: Begin with a small number of epochs to verify everything works

2. **Monitor Training**: Check the output logs regularly for loss values

3. **Save Checkpoints**: Checkpoints are saved automatically every epoch

4. **Evaluation Frequency**: Evaluate every 5 epochs to track progress

5. **Batch Size**: Adjust based on your GPU memory (2 for 16GB, 1 for 8GB)

6. **Learning Rate**: Scale with number of GPUs: `lr = base_lr / 8 * num_gpus`

7. **Data Augmentation**: Use `hflip` for basic, `lsj` for advanced

8. **Score Threshold**: Start with 0.5, increase for fewer false positives

9. **Resume Training**: Always keep the latest checkpoint for resuming

10. **Directory Organization**: Use descriptive names for experiment directories
