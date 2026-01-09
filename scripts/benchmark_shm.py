"""Benchmark script to compare baseline vs SHM-augmented checkpoints.

Usage:
  python scripts/benchmark_shm.py --baseline /path/to/baseline.pth --shm /path/to/shm.pth --config configs/train_config.yaml

The script will load each checkpoint, run evaluation on validation set, and print mAP results.
"""
import argparse
import os
import torch
import yaml

from scripts.train import load_config, get_transform
from src.models.model_builder import build_model
from src.data.dataset import get_coco_dataset, create_data_loader
from src.evaluation.evaluator import evaluate_model


def load_checkpoint_model(path, config, device):
    model = build_model(config['model']['name'], config['model']['num_classes'],
                        pretrained_backbone=config['model'].get('pretrained_backbone', True))
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('model', ckpt)
    # remove module. prefix
    new_state = {}
    for k, v in state.items():
        if k.startswith('module.'):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True)
    parser.add_argument('--shm', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build val loader
    val_dataset = get_coco_dataset(root=config['dataset']['data_path'], image_set=config['dataset'].get('val_set','val'),
                                   transforms=get_transform(False, config))
    val_loader = create_data_loader(val_dataset, batch_size=1, num_workers=4, shuffle=False)

    print('Evaluating baseline...')
    model_base = load_checkpoint_model(args.baseline, config, device)
    evaluate_model(model_base, val_loader, device)

    print('\nEvaluating SHM...')
    model_shm = load_checkpoint_model(args.shm, config, device)
    evaluate_model(model_shm, val_loader, device)

if __name__ == '__main__':
    main()
