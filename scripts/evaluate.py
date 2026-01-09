#!/usr/bin/env python3
"""
Evaluation script for object detection models.
Evaluates a trained model on the test/validation set.
"""

import argparse
import os
import sys
import yaml
import json

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.model_builder import get_model_instance
from src.data.dataset import get_coco_dataset, create_data_loader, get_num_classes, find_annotation_file
from src.data.multi_dataset import get_multi_dataset, MultiDatasetLoader
from src.data import presets
from src.evaluation.evaluator import run_evaluation
from src.evaluation.dg_metrics import DomainGeneralizationMetrics, DGEvaluationReport


def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate object detection model")
    
    # Config file
    parser.add_argument('--config', type=str, default='configs/eval_config.yaml',
                       help='Path to config file')
    
    # Model
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--num-classes', type=int, default=None, help='Number of classes')
    
    # Dataset
    parser.add_argument('--data-path', type=str, default=None, help='Dataset path')
    parser.add_argument('--eval-set', type=str, default=None, choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    return parser


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_args(config, args):
    """Merge config file and command line arguments."""
    if args.checkpoint is not None:
        config['model']['checkpoint'] = args.checkpoint
    if args.num_classes is not None:
        config['model']['num_classes'] = args.num_classes
    if args.data_path is not None:
        config['dataset']['data_path'] = args.data_path
    if args.eval_set is not None:
        config['dataset']['eval_set'] = args.eval_set
    if args.output_dir is not None:
        config['output']['output_dir'] = args.output_dir
    if args.device is not None:
        config['device'] = args.device
    
    return config


def main(args):
    """Main evaluation function."""
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        config = merge_config_args(config, args)
    else:
        print(f"Warning: Config file {args.config} not found")
        config = {
            'model': {'checkpoint': args.checkpoint, 'num_classes': args.num_classes or 6},
            'dataset': {'data_path': args.data_path or './data/indraEye_dataset',
                       'eval_set': args.eval_set or 'val'},
            'evaluation': {'batch_size': 1, 'num_workers': 4},
            'output': {'output_dir': args.output_dir or './outputs/evaluation'},
            'device': args.device
        }
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine if using multi-dataset configuration
    use_multi_dataset = 'multi_datasets' in config['dataset']
    
    if use_multi_dataset:
        print("\n" + "="*60)
        print("MULTI-DATASET CONFIGURATION DETECTED")
        print("="*60)
        
        eval_set = config['dataset'].get('eval_set', 'val')
        
        # Verify class compatibility
        compatible, msg, categories = MultiDatasetLoader.verify_class_compatibility(
            config['dataset']['multi_datasets'], 
            image_set=eval_set
        )
        
        if not compatible:
            raise ValueError(f"Dataset compatibility check failed: {msg}")
        
        print(f"Compatibility check: {msg}")
        print(f"Unified categories: {categories}")
        
        # Get enabled dataset names
        enabled = [d['name'] for d in config['dataset']['multi_datasets'] if d.get('enabled', True)]
        print(f"Enabled datasets for evaluation: {enabled}")
        print("="*60 + "\n")
        
        # Auto-determine num_classes if not specified
        if 'num_classes' not in config['model'] or config['model']['num_classes'] is None:
            config['model']['num_classes'] = len(categories) + 1  # +1 for background
    else:
        # Legacy single dataset mode
        # Get number of classes from annotation file if not specified
        if 'num_classes' not in config['model'] or config['model']['num_classes'] is None:
            eval_set = config['dataset'].get('eval_set', 'val')
            ann_file = find_annotation_file(
                config['dataset']['data_path'], 
                eval_set
            )
            config['model']['num_classes'] = get_num_classes(ann_file)
    
    print(f"Number of classes: {config['model']['num_classes']}")
    
    # Load model
    checkpoint_path = config['model']['checkpoint']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from: {checkpoint_path}")
    model, checkpoint_data = get_model_instance(
        checkpoint_path=checkpoint_path,
        num_classes=config['model']['num_classes'],
        device=device,
        config=config
    )
    
    # Print checkpoint info
    if 'epoch' in checkpoint_data:
        print(f"Checkpoint from epoch: {checkpoint_data['epoch']}")
    
    # Load dataset
    eval_set = config['dataset'].get('eval_set', 'val')
    print(f"Loading {eval_set} dataset...")
    
    if use_multi_dataset:
        # Use multi-dataset loader
        dataset = get_multi_dataset(
            dataset_configs=config['dataset']['multi_datasets'],
            image_set=eval_set,
            transforms=presets.DetectionPresetEval(),
            with_masks=False
        )
    else:
        # Legacy single dataset loading
        dataset = get_coco_dataset(
            root=config['dataset']['data_path'],
            image_set=eval_set,
            transforms=presets.DetectionPresetEval()
        )
    
    print(f"Dataset: {len(dataset)} images")
    
    # Create data loader
    data_loader = create_data_loader(
        dataset,
        batch_size=config['evaluation'].get('batch_size', 1),
        num_workers=config['evaluation'].get('num_workers', 4),
        shuffle=False
    )
    
    # Setup output directory
    output_dir = config['output'].get('output_dir')
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    print("\n" + "="*70)
    print(f"Evaluating on {eval_set} set...")
    print("="*70 + "\n")
    
    metrics = run_evaluation(model, data_loader, device, output_dir)
    
    # Print overall results
    print("\n" + "="*70)
    print("Overall Evaluation Results:")
    print("="*70)
    for metric_name, value in metrics.items():
        if metric_name != 'class_wise':
            print(f"  {metric_name}: {value:.4f}")
    print("="*70 + "\n")
    
    # Print class-wise results
    if 'class_wise' in metrics and metrics['class_wise']:
        print("\n" + "="*70)
        print("Class-wise Evaluation Results:")
        print("="*70)
        print(f"{'Class Name':<30} {'AP':<10} {'AP50':<10} {'AP75':<10} {'Recall':<10}")
        print("-" * 70)
        
        # Sort by class name for consistent display
        sorted_classes = sorted(metrics['class_wise'].items())
        
        for class_name, class_metrics in sorted_classes:
            print(f"{class_name:<30} "
                  f"{class_metrics['AP']:<10.4f} "
                  f"{class_metrics['AP50']:<10.4f} "
                  f"{class_metrics['AP75']:<10.4f} "
                  f"{class_metrics['Recall']:<10.4f}")
        
        print("="*70 + "\n")
        
        # Compute and display class-wise statistics
        ap_values = [m['AP'] for m in metrics['class_wise'].values()]
        ap50_values = [m['AP50'] for m in metrics['class_wise'].values()]
        
        print("Class-wise Statistics:")
        print(f"  Number of classes: {len(metrics['class_wise'])}")
        print(f"  Mean AP: {sum(ap_values)/len(ap_values):.4f}")
        print(f"  Mean AP50: {sum(ap50_values)/len(ap50_values):.4f}")
        print(f"  Min AP: {min(ap_values):.4f}")
        print(f"  Max AP: {max(ap_values):.4f}")
        print("="*70 + "\n")
    
    # Compute Novel DG Metrics if configured
    compute_dg_metrics = config.get('evaluation', {}).get('compute_dg_metrics', False)
    
    if compute_dg_metrics and output_dir:
        print("\n" + "="*70)
        print("Computing Novel Domain Generalization Metrics...")
        print("="*70 + "\n")
        
        # Save basic metrics for DG analysis
        metrics_file = os.path.join(output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Standard metrics saved to: {metrics_file}")
        print("\nNote: Advanced DG metrics (DG-GAP, SI-Score, CDCI, etc.) require")
        print("multi-domain evaluation. Please see NOVEL_CONTRIBUTIONS_PROPOSAL.md")
        print("for integration instructions and usage examples.")
    
    if output_dir:
        print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
