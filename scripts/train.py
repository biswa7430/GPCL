#!/usr/bin/env python3
"""
Training script for object detection models.
Supports both single-GPU and distributed multi-GPU training.
"""

import argparse
import datetime
import os
import sys
import time
import yaml

import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.model_builder import build_model
from src.models.shm_detector import wrap_model_with_shm
from src.data.dataset import get_coco_dataset, create_data_loader, get_num_classes, find_annotation_file
from src.data.multi_dataset import get_multi_dataset, MultiDatasetLoader
from src.data import presets
from src.utils.train_utils import train_one_epoch, save_checkpoint, load_checkpoint
from src.utils.shm_train_utils import train_one_epoch_shm
from src.utils.distributed import init_distributed_mode, is_main_process, cleanup_distributed
from src.evaluation.evaluator import evaluate_model
from src.utils import metrics


def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train object detection model")
    
    # Config file
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    
    # Model
    parser.add_argument('--model', type=str, default=None, help='Model name')
    parser.add_argument('--num-classes', type=int, default=None, help='Number of classes')
    
    # Dataset
    parser.add_argument('--data-path', type=str, default=None, help='Dataset path')
    parser.add_argument('--dataset', type=str, default='coco', help='Dataset name')
    
    # Training
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--workers', type=int, default=None, help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    # Distributed
    parser.add_argument('--world-size', type=int, default=1, help='Number of distributed processes')
    parser.add_argument('--dist-url', type=str, default='env://', help='URL for distributed training')
    
    # Mixed precision
    parser.add_argument('--amp', action='store_true', help='Use mixed precision training')
    
    # Misc
    parser.add_argument('--print-freq', type=int, default=20, help='Print frequency')
    parser.add_argument('--test-only', action='store_true', help='Only test the model')
    
    return parser


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_args(config, args):
    """Merge config file and command line arguments."""
    # Command line args override config file
    if args.model is not None:
        config['model']['name'] = args.model
    if args.num_classes is not None:
        config['model']['num_classes'] = args.num_classes
    if args.data_path is not None:
        config['dataset']['data_path'] = args.data_path
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.workers is not None:
        config['training']['num_workers'] = args.workers
    if args.output_dir is not None:
        config['output']['output_dir'] = args.output_dir
    if args.device is not None:
        config['device'] = args.device
    
    return config


def get_transform(is_train, config):
    """Get data transforms."""
    use_v2 = config.get('use_v2', False)
    backend = config.get('backend', 'pil')
    
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=config.get('training', {}).get('data_augmentation', 'hflip'),
            backend=backend,
            use_v2=use_v2
        )
    else:
        return presets.DetectionPresetEval(backend=backend, use_v2=use_v2)


def main(args):
    """Main training function."""
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        config = merge_config_args(config, args)
    else:
        print(f"Warning: Config file {args.config} not found, using command line args only")
        # Create minimal config from args
        config = {
            'model': {'name': args.model or 'fasterrcnn_resnet50_fpn', 
                     'num_classes': args.num_classes or 91},
            'dataset': {'data_path': args.data_path or './data/indraEye_dataset'},
            'training': {'epochs': args.epochs or 26, 'batch_size': args.batch_size or 2,
                        'learning_rate': args.lr or 0.02, 'num_workers': args.workers or 4},
            'output': {'output_dir': args.output_dir or './checkpoints'},
            'device': args.device
        }
    
    # Setup output directory
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine if using multi-dataset configuration
    use_multi_dataset = 'multi_datasets' in config['dataset']
    
    if use_multi_dataset:
        print("\n" + "="*60)
        print("MULTI-DATASET CONFIGURATION DETECTED")
        print("="*60)
        
        # Verify class compatibility
        compatible, msg, categories = MultiDatasetLoader.verify_class_compatibility(
            config['dataset']['multi_datasets'], 
            image_set=config['dataset'].get('train_set', 'train')
        )
        
        if not compatible:
            raise ValueError(f"Dataset compatibility check failed: {msg}")
        
        print(f"Compatibility check: {msg}")
        print(f"Unified categories: {categories}")
        
        # Get enabled dataset names
        enabled = [d['name'] for d in config['dataset']['multi_datasets'] if d.get('enabled', True)]
        print(f"Enabled datasets for training: {enabled}")
        print("="*60 + "\n")
        
        # Auto-determine num_classes if not specified
        if 'num_classes' not in config['model'] or config['model']['num_classes'] is None:
            config['model']['num_classes'] = len(categories) + 1  # +1 for background
    else:
        # Legacy single dataset mode
        # Get number of classes from annotation file if not specified
        if 'num_classes' not in config['model'] or config['model']['num_classes'] is None:
            ann_file = find_annotation_file(
                config['dataset']['data_path'], 
                config['dataset'].get('train_set', 'train')
            )
            config['model']['num_classes'] = get_num_classes(ann_file)
    
    print(f"Number of classes: {config['model']['num_classes']}")
    
    # Build model
    print(f"Building model: {config['model']['name']}")
    model = build_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained_backbone=config['model'].get('pretrained_backbone', True),
        trainable_backbone_layers=config['model'].get('trainable_backbone_layers'),
        rpn_score_thresh=config['model'].get('rpn_score_thresh')
    )
    model.to(device)

    # Optionally wrap model with SHM (does nothing if shm.enable is False)
    model = wrap_model_with_shm(model, config)
    
    # Load datasets
    print("Loading datasets...")
    use_v2 = config.get('use_v2', False)
    
    if use_multi_dataset:
        # Use multi-dataset loader
        dataset_train = get_multi_dataset(
            dataset_configs=config['dataset']['multi_datasets'],
            image_set=config['dataset'].get('train_set', 'train'),
            transforms=get_transform(True, config),
            use_v2=use_v2,
            with_masks=False
        )
        
        dataset_val = get_multi_dataset(
            dataset_configs=config['dataset']['multi_datasets'],
            image_set=config['dataset'].get('val_set', 'val'),
            transforms=get_transform(False, config),
            use_v2=use_v2,
            with_masks=False
        )
    else:
        # Legacy single dataset loading
        dataset_train = get_coco_dataset(
            root=config['dataset']['data_path'],
            image_set='train',
            transforms=get_transform(True, config),
            use_v2=use_v2,
            with_masks=False
        )
        
        dataset_val = get_coco_dataset(
            root=config['dataset']['data_path'],
            image_set='val',
            transforms=get_transform(False, config),
            use_v2=use_v2,
            with_masks=False
        )
    
    print(f"Train dataset: {len(dataset_train)} images")
    print(f"Val dataset: {len(dataset_val)} images")
    
    # Create data loaders
    train_loader = create_data_loader(
        dataset_train,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=True
    )
    
    val_loader = create_data_loader(
        dataset_val,
        batch_size=1,
        num_workers=config['training']['num_workers'],
        shuffle=False
    )
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config['training']['learning_rate'],
        momentum=config['training'].get('momentum', 0.9),
        weight_decay=config['training'].get('weight_decay', 0.0001)
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['training'].get('lr_steps', [16, 22]),
        gamma=config['training'].get('lr_gamma', 0.1)
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.amp or config['training'].get('amp', False) else None
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(model, args.resume, optimizer, device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    # Test only mode
    if args.test_only:
        print("Running evaluation...")
        evaluate_model(model, val_loader, device)
        return
    
    # Training loop
    print(f"Starting training for {config['training']['epochs']} epochs...")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    
    # Get validation frequency from config
    val_freq = config['training'].get('val_freq', 5)
    print(f"Validation frequency: every {val_freq} epoch(s)")
    
    # Track best validation mAP
    best_map = 0.0
    best_epoch = -1
    
    start_time = time.time()
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train for one epoch (use SHM-aware loop when enabled)
        if config.get('shm', {}).get('enable', False):
            train_metrics = train_one_epoch_shm(
                model, optimizer, train_loader, device, epoch, config,
                scaler=scaler, print_freq=args.print_freq
            )
        else:
            train_metrics = train_one_epoch(
                model, optimizer, train_loader, device, epoch,
                print_freq=args.print_freq, scaler=scaler
            )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set at specified frequency
        current_map = None
        if (epoch + 1) % val_freq == 0 or epoch == config['training']['epochs'] - 1:
            print(f"\nRunning validation at epoch {epoch + 1}...")
            coco_evaluator = evaluate_model(model, val_loader, device)
            
            # Extract mAP (AP @ IoU=0.50:0.95)
            if coco_evaluator is not None and hasattr(coco_evaluator, 'coco_eval'):
                stats = coco_evaluator.coco_eval['bbox'].stats
                current_map = stats[0]  # mAP @ IoU=0.50:0.95
                print(f"Validation mAP: {current_map:.4f}")
        
        # Save checkpoint logic
        if is_main_process():
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'config': config,
                'best_map': best_map
            }
            
            # Always save the last checkpoint (overwrites each epoch)
            last_checkpoint_path = os.path.join(config['output']['output_dir'], 'checkpoint_last.pth')
            torch.save(checkpoint, last_checkpoint_path)
            
            # Save best model if current mAP is better
            if current_map is not None and current_map > best_map:
                best_map = current_map
                best_epoch = epoch
                best_checkpoint_path = os.path.join(config['output']['output_dir'], 'checkpoint_best.pth')
                torch.save(checkpoint, best_checkpoint_path)
                print(f"★ New best model saved! mAP: {best_map:.4f} at epoch {epoch + 1}")
        
        # Print best performance so far
        if best_epoch >= 0:
            print(f"Best mAP so far: {best_map:.4f} at epoch {best_epoch + 1}")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\nTraining time: {total_time_str}')
    
    # Print best model summary
    if best_epoch >= 0:
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Best model saved at epoch: {best_epoch + 1}")
        print(f"Best validation mAP: {best_map:.4f}")
        print(f"Best model path: {os.path.join(config['output']['output_dir'], 'checkpoint_best.pth')}")
        print(f"Last model path: {os.path.join(config['output']['output_dir'], 'checkpoint_last.pth')}")
        print("="*60 + "\n")
    
    # Final evaluation
    print("\n" + "="*50)
    print("Running final evaluation on validation set...")
    print("="*50)
    evaluate_model(model, val_loader, device)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
