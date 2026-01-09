"""Training utilities and helper functions."""

import math
import sys
import time
import torch
from .metrics import MetricLogger, SmoothedValue


def train_one_epoch(model, optimizer, data_loader, device, epoch, 
                    print_freq=20, scaler=None):
    """
    Train model for one epoch.
    
    Args:
        model: The detection model
        optimizer: Optimizer
        data_loader: Training data loader
        device: Device to train on
        epoch (int): Current epoch number
        print_freq (int): Print frequency
        scaler: GradScaler for mixed precision training
    
    Returns:
        MetricLogger: Training metrics
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # Warmup learning rate scheduler for first epoch
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Reduce losses over all GPUs for logging
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def save_checkpoint(model, optimizer, epoch, args, filename='checkpoint.pth'):
    """
    Save training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer
        epoch (int): Current epoch
        args: Training arguments
        filename (str): Checkpoint filename
    """
    import os
    
    checkpoint = {
        'model': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    
    filepath = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cuda'):
    """
    Load checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path (str): Path to checkpoint
        optimizer: Optional optimizer to load state
        device (str): Device to map tensors to
    
    Returns:
        dict: Checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle DDP wrapped models
    state_dict = checkpoint.get('model', checkpoint)
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint
