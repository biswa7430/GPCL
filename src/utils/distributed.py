"""Distributed training utilities."""

import os
import torch
import torch.distributed as dist


def init_distributed_mode(args):
    """
    Initialize distributed training mode.
    
    Args:
        args: Arguments containing distributed training settings
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    torch.distributed.barrier()


def is_main_process():
    """Check if current process is main process."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def save_on_master(*args, **kwargs):
    """Save only on master process."""
    if is_main_process():
        torch.save(*args, **kwargs)


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
