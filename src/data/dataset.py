"""Dataset and data loading utilities for COCO-format datasets."""

import os
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from .coco_utils import get_coco, collate_fn_with_validation
from .multi_dataset import get_multi_dataset, MultiDatasetLoader


def get_coco_dataset(root, image_set, transforms=None, mode='instances', use_v2=False, with_masks=False):
    """
    Load COCO-format dataset.
    
    Args:
        root (str): Root directory of the dataset
        image_set (str): 'train' or 'val'
        transforms: Transformations to apply
        mode (str): 'instances' for detection, 'person_keypoints' for keypoints
        use_v2 (bool): Whether to use v2 transforms
        with_masks (bool): Whether to include masks
    
    Returns:
        dataset: COCO dataset
    """
    dataset = get_coco(
        root=root,
        image_set=image_set,
        transforms=transforms,
        mode=mode,
        use_v2=use_v2,
        with_masks=with_masks
    )
    
    return dataset


def create_data_loader(dataset, batch_size, num_workers=4, 
                       shuffle=True, collate_fn=None, sampler=None, 
                       validate_boxes=True, min_box_size=1.0):
    """
    Create DataLoader for the dataset.
    
    Args:
        dataset: Dataset to load
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        shuffle (bool): Whether to shuffle data
        collate_fn: Custom collate function. If None and validate_boxes=True,
                   uses collate_fn_with_validation
        sampler: Custom sampler
        validate_boxes (bool): Whether to validate and filter invalid boxes (default: True)
        min_box_size (float): Minimum box size in pixels when validating (default: 1.0)
    
    Returns:
        DataLoader: Data loader instance
    """
    if collate_fn is None:
        if validate_boxes:
            # Use safe collate function that validates boxes
            collate_fn = lambda batch: collate_fn_with_validation(batch, min_box_size=min_box_size)
        else:
            collate_fn = lambda batch: tuple(zip(*batch))
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=sampler,
        pin_memory=True
    )
    
    return data_loader


def find_annotation_file(data_path, image_set, mode='instances'):
    """
    Find annotation file supporting both naming conventions.
    
    Args:
        data_path (str): Root path of dataset
        image_set (str): 'train', 'val', or 'test'
        mode (str): 'instances' for detection
    
    Returns:
        str: Path to annotation file
    """
    # Try simple format first (train.json)
    simple_ann_file = os.path.join(data_path, 'annotations', f'{image_set}.json')
    
    # Try standard COCO format (instances_train2017.json)
    coco_ann_file = os.path.join(data_path, 'annotations', f'{mode}_{image_set}2017.json')
    
    if os.path.exists(simple_ann_file):
        return simple_ann_file
    elif os.path.exists(coco_ann_file):
        return coco_ann_file
    else:
        # Return simple format as fallback (will raise error if not found)
        return simple_ann_file


def get_num_classes(annotation_file):
    """
    Get number of classes from COCO annotation file.
    
    Args:
        annotation_file (str): Path to annotation JSON file
    
    Returns:
        int: Number of classes (including background)
    """
    import json
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Get unique category IDs
    categories = data.get('categories', [])
    num_classes = len(categories) + 1  # +1 for background
    
    return num_classes


def get_class_names(annotation_file):
    """
    Load class names from COCO annotation file.
    
    Args:
        annotation_file (str): Path to annotation JSON file
    
    Returns:
        dict: Mapping from category ID to class name
    """
    import json
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Create category id to name mapping
    categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    
    return categories
