"""Multi-dataset utilities for combining multiple COCO-format datasets."""

import os
import torch
from torch.utils.data import ConcatDataset
from .coco_utils import get_coco


class MultiDatasetLoader:
    """
    Load and combine multiple COCO-format datasets.
    
    Supports cross-combination training and testing scenarios:
    - Train on dataset A, test on dataset B
    - Train on dataset A+B, test on dataset C
    - Train on dataset A, test on dataset A+B
    """
    
    @staticmethod
    def load_dataset(dataset_configs, image_set, transforms=None, mode='instances', 
                     use_v2=False, with_masks=False):
        """
        Load one or more datasets and optionally combine them.
        
        Args:
            dataset_configs (list): List of dataset config dictionaries, each with:
                - 'name': Dataset identifier (e.g., 'indraEye', 'visdrone')
                - 'root': Root directory path
                - 'enabled': Whether to include this dataset
            image_set (str): 'train', 'val', or 'test'
            transforms: Transformations to apply
            mode (str): 'instances' for detection
            use_v2 (bool): Whether to use v2 transforms
            with_masks (bool): Whether to include masks
        
        Returns:
            dataset: Single dataset or ConcatDataset if multiple datasets enabled
        """
        datasets = []
        enabled_datasets = [cfg for cfg in dataset_configs if cfg.get('enabled', True)]
        
        if len(enabled_datasets) == 0:
            raise ValueError("At least one dataset must be enabled")
        
        for config in enabled_datasets:
            dataset_name = config['name']
            dataset_root = config['root']
            
            print(f"Loading {dataset_name} dataset from {dataset_root}...")
            
            # Load the dataset using existing get_coco function
            dataset = get_coco(
                root=dataset_root,
                image_set=image_set,
                transforms=transforms,
                mode=mode,
                use_v2=use_v2,
                with_masks=with_masks
            )
            
            # Add dataset identifier for tracking
            dataset.dataset_name = dataset_name
            datasets.append(dataset)
            
            print(f"  Loaded {len(dataset)} images from {dataset_name}")
        
        # If only one dataset, return it directly
        if len(datasets) == 1:
            return datasets[0]
        
        # Combine multiple datasets
        combined_dataset = ConcatDataset(datasets)
        combined_dataset.dataset_names = [cfg['name'] for cfg in enabled_datasets]
        
        total_images = sum(len(d) for d in datasets)
        print(f"Combined {len(enabled_datasets)} datasets: {total_images} total images")
        
        return combined_dataset
    
    @staticmethod
    def get_dataset_info(annotation_file):
        """
        Get information about a dataset from its annotation file.
        
        Args:
            annotation_file (str): Path to annotation JSON file
        
        Returns:
            dict: Dataset information including categories, num_images, etc.
        """
        import json
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        info = {
            'num_categories': len(data.get('categories', [])),
            'num_images': len(data.get('images', [])),
            'num_annotations': len(data.get('annotations', [])),
            'categories': {cat['id']: cat['name'] for cat in data.get('categories', [])},
            'category_list': data.get('categories', [])
        }
        
        return info
    
    @staticmethod
    def verify_class_compatibility(dataset_configs, image_set='train'):
        """
        Verify that all datasets have compatible class structures.
        
        Args:
            dataset_configs (list): List of dataset configurations
            image_set (str): Which annotation set to check
        
        Returns:
            tuple: (compatible, message, unified_categories)
        """
        enabled_configs = [cfg for cfg in dataset_configs if cfg.get('enabled', True)]
        
        if len(enabled_configs) == 0:
            return False, "No datasets enabled", None
        
        if len(enabled_configs) == 1:
            # Single dataset, always compatible
            config = enabled_configs[0]
            ann_file = os.path.join(config['root'], 'annotations', f'{image_set}.json')
            info = MultiDatasetLoader.get_dataset_info(ann_file)
            return True, "Single dataset", info['categories']
        
        # Check multiple datasets
        all_categories = []
        
        for config in enabled_configs:
            ann_file = os.path.join(config['root'], 'annotations', f'{image_set}.json')
            
            if not os.path.exists(ann_file):
                return False, f"Annotation file not found: {ann_file}", None
            
            info = MultiDatasetLoader.get_dataset_info(ann_file)
            all_categories.append({
                'name': config['name'],
                'categories': info['categories']
            })
        
        # Compare categories
        base_cats = all_categories[0]['categories']
        
        for dataset_cats in all_categories[1:]:
            if dataset_cats['categories'] != base_cats:
                # Check if categories match but with different IDs
                base_names = sorted(base_cats.values())
                curr_names = sorted(dataset_cats['categories'].values())
                
                if base_names != curr_names:
                    return False, f"Category mismatch between datasets", None
        
        # All datasets have compatible categories
        return True, "All datasets compatible", base_cats


def get_multi_dataset(dataset_configs, image_set, transforms=None, 
                      mode='instances', use_v2=False, with_masks=False):
    """
    Convenience function to load multi-dataset.
    
    Args:
        dataset_configs (list): List of dataset configurations
        image_set (str): 'train', 'val', or 'test'
        transforms: Data transformations
        mode (str): 'instances' for detection
        use_v2 (bool): Use v2 transforms
        with_masks (bool): Include masks
    
    Returns:
        dataset: Combined or single dataset
    """
    return MultiDatasetLoader.load_dataset(
        dataset_configs=dataset_configs,
        image_set=image_set,
        transforms=transforms,
        mode=mode,
        use_v2=use_v2,
        with_masks=with_masks
    )
