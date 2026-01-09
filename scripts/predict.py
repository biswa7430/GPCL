#!/usr/bin/env python3
"""
Prediction/Inference script for object detection models.
Run predictions on images and visualize results.
"""

import argparse
import os
import sys
import yaml

import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.model_builder import get_model_instance
from src.data.dataset import get_class_names, find_annotation_file
from src.data.multi_dataset import MultiDatasetLoader


def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run predictions with trained model")
    
    # Config file
    parser.add_argument('--config', type=str, default='configs/predict_config.yaml',
                       help='Path to config file')
    
    # Model
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--num-classes', type=int, default=None, help='Number of classes')
    
    # Input
    parser.add_argument('--image', type=str, default=None, help='Path to single image')
    parser.add_argument('--data-path', type=str, default=None, help='Dataset path (for batch prediction)')
    parser.add_argument('--image-set', type=str, default=None, choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--num-images', type=int, default=None, help='Number of images to predict')
    
    # Prediction
    parser.add_argument('--score-threshold', type=float, default=None, help='Confidence threshold')
    
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
    if args.image is not None:
        config['prediction']['single_image'] = args.image
    if args.data_path is not None:
        config['dataset']['data_path'] = args.data_path
    if args.image_set is not None:
        config['dataset']['image_set'] = args.image_set
    if args.num_images is not None:
        config['prediction']['num_images'] = args.num_images
    if args.score_threshold is not None:
        config['prediction']['score_threshold'] = args.score_threshold
    if args.output_dir is not None:
        config['output']['output_dir'] = args.output_dir
    if args.device is not None:
        config['device'] = args.device
    
    return config


def predict_image(model, image_path, device, class_names, score_threshold=0.5,
                  output_dir=None, config=None):
    """
    Run prediction on a single image.
    
    Args:
        model: Detection model
        image_path: Path to image file
        device: Device to run on
        class_names: Dictionary mapping class IDs to names
        score_threshold: Minimum confidence score
        output_dir: Directory to save results
        config: Configuration dictionary
    
    Returns:
        PIL Image with predictions drawn
    """
    # Read image
    img = read_image(image_path)
    
    # Convert to float and normalize
    img_float = img.float() / 255.0
    
    # Run inference
    model.eval()
    with torch.no_grad():
        img_tensor = img_float.to(device)
        predictions = model([img_tensor])[0]
    
    # Filter predictions by score threshold
    scores = predictions['scores'].cpu()
    boxes = predictions['boxes'].cpu()
    labels = predictions['labels'].cpu()
    
    mask = scores > score_threshold
    filtered_boxes = boxes[mask]
    filtered_labels = labels[mask]
    filtered_scores = scores[mask]
    
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Detected {len(filtered_boxes)} objects with confidence > {score_threshold}")
    
    # Create labels with class names and scores
    label_texts = []
    for label, score in zip(filtered_labels, filtered_scores):
        class_name = class_names.get(label.item(), f"Class_{label.item()}")
        label_texts.append(f"{class_name}: {score:.2f}")
        print(f"  - {class_name}: {score:.3f}")
    
    # Draw bounding boxes
    if len(filtered_boxes) > 0:
        vis_config = config.get('visualization', {}) if config else {}
        
        # Define color palette for different classes (RGB tuples)
        color_palette = [
            (255, 0, 0),      # Red - people
            (0, 255, 0),      # Green - bus
            (0, 0, 255),      # Blue - car
            (255, 255, 0),    # Yellow - motorcycle
            (255, 0, 255),    # Magenta - truck
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 255, 128),    # Spring green
            (255, 0, 128),    # Rose
        ]
        
        # Assign colors based on class labels
        box_colors = [color_palette[label.item() % len(color_palette)] for label in filtered_labels]
        
        img_with_boxes = draw_bounding_boxes(
            img,
            filtered_boxes,
            labels=label_texts if vis_config.get('show_labels', True) else None,
            colors=box_colors,
            width=vis_config.get('box_width', 3),
            font_size=vis_config.get('font_size', 60)  # Increased from 20 to 30
        )
        img_pil = to_pil_image(img_with_boxes)
    else:
        img_pil = to_pil_image(img)
        print("  No objects detected above threshold")
    
    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"pred_{os.path.basename(image_path)}")
        img_pil.save(output_path)
        print(f"Saved prediction to: {output_path}")
    
    return img_pil


def predict_dataset(model, data_path, image_set, device, class_names,
                    score_threshold=0.5, num_images=10, output_dir=None, config=None):
    """
    Run predictions on multiple images from dataset.
    
    Args:
        model: Detection model
        data_path: Path to dataset directory
        image_set: 'train', 'val', or 'test'
        device: Device to run on
        class_names: Dictionary mapping class IDs to names
        score_threshold: Minimum confidence score
        num_images: Number of images to process (0 = all)
        output_dir: Directory to save results
        config: Configuration dictionary
    """
    # Try simple format first (train/)
    img_folder = os.path.join(data_path, image_set)
    
    # If not found, try COCO format (train2017/)
    if not os.path.exists(img_folder):
        img_folder = os.path.join(data_path, f"{image_set}2017")
    
    if not os.path.exists(img_folder):
        print(f"Error: Image folder not found: {img_folder}")
        return
    
    # Get all images
    image_files = sorted([f for f in os.listdir(img_folder) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # Limit number of images
    if num_images > 0:
        image_files = image_files[:num_images]
    
    print(f"\nRunning predictions on {len(image_files)} images from {image_set} set...")
    print(f"Score threshold: {score_threshold}")
    print("=" * 70)
    
    for img_file in image_files:
        img_path = os.path.join(img_folder, img_file)
        try:
            predict_image(model, img_path, device, class_names, 
                        score_threshold, output_dir, config)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")


def main(args):
    """Main prediction function."""
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        config = merge_config_args(config, args)
    else:
        print(f"Warning: Config file {args.config} not found")
        config = {
            'model': {'checkpoint': args.checkpoint, 'num_classes': args.num_classes or 6},
            'dataset': {'data_path': args.data_path or './data/indraEye_dataset',
                       'image_set': args.image_set or 'val'},
            'prediction': {'score_threshold': args.score_threshold or 0.5,
                          'num_images': args.num_images or 10,
                          'single_image': args.image},
            'output': {'output_dir': args.output_dir or './outputs/predictions'},
            'device': args.device
        }
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine if using multi-dataset configuration
    use_multi_dataset = 'multi_datasets' in config['dataset']
    
    # Load model
    checkpoint_path = config['model']['checkpoint']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from: {checkpoint_path}")
    model, _ = get_model_instance(
        checkpoint_path=checkpoint_path,
        num_classes=config['model']['num_classes'],
        device=device,
        config=config
    )
    
    # Load class names from appropriate annotation file
    image_set = config['dataset'].get('image_set', 'val')
    
    if use_multi_dataset:
        # Get class names from the first enabled dataset
        enabled_datasets = [d for d in config['dataset']['multi_datasets'] if d.get('enabled', True)]
        if enabled_datasets:
            dataset_root = enabled_datasets[0]['root']
            dataset_name = enabled_datasets[0]['name']
            print(f"Loading class names from {dataset_name} dataset")
            ann_file = find_annotation_file(dataset_root, image_set)
        else:
            raise ValueError("No datasets enabled in multi_datasets configuration")
    else:
        # Legacy single dataset
        ann_file = find_annotation_file(config['dataset']['data_path'], image_set)
    
    if os.path.exists(ann_file):
        class_names = get_class_names(ann_file)
        print(f"Loaded {len(class_names)} class names")
    else:
        print(f"Warning: Annotation file not found: {ann_file}")
        class_names = {}
    
    # Setup output directory
    output_dir = config['output'].get('output_dir')
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run predictions
    single_image = config['prediction'].get('single_image')
    score_threshold = config['prediction'].get('score_threshold', 0.5)
    
    if single_image:
        # Predict on single image
        if not os.path.exists(single_image):
            raise FileNotFoundError(f"Image not found: {single_image}")
        print(f"\nPredicting on single image: {single_image}")
        predict_image(model, single_image, device, class_names, 
                     score_threshold, output_dir, config)
    else:
        # Predict on dataset
        num_images = config['prediction'].get('num_images', 10)
        
        if use_multi_dataset:
            # Multi-dataset prediction
            enabled_datasets = [d for d in config['dataset']['multi_datasets'] if d.get('enabled', True)]
            
            for dataset_cfg in enabled_datasets:
                dataset_name = dataset_cfg['name']
                dataset_root = dataset_cfg['root']
                
                print(f"\n{'='*70}")
                print(f"Running predictions on {dataset_name} dataset")
                print(f"{'='*70}")
                
                # Create dataset-specific output directory
                dataset_output_dir = os.path.join(output_dir, dataset_name) if output_dir else None
                if dataset_output_dir:
                    os.makedirs(dataset_output_dir, exist_ok=True)
                
                predict_dataset(model, dataset_root, image_set,
                               device, class_names, score_threshold, num_images, 
                               dataset_output_dir, config)
        else:
            # Single dataset prediction
            predict_dataset(model, config['dataset']['data_path'], image_set,
                           device, class_names, score_threshold, num_images, 
                           output_dir, config)
    
    print("\nDone!")
    if output_dir:
        print(f"Predictions saved to: {output_dir}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
