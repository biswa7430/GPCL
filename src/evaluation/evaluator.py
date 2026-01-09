"""Model evaluation utilities."""

import time
import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .coco_eval import CocoEvaluator
from ..data.coco_utils import get_coco_api_from_dataset
from ..utils.metrics import MetricLogger


def evaluate_model(model, data_loader, device):
    """
    Evaluate model on dataset.
    
    Args:
        model: The detection model
        data_loader: Evaluation data loader
        device: Device to evaluate on
    
    Returns:
        CocoEvaluator: Evaluation results
    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    with torch.inference_mode():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"]: output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # Accumulate predictions and evaluate
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    
    return coco_evaluator


def _get_iou_types(model):
    """Get IoU types based on model architecture."""
    import torchvision
    
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def run_evaluation(model, data_loader, device, output_dir=None):
    """
    Run full evaluation and optionally save results.
    
    Args:
        model: The detection model
        data_loader: Evaluation data loader
        device: Device to evaluate on
        output_dir (str): Optional directory to save results
    
    Returns:
        dict: Evaluation metrics including class-wise metrics
    """
    print("Running evaluation...")
    coco_evaluator = evaluate_model(model, data_loader, device)
    
    # Extract overall metrics
    stats = coco_evaluator.coco_eval['bbox'].stats
    metrics = {
        'mAP': stats[0],
        'mAP_50': stats[1],
        'mAP_75': stats[2],
        'mAP_small': stats[3],
        'mAP_medium': stats[4],
        'mAP_large': stats[5],
    }
    
    # Extract class-wise metrics
    class_wise_metrics = get_class_wise_metrics(coco_evaluator, data_loader)
    metrics['class_wise'] = class_wise_metrics
    
    if output_dir:
        import json
        os.makedirs(output_dir, exist_ok=True)
        
        # Save overall results
        results_file = os.path.join(output_dir, 'eval_results.json')
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved evaluation results to {results_file}")
        
        # Save class-wise results in a separate file for easier reading
        class_results_file = os.path.join(output_dir, 'class_wise_results.json')
        with open(class_results_file, 'w') as f:
            json.dump(class_wise_metrics, f, indent=4)
        print(f"Saved class-wise results to {class_results_file}")
        
        # Also save class-wise results as CSV for easier analysis
        csv_file = os.path.join(output_dir, 'class_wise_results.csv')
        save_class_wise_csv(class_wise_metrics, csv_file)
        print(f"Saved class-wise results (CSV) to {csv_file}")
    
    return metrics


def save_class_wise_csv(class_wise_metrics, csv_file):
    """
    Save class-wise metrics to CSV file.
    
    Args:
        class_wise_metrics (dict): Dictionary of class-wise metrics
        csv_file (str): Path to save CSV file
    """
    import csv
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Class Name', 'Category ID', 'AP', 'AP50', 'AP75', 'Recall'])
        
        # Sort by class name
        sorted_classes = sorted(class_wise_metrics.items())
        
        # Write data rows
        for class_name, metrics in sorted_classes:
            writer.writerow([
                class_name,
                metrics['category_id'],
                f"{metrics['AP']:.4f}",
                f"{metrics['AP50']:.4f}",
                f"{metrics['AP75']:.4f}",
                f"{metrics['Recall']:.4f}"
            ])


def get_class_wise_metrics(coco_evaluator, data_loader):
    """
    Extract per-class AP metrics from COCO evaluator.
    
    Args:
        coco_evaluator: CocoEvaluator instance after evaluation
        data_loader: Data loader used for evaluation (to get category info)
    
    Returns:
        dict: Class-wise metrics including AP, AP50, AP75 for each class
    """
    import json
    
    # Get COCO evaluator for bbox
    coco_eval = coco_evaluator.coco_eval['bbox']
    
    # Get category information from dataset
    coco_gt = coco_eval.cocoGt
    
    # Get category names
    if hasattr(coco_gt, 'cats'):
        categories = coco_gt.cats
    else:
        # Fallback: try to get from dataset
        categories = {}
        if hasattr(data_loader.dataset, 'coco') and hasattr(data_loader.dataset.coco, 'cats'):
            categories = data_loader.dataset.coco.cats
    
    # Compute per-category metrics
    class_wise_metrics = {}
    
    # Get precision values (shape: [TxRxKxAxM])
    # T: IoU thresholds, R: recall thresholds, K: categories, A: area ranges, M: max detections
    precisions = coco_eval.eval['precision']
    
    # Get recalls
    recalls = coco_eval.eval['recall']
    
    # IoU thresholds
    iou_thresholds = coco_eval.params.iouThrs
    
    # Get category IDs
    cat_ids = coco_eval.params.catIds
    
    for idx, cat_id in enumerate(cat_ids):
        # Get category name
        if cat_id in categories:
            cat_name = categories[cat_id]['name']
        else:
            cat_name = f"class_{cat_id}"
        
        # AP at IoU=0.50:0.95 (average over IoU thresholds and recall thresholds)
        # precisions shape: [num_iou_thresholds, num_recall_thresholds, num_categories, num_area_ranges, num_max_dets]
        # We use area range index 0 (all) and max det index -1 (last, typically 100)
        ap_all = precisions[:, :, idx, 0, -1]
        ap = np.mean(ap_all[ap_all > -1])  # -1 means no predictions
        
        # AP50 (IoU=0.50)
        iou_50_idx = np.where(np.abs(np.array(iou_thresholds) - 0.5) < 0.01)[0]
        if len(iou_50_idx) > 0:
            ap50_vals = precisions[iou_50_idx[0], :, idx, 0, -1]
            ap50 = np.mean(ap50_vals[ap50_vals > -1])
        else:
            ap50 = -1
        
        # AP75 (IoU=0.75)
        iou_75_idx = np.where(np.abs(np.array(iou_thresholds) - 0.75) < 0.01)[0]
        if len(iou_75_idx) > 0:
            ap75_vals = precisions[iou_75_idx[0], :, idx, 0, -1]
            ap75 = np.mean(ap75_vals[ap75_vals > -1])
        else:
            ap75 = -1
        
        # Recall at different IoU thresholds
        recall_all = recalls[:, idx, 0, -1]
        recall = np.mean(recall_all[recall_all > -1])
        
        # Store metrics
        class_wise_metrics[cat_name] = {
            'category_id': int(cat_id),
            'AP': float(ap) if not np.isnan(ap) else 0.0,
            'AP50': float(ap50) if not np.isnan(ap50) else 0.0,
            'AP75': float(ap75) if not np.isnan(ap75) else 0.0,
            'Recall': float(recall) if not np.isnan(recall) else 0.0,
        }
    
    return class_wise_metrics
