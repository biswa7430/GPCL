"""Custom RoI Head wrapper to extract intermediate features for GPCL."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict


class RoIHeadWithFeatures(nn.Module):
    """
    Wrapper around Faster R-CNN's RoI heads that captures intermediate features.
    
    This is necessary for GPCL which needs the actual box features (256-D vectors)
    rather than just the final predictions.
    
    Architecture:
        RoI Align → Box Head (FC layers) → [256-D features] → Box Predictor
                                              ↑
                                         Extract here!
    """
    
    def __init__(self, original_roi_heads):
        super().__init__()
        self.original_roi_heads = original_roi_heads
        self.captured_features = {}  # Store features during forward pass
        self.capture_mode = False
        
    def enable_feature_capture(self):
        """Enable capturing of intermediate features."""
        self.capture_mode = True
        
    def disable_feature_capture(self):
        """Disable capturing and clear stored features."""
        self.capture_mode = False
        self.captured_features = {}
        
    def get_captured_features(self) -> Dict[str, torch.Tensor]:
        """
        Get captured features from last forward pass.
        
        Returns:
            dict with keys:
                'box_features': [N, 256] tensor of RoI features
                'boxes': [N, 4] corresponding box coordinates
                'image_ids': [N] which image each box belongs to
        """
        return self.captured_features
    
    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Forward pass with feature capture.
        
        Args:
            features: FPN features from backbone
            proposals: RPN proposals
            image_shapes: Image sizes
            targets: Ground truth (training only)
            
        Returns:
            Same as original RoI heads: (detections, losses) or ([], detections)
        """
        # Call original forward, but intercept box_head output
        if self.training and targets is not None:
            return self._forward_train(features, proposals, image_shapes, targets)
        else:
            return self._forward_test(features, proposals, image_shapes)
    
    def _forward_train(self, features, proposals, image_shapes, targets):
        """Training forward with feature capture."""
        roi_heads = self.original_roi_heads
        
        # Match targets to proposals (from torchvision implementation)
        proposals, matched_idxs, labels, regression_targets = \
            roi_heads.select_training_samples(proposals, targets)
        
        # Extract features using box_roi_pool
        box_features = roi_heads.box_roi_pool(features, proposals, image_shapes)
        
        # Pass through box_head to get 256-D features
        box_features_flat = roi_heads.box_head(box_features)
        
        # CAPTURE FEATURES HERE
        if self.capture_mode:
            self._capture_training_features(box_features_flat, proposals)
        
        # Continue to predictions
        class_logits, box_regression = roi_heads.box_predictor(box_features_flat)
        
        # Compute losses - concatenate batch results first
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        
        # Classification loss
        classification_loss = torch.nn.functional.cross_entropy(class_logits, labels)
        
        # Box regression loss (only for foreground)
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
        
        box_loss = torch.nn.functional.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1.0,
            reduction='sum',
        )
        box_loss = box_loss / labels.numel()
        
        losses = {
            "loss_classifier": classification_loss,
            "loss_box_reg": box_loss
        }
        
        # Return format for torchvision: (detections, losses) in training
        return [], losses
    
    def _forward_test(self, features, proposals, image_shapes):
        """Test forward with feature capture."""
        roi_heads = self.original_roi_heads
        
        # Extract features
        box_features = roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features_flat = roi_heads.box_head(box_features)
        
        # CAPTURE FEATURES HERE
        if self.capture_mode:
            self._capture_test_features(box_features_flat, proposals)
        
        # Get predictions
        class_logits, box_regression = roi_heads.box_predictor(box_features_flat)
        
        # Post-process
        boxes, scores, labels = roi_heads.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )
        
        detections = []
        for boxes_per_image, scores_per_image, labels_per_image in zip(boxes, scores, labels):
            detections.append({
                "boxes": boxes_per_image,
                "scores": scores_per_image,
                "labels": labels_per_image,
            })
        
        # Return format for torchvision: (detections, losses) in inference (empty losses)
        return detections, {}
    
    def _capture_training_features(self, features: torch.Tensor, proposals: List[torch.Tensor]):
        """
        Capture features during training.
        
        Args:
            features: [total_rois, 256] flat features from box_head
            proposals: List of [num_rois_per_image, 4] proposal boxes
        """
        # Flatten proposals and track image IDs
        all_boxes = []
        image_ids = []
        
        for img_idx, boxes_per_img in enumerate(proposals):
            all_boxes.append(boxes_per_img)
            image_ids.extend([img_idx] * len(boxes_per_img))
        
        all_boxes = torch.cat(all_boxes, dim=0)
        image_ids = torch.tensor(image_ids, device=features.device)
        
        self.captured_features = {
            'box_features': features,  # [N, 256]
            'boxes': all_boxes,        # [N, 4]
            'image_ids': image_ids     # [N]
        }
    
    def _capture_test_features(self, features: torch.Tensor, proposals: List[torch.Tensor]):
        """Capture features during inference."""
        self._capture_training_features(features, proposals)
    
    # Expose original attributes for compatibility
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_roi_heads, name)


def wrap_roi_heads_with_feature_extraction(model):
    """
    Wrap a Faster R-CNN model's RoI heads to enable feature extraction.
    
    Args:
        model: Faster R-CNN model (e.g., from fasterrcnn_resnet50_fpn)
        
    Returns:
        Modified model with RoIHeadWithFeatures wrapper
        
    Usage:
        model = fasterrcnn_resnet50_fpn(num_classes=6)
        model = wrap_roi_heads_with_feature_extraction(model)
        
        # During training:
        model.roi_heads.enable_feature_capture()
        loss_dict = model(images, targets)
        features = model.roi_heads.get_captured_features()
        # features['box_features'] is [N, 256] for GPCL
    """
    if not hasattr(model, 'roi_heads'):
        raise ValueError("Model must have 'roi_heads' attribute (Faster R-CNN)")
    
    # Wrap the RoI heads
    model.roi_heads = RoIHeadWithFeatures(model.roi_heads)
    
    return model
