"""
Geometry-Preserving Contrastive Learning (GPCL)
Novel contribution for domain generalization in object detection

This module implements contrastive learning that preserves geometric relationships
between objects across style variations using REAL RoI features.

Author: Your Name
Novel Contribution for ICPR 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


class GeometryPreservingContrastiveLoss(nn.Module):
    """
    Novel Contribution: Geometry-Preserving Contrastive Learning with RoI Features
    
    Unlike standard contrastive learning:
    1. Works with actual 256-D RoI features from detection model
    2. Weights contrastive loss by geometric (IoU) similarity
    3. Batch-level learning across multiple images
    4. Preserves spatial relationships across style variations
    
    Key Insight: Objects with high geometric overlap should have similar 
    features regardless of style variations. Uses real RoI features, not boxes.
    """
    
    def __init__(self, temperature=0.07, feature_dim=1024, projection_dim=128, use_geometric_weighting=True, top_k=100):
        """
        Args:
            temperature: Contrastive loss temperature (lower = harder negatives)
            feature_dim: Dimensionality of RoI features (1024 for Faster R-CNN TwoMLPHead)
            projection_dim: Dimensionality of projection space
            use_geometric_weighting: Whether to weight loss by IoU similarity
            top_k: Maximum number of detections to use per image (for memory)
        """
        super().__init__()
        self.temperature = temperature
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.use_geometric_weighting = use_geometric_weighting
        self.top_k = top_k
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)  # Project to specified dim
        )
        
        print(f"[GPCL] Initialized with real RoI features (dim={feature_dim})")
        print(f"[GPCL] Projection: {feature_dim} → {projection_dim}, temp={temperature}")
    
    def forward(self, roi_features_dict):
        """
        Compute geometry-preserving contrastive loss from actual RoI features
        
        Args:
            roi_features_dict: Dict from model.get_roi_features() with:
                'box_features': [N, 256] - actual RoI features from box_head
                'boxes': [N, 4] - corresponding box coordinates
                'image_ids': [N] - which image each box belongs to
        
        Returns:
            Scalar loss value
        """
        if roi_features_dict is None or 'box_features' not in roi_features_dict:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        box_features = roi_features_dict['box_features']  # [N, 256]
        boxes = roi_features_dict['boxes']  # [N, 4]
        image_ids = roi_features_dict['image_ids']  # [N]
        
        if box_features.shape[0] == 0:
            return torch.tensor(0.0, device=box_features.device)
        
        # Compute contrastive loss per image and average
        return self._compute_batch_contrastive_loss(box_features, boxes, image_ids)
    
    def _compute_batch_contrastive_loss(self, box_features, boxes, image_ids):
        """
        Compute contrastive loss across a batch using actual RoI features
        
        Args:
            box_features: [N, 256] - RoI features from box_head
            boxes: [N, 4] - box coordinates
            image_ids: [N] - which image each box belongs to
            
        Returns:
            Scalar loss
        """
        total_loss = 0.0
        num_images = 0
        
        # Group by image
        unique_ids = torch.unique(image_ids)
        
        for img_id in unique_ids:
            mask = (image_ids == img_id)
            img_features = box_features[mask]  # [K, 256]
            img_boxes = boxes[mask]  # [K, 4]
            
            if img_features.shape[0] < 2:
                # Need at least 2 boxes for contrastive learning
                continue
            
            # Limit to top_k for memory efficiency
            if img_features.shape[0] > self.top_k:
                img_features = img_features[:self.top_k]
                img_boxes = img_boxes[:self.top_k]
            
            # Compute contrastive loss for this image
            loss = self._compute_image_contrastive_loss(img_features, img_boxes)
            total_loss += loss
            num_images += 1
        
        if num_images == 0:
            return torch.tensor(0.0, device=box_features.device)
        
        return total_loss / num_images
    
    def _compute_image_contrastive_loss(self, roi_features, boxes):
        """
        Compute geometry-weighted contrastive loss for boxes in one image
        
        Args:
            roi_features: [N, 256] features from RoI head
            boxes: [N, 4] box coordinates
            
        Returns:
            Scalar loss
        """
        if roi_features.shape[0] < 2:
            return torch.tensor(0.0, device=roi_features.device)
        
        # Project features to contrastive space
        z = self.projection(roi_features)  # [N, projection_dim]
        z = F.normalize(z, dim=1)  # Normalize for cosine similarity
        
        # Compute feature similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # [N, N]
        
        # Compute geometric similarity (IoU) for weighting
        if self.use_geometric_weighting:
            iou_matrix = box_iou(boxes, boxes)  # [N, N]
            geometric_weights = iou_matrix.detach()  # Don't backprop through IoU
        else:
            geometric_weights = torch.ones_like(sim_matrix)
        
        # InfoNCE contrastive loss
        N = roi_features.shape[0]
        
        # For each anchor, pull positives (same object) and push negatives (different objects)
        # Since we don't have explicit positive pairs from style mixing here,
        # we use geometric similarity as soft positive labels
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(N, device=roi_features.device).bool()
        sim_matrix_masked = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Weighted contrastive loss: objects with high IoU should be similar
        # This encourages spatial clustering and relationship preservation
        loss = 0.0
        for i in range(N):
            # Use geometric weights to identify which boxes are "positive" (high IoU)
            weights_i = geometric_weights[i]  # [N]
            weights_i[i] = 0  # Exclude self
            
            if weights_i.sum() < 1e-6:
                continue
            
            # Positive: high IoU boxes
            positive_mask = weights_i > 0.3  # Threshold for "positive"
            if not positive_mask.any():
                continue
            
            positive_sim = sim_matrix[i, positive_mask].mean()
            
            # Negative: all boxes (standard InfoNCE denominator)
            log_sum_exp = torch.logsumexp(sim_matrix_masked[i], dim=0)
            
            loss += -positive_sim + log_sum_exp
        
        if N > 0:
            loss = loss / N
        
        return loss
    
    def _compute_pairwise_contrastive_loss(self, roi_features_orig, roi_features_mixed, boxes_orig, boxes_mixed):
        """
        Compute geometry-preserving contrastive loss for a single image pair
        
        Args:
            roi_features_orig: [N, D] features from original images
            roi_features_mixed: [N, D] features from style-mixed images
            boxes_orig: [N, 4] Bounding boxes in original images
            boxes_mixed: [N, 4] Corresponding boxes in mixed images
        
        Returns:
            Scalar loss value
        """
        if roi_features_orig.shape[0] == 0:
            return torch.tensor(0.0, device=roi_features_orig.device)
        
        # Project features to contrastive space
        z_orig = self.projection(roi_features_orig)  # [N, projection_dim]
        z_mixed = self.projection(roi_features_mixed)  # [N, projection_dim]
        
        # Normalize features (for cosine similarity)
        z_orig = F.normalize(z_orig, dim=1)
        z_mixed = F.normalize(z_mixed, dim=1)
        
        # Compute feature similarity matrix
        sim_orig_mixed = torch.matmul(z_orig, z_mixed.T) / self.temperature  # [N, N]
        sim_orig_orig = torch.matmul(z_orig, z_orig.T) / self.temperature   # [N, N]
        
        # Compute geometric similarity matrix (IoU)
        if self.use_geometric_weighting:
            iou_matrix = box_iou(boxes_orig, boxes_orig)  # [N, N]
            geometric_weights = iou_matrix.detach()
        else:
            geometric_weights = torch.ones_like(sim_orig_orig)
        
        # Contrastive loss: pull positive pairs, push negative pairs
        batch_size = z_orig.shape[0]
        
        # Positive pairs: same object across style variations
        positive_sim = torch.diag(sim_orig_mixed)  # [N]
        
        # Negative pairs: all other objects in batch
        mask = torch.eye(batch_size, device=z_orig.device).bool()
        negative_sim = sim_orig_orig.masked_fill(mask, float('-inf'))  # [N, N]
        
        # InfoNCE-style contrastive loss
        # log(exp(pos) / (exp(pos) + sum(exp(neg))))
        numerator = torch.exp(positive_sim)  # [N]
        denominator = numerator + torch.exp(negative_sim).sum(dim=1)  # [N]
        loss = -torch.log(numerator / (denominator + 1e-8))  # [N]
        
        # Novel: Weight by geometric similarity
        # Objects with high IoU are more similar → their loss matters more
        if self.use_geometric_weighting:
            geometric_importance = torch.diag(geometric_weights)  # [N]
            # Normalize weights to sum to 1
            geometric_importance = geometric_importance / (geometric_importance.sum() + 1e-8)
            loss = (loss * geometric_importance * batch_size).sum()
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_loss_with_detections(self, detections_orig, detections_mixed, model):
        """
        Convenience method to compute loss from detection outputs
        
        Args:
            detections_orig: List of dicts with 'boxes', 'scores', 'labels'
            detections_mixed: List of dicts (corresponding mixed predictions)
            model: Detection model (to extract RoI features)
        
        Returns:
            Contrastive loss
        """
        total_loss = 0.0
        num_images = 0
        
        for det_orig, det_mixed in zip(detections_orig, detections_mixed):
            boxes_orig = det_orig['boxes']
            boxes_mixed = det_mixed['boxes']
            
            if len(boxes_orig) == 0 or len(boxes_mixed) == 0:
                continue
            
            # Match boxes between orig and mixed (use Hungarian matching or top-K)
            K = min(len(boxes_orig), len(boxes_mixed))
            boxes_orig = boxes_orig[:K]  # Top-K by score
            boxes_mixed = boxes_mixed[:K]
            
            # Extract RoI features (requires model access)
            # This is a simplified version - adapt to your model architecture
            roi_features_orig = self._extract_roi_features(boxes_orig, model)
            roi_features_mixed = self._extract_roi_features(boxes_mixed, model)
            
            # Compute contrastive loss
            loss = self.forward(roi_features_orig, roi_features_mixed, 
                               boxes_orig, boxes_mixed)
            total_loss += loss
            num_images += 1
        
        return total_loss / max(num_images, 1)
    
    def _extract_roi_features(self, boxes, model):
        """
        Extract RoI features from detection model
        
        Note: This is model-specific. Adapt based on your architecture.
        For Faster R-CNN, use RoI pooling on backbone features.
        """
        # Placeholder - implement based on your model
        # For Faster R-CNN with ResNet-50-FPN:
        # 1. Get backbone features
        # 2. Apply RoI Align
        # 3. Apply box head
        # 4. Return features before final FC layers
        raise NotImplementedError("Implement based on your model architecture")


class GPCLIntegration(nn.Module):
    """
    Integration module for GPCL in detection training
    
    Usage:
        gpcl = GPCLIntegration(model, temperature=0.07)
        loss = gpcl.compute_loss(images, images_mixed, targets)
    """
    
    def __init__(self, model, temperature=0.07, loss_weight=0.5):
        super().__init__()
        self.model = model
        self.gpcl_loss = GeometryPreservingContrastiveLoss(
            temperature=temperature,
            feature_dim=256  # Adjust based on your model
        )
        self.loss_weight = loss_weight
    
    def compute_loss(self, images_orig, images_mixed, targets):
        """
        Compute GPCL loss during training
        
        Args:
            images_orig: List of original images
            images_mixed: List of style-mixed images
            targets: Ground truth annotations
        
        Returns:
            GPCL loss value
        """
        # Get detection predictions
        with torch.no_grad():
            self.model.eval()
            detections_orig = self.model(images_orig)
            detections_mixed = self.model(images_mixed)
            self.model.train()
        
        # Compute contrastive loss
        loss = self.gpcl_loss.compute_loss_with_detections(
            detections_orig, detections_mixed, self.model
        )
        
        return self.loss_weight * loss


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """
    Example of how to integrate GPCL into your training loop
    """
    # Initialize GPCL module
    gpcl_module = GeometryPreservingContrastiveLoss(
        temperature=0.07,
        feature_dim=256,
        use_geometric_weighting=True
    )
    
    # During training loop:
    # 1. Get original and mixed features/boxes
    roi_features_orig = torch.randn(10, 256)  # Example: 10 RoIs
    roi_features_mixed = torch.randn(10, 256)
    boxes_orig = torch.rand(10, 4) * 100  # Random boxes
    boxes_mixed = boxes_orig + torch.randn(10, 4) * 5  # Slightly perturbed
    
    # 2. Compute GPCL loss
    gpcl_loss = gpcl_module(roi_features_orig, roi_features_mixed, 
                            boxes_orig, boxes_mixed)
    
    # 3. Add to total loss
    total_loss = detection_loss + 0.5 * gpcl_loss
    
    print(f"GPCL Loss: {gpcl_loss.item():.4f}")
    
    return total_loss


if __name__ == "__main__":
    print("Geometry-Preserving Contrastive Learning (GPCL) Module")
    print("=" * 60)
    print("Novel contribution for domain generalization in object detection")
    print()
    
    # Run example
    loss = example_usage()
    print(f"Total Loss: {loss.item():.4f}")
