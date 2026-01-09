"""SHM-enabled detection wrapper with RoI feature extraction for GPCL."""
from typing import Optional, Tuple, List, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.augmentation.style_hallucination import ImageStyleMixer, StyleHallucination
from src.models.roi_feature_extractor import wrap_roi_heads_with_feature_extraction


class SHMDetector(nn.Module):
    """Wrapper around a torchvision detection model that applies SHM and extracts RoI features."""

    def __init__(self,
                 base_model: nn.Module,
                 image_mixer: Optional[ImageStyleMixer] = None,
                 feature_shm: Optional[StyleHallucination] = None,
                 enable_image_level: bool = True,
                 enable_feature_level: bool = False,
                 config: Optional[dict] = None,
                 enable_roi_features: bool = True):
        super().__init__()
        
        # Wrap base model's RoI heads for feature extraction
        if enable_roi_features and hasattr(base_model, 'roi_heads'):
            base_model = wrap_roi_heads_with_feature_extraction(base_model)
            print("[SHMDetector] RoI feature extraction enabled")
        
        self.base_model = base_model
        self.image_mixer = image_mixer
        self.feature_shm = feature_shm
        self.enable_image_level = enable_image_level
        self.enable_feature_level = enable_feature_level
        self.config = config or {}
        self.enable_roi_features = enable_roi_features
        
        # Create convenient aliases
        self.image_level = enable_image_level
        self.feature_level = enable_feature_level
        
        # Storage for captured features
        self.last_roi_features = None

    def forward(self, images, targets=None):
        """Forward pass with style mixing and RoI feature extraction."""
        if not self.training:
            # Inference mode - no style mixing
            return self.base_model(images, targets)
        
        # Training mode with style mixing
        use_consistency = self.config.get('shm', {}).get('consistency_loss_weight', 0) > 0
        
        # Image-level mixing
        if self.image_level and self.image_mixer is not None:
            orig_imgs, mixed_imgs = self.image_mixer.mix_batch(images)
        else:
            orig_imgs = images
            mixed_imgs = images
        
        # Enable RoI feature capture if needed
        if self.enable_roi_features and hasattr(self.base_model, 'roi_heads'):
            self.base_model.roi_heads.enable_feature_capture()
        
        # Get predictions from original images (with gradients)
        loss_dict_orig = self.base_model(orig_imgs, targets)
        
        # Capture RoI features from forward pass
        if self.enable_roi_features and hasattr(self.base_model, 'roi_heads'):
            self.last_roi_features = self.base_model.roi_heads.get_captured_features()
            self.base_model.roi_heads.disable_feature_capture()
        
        if use_consistency and self.image_level and self.image_mixer is not None:
            # Get predictions from mixed images
            with torch.no_grad():
                self.base_model.eval()
                mixed_outputs = self.base_model(mixed_imgs)
                self.base_model.train()
            
            # Get predictions from original in eval mode too
            with torch.no_grad():
                self.base_model.eval()
                orig_outputs = self.base_model(orig_imgs)
                self.base_model.train()
            
            # Compute consistency loss
            consistency_loss = self._compute_prediction_consistency(
                orig_outputs, mixed_outputs
            )
            
            # Add to loss dict
            weight = self.config.get('shm', {}).get('consistency_loss_weight', 0.5)
            loss_dict_orig['consistency_loss'] = consistency_loss * weight
        else:
            loss_dict_orig['consistency_loss'] = torch.tensor(0.0, device=images[0].device)
        
        return loss_dict_orig
    
    def get_roi_features(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get RoI features from last forward pass.
        
        Returns:
            dict with:
                'box_features': [N, 256] RoI features
                'boxes': [N, 4] box coordinates
                'image_ids': [N] image indices
        """
        return self.last_roi_features
    
    def _compute_prediction_consistency(self, orig_outputs, mixed_outputs):
        """
        Compute consistency using MINIMUM SIZE approach with proper scaling.
        
        FIXES:
        1. Use minimum-size truncation to avoid indexing issues
        2. Normalize box coordinates to [0, 1] range for proper loss scaling
        3. Use IoU-based loss instead of L1 for better geometric meaning
        """
        consistency_loss = torch.tensor(0.0, device=orig_outputs[0]['boxes'].device)
        
        box_weight = self.config.get('shm', {}).get('consistency_box_weight', 1.0)
        score_weight = self.config.get('shm', {}).get('consistency_cls_weight', 0.1)
        
        num_pairs = 0
        
        # Process each image in batch
        for orig_out, mixed_out in zip(orig_outputs, mixed_outputs):
            try:
                # Get number of detections
                num_orig = len(orig_out['boxes'])
                num_mixed = len(mixed_out['boxes'])
                
                # Skip if either has no detections
                if num_orig == 0 or num_mixed == 0:
                    continue
                
                # Use minimum size approach - take top K detections
                K = min(num_orig, num_mixed)
                
                # Both are sorted by confidence, so take top K
                orig_boxes_k = orig_out['boxes'][:K]
                mixed_boxes_k = mixed_out['boxes'][:K]
                
                # CRITICAL FIX: Use IoU-based loss instead of L1
                # This is scale-invariant and geometrically meaningful
                from torchvision.ops import box_iou
                
                # Compute pairwise IoU
                iou = box_iou(orig_boxes_k, mixed_boxes_k)
                
                # Take diagonal (matching pairs)
                iou_diag = torch.diag(iou)
                
                # IoU loss: maximize IoU = minimize (1 - IoU)
                box_loss = (1.0 - iou_diag).mean()
                
                consistency_loss += box_weight * box_loss
                
                # Score consistency (if available and 1D)
                if 'scores' in orig_out and 'scores' in mixed_out:
                    orig_scores_k = orig_out['scores'][:K]
                    mixed_scores_k = mixed_out['scores'][:K]
                    
                    # Only compare if both are 1D tensors
                    if orig_scores_k.dim() == 1 and mixed_scores_k.dim() == 1:
                        score_loss = F.mse_loss(orig_scores_k, mixed_scores_k)
                        consistency_loss += score_weight * score_loss
                
                num_pairs += 1
                
            except Exception as e:
                # Debug: print error but continue
                print(f"WARNING: Error computing consistency loss: {e}")
                continue
        
        # Average over batch
        if num_pairs > 0:
            consistency_loss = consistency_loss / num_pairs
        
        return consistency_loss


def wrap_model_with_shm(base_model: nn.Module, config: dict) -> Union[SHMDetector, nn.Module]:
    """Factory to wrap a base model using config dictionary."""
    shm_cfg = config.get('shm', {})
    enable = shm_cfg.get('enable', False)
    if not enable:
        return base_model

    image_mixer = None
    feature_shm = None
    enable_image = shm_cfg.get('image_level', True)
    enable_feat = shm_cfg.get('feature_level', False)

    if enable_image:
        image_mixer = ImageStyleMixer(
            mix_prob=shm_cfg.get('image_mix_prob', 0.5), 
            alpha=shm_cfg.get('image_alpha', 0.2)
        )

    if enable_feat:
        style_dim = shm_cfg.get('style_dim', 256)
        base_num = shm_cfg.get('base_style_num', 32)
        concentration = shm_cfg.get('concentration', 0.02)
        feature_shm = StyleHallucination(
            channels=style_dim, 
            num_prototypes=base_num, 
            concentration=concentration
        )

    return SHMDetector(
        base_model, 
        image_mixer=image_mixer, 
        feature_shm=feature_shm,
        enable_image_level=enable_image, 
        enable_feature_level=enable_feat,
        config=config
    )