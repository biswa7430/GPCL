"""Style Hallucination module adapted for detection.

Contains:
- ImageStyleMixer: image-level style mixing (PIL/torch tensor aware)
- StyleHallucination: feature-level mixing using prototype statistics

Designed to be lightweight and compatible with torchvision detection pipelines.
"""
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import random


class ImageStyleMixer:
    """Image-level style mixing."""

    def __init__(self, mix_prob: float = 0.5, alpha: float = 0.2, device: Optional[torch.device] = None):
        """
        Args:
            mix_prob: probability of applying mixing to an image
            alpha: mixing coefficient (0 = original, 1 = full style transfer)
            device: torch device
        """
        self.mix_prob = mix_prob
        # Ensure alpha is meaningful - increase minimum to 0.3 for stronger mixing
        self.alpha = max(0.3, min(0.6, alpha))
        self.device = device

    def mix_batch(self, images: List[torch.Tensor], force_mix: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Mix style within a batch of images.

        Args:
            images: List of tensors [C, H, W], can be variable sizes
            force_mix: If True, mix ALL images (ignore mix_prob). 
                      Use this for consistency loss computation.

        Returns:
            (original_images, mixed_images)
        """
        if len(images) < 2:
            # Cannot mix with less than 2 images
            return images, images

        mixed_images = []
        
        for i, img in enumerate(images):
            # Decide whether to mix this image
            should_mix = force_mix or (random.random() <= self.mix_prob)
            
            if not should_mix:
                # Don't mix - keep original
                mixed_images.append(img.clone())
                continue
            
            # Select a random partner image (different from current)
            partner_idx = random.choice([j for j in range(len(images)) if j != i])
            partner = images[partner_idx]
            
            # Apply style mixing: transfer mean/std from partner to current image
            mixed = self._style_mix_single(img, partner)
            mixed_images.append(mixed)
        
        return images, mixed_images

    def _style_mix_single(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Mix style from style image into content image using AdaIN.

        Args:
            content: [C, H, W] - image to modify
            style: [C, H, W] - image to take style from

        Returns:
            [C, H, W] - style-mixed image
        """
        C = content.shape[0]
        
        # Flatten spatial dimensions
        content_flat = content.view(C, -1)  # [C, H*W]
        style_flat = style.view(C, -1)      # [C, H*W]
        
        # Compute statistics per channel (using more robust statistics)
        content_mean = content_flat.mean(dim=1, keepdim=True)  # [C, 1]
        content_std = content_flat.std(dim=1, keepdim=True) + 1e-5  # [C, 1]
        
        style_mean = style_flat.mean(dim=1, keepdim=True)  # [C, 1]
        style_std = style_flat.std(dim=1, keepdim=True) + 1e-5  # [C, 1]
        
        # AdaIN (Adaptive Instance Normalization)
        # Normalize content to zero mean and unit variance
        normalized = (content_flat - content_mean) / content_std
        
        # Scale and shift with style statistics
        stylized_flat = normalized * style_std + style_mean
        
        # ENHANCED: Use stronger alpha blending for better domain shift
        # alpha controls how much style to transfer (higher = more style transfer)
        mixed_flat = (1 - self.alpha) * content_flat + self.alpha * stylized_flat
        
        # Clamp to valid range [0, 1] to avoid extreme values
        mixed_flat = torch.clamp(mixed_flat, 0.0, 1.0)
        
        # Reshape back to original shape
        mixed = mixed_flat.view_as(content)
        
        return mixed


class StyleHallucination(nn.Module):
    """Feature-level style hallucination using learned style prototypes.

    This creates synthetic feature variations by mixing channel statistics
    (mean/std) with learned style prototypes.
    """

    def __init__(self, channels: int, num_prototypes: int = 32, concentration: float = 0.02):
        """
        Args:
            channels: number of feature channels
            num_prototypes: number of style prototypes to learn
            concentration: dirichlet concentration parameter (lower = more diverse mixing)
        """
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes
        self.concentration = concentration

        # Learnable style prototypes: mean and std vectors
        self.style_mean = nn.Parameter(torch.randn(num_prototypes, channels))
        self.style_std = nn.Parameter(torch.ones(num_prototypes, channels))

    def forward(self, features: torch.Tensor, return_both: bool = False) -> torch.Tensor:
        """Apply style hallucination to features.

        Args:
            features: [B, C, H, W] feature maps
            return_both: if True, return [orig, hallucinated] concatenated on batch dim

        Returns:
            [B, C, H, W] or [2*B, C, H, W] hallucinated features
        """
        B, C, H, W = features.shape

        if return_both:
            # Return both original and hallucinated
            hallucinated = self._hallucinate(features)
            return torch.cat([features, hallucinated], dim=0)
        else:
            # Return only hallucinated
            return self._hallucinate(features)

    def _hallucinate(self, features: torch.Tensor) -> torch.Tensor:
        """Internal hallucination logic.

        Args:
            features: [B, C, H, W]

        Returns:
            [B, C, H, W] hallucinated features
        """
        B, C, H, W = features.shape

        # Compute feature statistics
        feat_flat = features.view(B, C, -1)  # [B, C, H*W]
        feat_mean = feat_flat.mean(dim=2, keepdim=True)  # [B, C, 1]
        feat_std = feat_flat.std(dim=2, keepdim=True) + 1e-6  # [B, C, 1]

        # Sample mixing weights from Dirichlet distribution
        # For each batch element, sample weights over prototypes
        alpha = torch.ones(self.num_prototypes, device=features.device) * self.concentration
        # Use simple random sampling (Dirichlet approximation)
        weights = torch.softmax(torch.randn(B, self.num_prototypes, device=features.device), dim=1)  # [B, K]

        # Mix style prototypes
        mixed_mean = torch.matmul(weights, self.style_mean)  # [B, C]
        mixed_std = torch.matmul(weights, self.style_std.abs())  # [B, C]

        # Reshape for broadcasting
        mixed_mean = mixed_mean.unsqueeze(2)  # [B, C, 1]
        mixed_std = mixed_std.unsqueeze(2)    # [B, C, 1]

        # Normalize and apply hallucinated style
        normalized = (feat_flat - feat_mean) / feat_std
        hallucinated_flat = normalized * mixed_std + mixed_mean

        # Reshape back
        hallucinated = hallucinated_flat.view(B, C, H, W)

        return hallucinated


# Backwards-compatible factory
def build_style_hallucination_for_channel(ch: int, base_style_num: int = 32, concentration: float = 0.02, device: Optional[torch.device] = None) -> StyleHallucination:
    """Factory function for StyleHallucination module.

    Args:
        ch: number of channels
        base_style_num: number of style prototypes
        concentration: dirichlet concentration
        device: torch device

    Returns:
        StyleHallucination module
    """
    module = StyleHallucination(ch, base_style_num, concentration)
    if device is not None:
        module = module.to(device)
    return module