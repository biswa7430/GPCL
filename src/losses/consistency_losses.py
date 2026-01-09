"""Detection-specific consistency losses for SHM - FULLY FIXED VERSION.

Includes:
- classification_consistency: KL divergence between class probability distributions
- box_consistency: IoU-based distance for box predictions
- combined loss wrapper using minimum size approach (robust and simple)
"""
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F


def classification_consistency(p_orig: torch.Tensor, p_hall: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute symmetric KL divergence between two probability tensors.

    Args:
        p_orig: Tensor [N, num_classes] probabilities (softmax outputs)
        p_hall: Tensor [N, num_classes]

    Returns:
        scalar tensor loss
    """
    # clamp to avoid log(0)
    p_orig = p_orig.clamp(min=eps)
    p_hall = p_hall.clamp(min=eps)

    m = 0.5 * (p_orig + p_hall)
    loss = 0.5 * (F.kl_div(m.log(), p_orig, reduction='batchmean') + 
                  F.kl_div(m.log(), p_hall, reduction='batchmean'))
    return loss


def box_consistency(boxes_a: torch.Tensor, boxes_b: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """IoU-based consistency loss between two sets of boxes.

    CRITICAL FIX: Assumes boxes_a and boxes_b are ALREADY SAME SIZE (pre-truncated by caller).
    
    Args:
        boxes_a: Tensor [K, 4] in (x1, y1, x2, y2) format
        boxes_b: Tensor [K, 4] in (x1, y1, x2, y2) format
        
    Returns:
        mean (1 - IoU) for the boxes.
    """
    # Validate inputs
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        device = boxes_a.device if boxes_a.numel() > 0 else boxes_b.device
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    # CRITICAL: Verify same size
    if boxes_a.shape[0] != boxes_b.shape[0]:
        raise ValueError(
            f"boxes_a ({boxes_a.shape[0]}) and boxes_b ({boxes_b.shape[0]}) must have same size! "
            f"Caller must truncate to K=min(N_a, N_b) before calling this function."
        )

    K = boxes_a.shape[0]
    
    if verbose:
        print(f"[box_consistency] Comparing {K} boxes")

    # Compute pairwise IoU between all pairs
    # boxes_a: [K, 4] -> [K, 1, 4]
    # boxes_b: [K, 4] -> [1, K, 4]
    a = boxes_a.unsqueeze(1)  # [K, 1, 4]
    b = boxes_b.unsqueeze(0)  # [1, K, 4]

    # Broadcast to [K, K, 4]
    # Compute intersection
    inter_x1 = torch.max(a[..., 0], b[..., 0])
    inter_y1 = torch.max(a[..., 1], b[..., 1])
    inter_x2 = torch.min(a[..., 2], b[..., 2])
    inter_y2 = torch.min(a[..., 3], b[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h  # [K, K]

    # Compute areas
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)

    # Compute IoU
    union = area_a + area_b - inter + 1e-6
    iou = inter / union  # [K, K]

    # For each box in a, find best match in b
    best_iou_vals, _ = iou.max(dim=1)  # [K]

    if verbose:
        print(f"[box_consistency] Best IoU values (first 5): {best_iou_vals[:5]}")
        print(f"[box_consistency] Mean IoU: {best_iou_vals.mean().item():.4f}")

    # Loss = mean(1 - iou)
    loss = (1.0 - best_iou_vals).mean()
    
    if verbose:
        print(f"[box_consistency] Loss: {loss.item():.4f}")
    
    return loss


def combined_consistency(
    preds_orig: List[Dict], 
    preds_hall: List[Dict], 
    box_weight: float = 1.0, 
    cls_weight: float = 1.0, 
    verbose: bool = False
) -> torch.Tensor:
    """Compute combined consistency loss given detector outputs.

    preds_* are expected in torchvision format: list[dict] per image with keys:
      'boxes': Tensor [N,4], 'scores': [N], 'labels': [N]

    Strategy (MINIMUM SIZE APPROACH - PROPERLY IMPLEMENTED):
      - For each image pair, compute K = min(N_orig, N_hall)
      - Truncate BOTH predictions to K elements BEFORE passing to sub-functions
      - This guarantees no size mismatches
      
    Returns:
      scalar loss on same device as predictions
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if len(preds_orig) > 0 and 'boxes' in preds_orig[0] and preds_orig[0]['boxes'].numel() > 0:
        device = preds_orig[0]['boxes'].device
    elif len(preds_hall) > 0 and 'boxes' in preds_hall[0] and preds_hall[0]['boxes'].numel() > 0:
        device = preds_hall[0]['boxes'].device
    
    total_box_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    total_cls_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    count_box = 0
    count_cls = 0

    if verbose:
        print(f"\n[combined_consistency] Computing consistency for {len(preds_orig)} images")

    for idx, (o, h) in enumerate(zip(preds_orig, preds_hall)):
        # Get number of boxes
        num_boxes_o = len(o['boxes']) if 'boxes' in o else 0
        num_boxes_h = len(h['boxes']) if 'boxes' in h else 0
        
        if verbose and idx == 0:
            print(f"[combined_consistency] Image 0: orig={num_boxes_o} boxes, hall={num_boxes_h} boxes")
        
        # Skip if either has no boxes
        if num_boxes_o == 0 or num_boxes_h == 0:
            continue
        
        # CRITICAL: Compute K and truncate IMMEDIATELY
        K = min(num_boxes_o, num_boxes_h)
        
        if K == 0:
            continue
        
        # Truncate ALL tensors to exactly K elements
        boxes_o = o['boxes'][:K].clone()  # Clone to avoid issues
        boxes_h = h['boxes'][:K].clone()
        scores_o = o['scores'][:K].clone() if 'scores' in o else None
        scores_h = h['scores'][:K].clone() if 'scores' in h else None
        
        # VERIFY truncation (helps debugging)
        assert boxes_o.shape[0] == K, f"boxes_o truncation failed: {boxes_o.shape[0]} != {K}"
        assert boxes_h.shape[0] == K, f"boxes_h truncation failed: {boxes_h.shape[0]} != {K}"
        
        if verbose and idx == 0:
            print(f"[combined_consistency] Using K={K} predictions")
            print(f"  boxes_o shape: {boxes_o.shape}")
            print(f"  boxes_h shape: {boxes_h.shape}")
        
        # Compute box consistency (both tensors guaranteed to be [K, 4])
        try:
            b_loss = box_consistency(boxes_o, boxes_h, verbose=(verbose and idx == 0))
            total_box_loss = total_box_loss + b_loss
            count_box += 1
        except Exception as e:
            print(f"WARNING: box_consistency failed for image {idx}: {e}")
            continue

        # Classification consistency - simple L1 on scores
        if scores_o is not None and scores_h is not None:
            assert scores_o.shape[0] == K
            assert scores_h.shape[0] == K
            c_loss = (scores_o - scores_h).abs().mean()
            total_cls_loss = total_cls_loss + c_loss
            count_cls += 1

    if count_box == 0 and count_cls == 0:
        if verbose:
            print(f"[combined_consistency] No valid pairs - returning 0")
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    # Compute averages
    box_loss_avg = total_box_loss / max(1, count_box)
    cls_loss_avg = total_cls_loss / max(1, count_cls)
    
    loss = box_weight * box_loss_avg + cls_weight * cls_loss_avg
    
    if verbose:
        print(f"[combined_consistency] Final: box={box_loss_avg.item():.4f}, "
              f"cls={cls_loss_avg.item():.4f}, total={loss.item():.4f}")
    
    return loss