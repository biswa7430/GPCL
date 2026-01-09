"""SHM-aware training utilities with detailed diagnostics.

This fi        print("="*60)
        print("SHM DIAGNOSTIC INFO")
        print("="*60)
        print(f"SHM Enabled: {use_shm}")
        print(f"Use Consistency Loss: {use_consistency}")
        print(f"Consistency Weight: {consistency_weight}")
        print(f"Use GPCL Loss: {use_gpcl}")
        print(f"GPCL Weight: {gpcl_weight}")
        print(f"Model Type: {type(model).__name__}")
        print(f"Has base_model: {hasattr(model, 'base_model')}")
        print(f"Has image_mixer: {hasattr(model, 'image_mixer')}")
        print(f"Has feature_shm: {hasattr(model, 'feature_shm')}")
        if hasattr(model, 'enable_image_level'):
            print(f"Image Level Enabled: {model.enable_image_level}")
        if hasattr(model, 'enable_feature_level'):
            print(f"Feature Level Enabled: {model.enable_feature_level}")
        print("="*60 + "\n")raining wrapper that integrates SHM components and
consistency losses while preserving existing training behavior.
"""
from typing import Optional, Dict, List
import torch
from torch.cuda.amp import GradScaler
from src.losses.consistency_losses import combined_consistency
from src.losses.gpcl_loss import GeometryPreservingContrastiveLoss


def train_one_epoch_shm(
    model, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    config, 
    scaler: Optional[GradScaler] = None, 
    print_freq: int = 20
):
    """Train one epoch with optional SHM consistency losses.

    This version properly handles detection models by:
    1. Computing detection losses normally in training mode
    2. Optionally computing consistency losses in eval mode (if enabled)
    3. Handling NaN losses with gradient clipping and loss validation
    4. DIAGNOSTIC: Adds detailed logging to debug why consistency loss is 0

    Args:
        model: model (already wrapped by SHMDetector or raw)
        optimizer: optimizer
        data_loader: dataloader
        device: torch.device
        epoch: int
        config: config dict holding shm settings
        scaler: optional GradScaler for AMP
        print_freq: print frequency
        
    Returns:
        metric logger dict
    """
    model.train()

    # Extract SHM config
    shm_config = config.get('shm', {})
    use_shm = shm_config.get('enable', False)
    use_consistency = use_shm and shm_config.get('consistency_loss_weight', 0.0) > 0.0
    consistency_weight = shm_config.get('consistency_loss_weight', 0.0)
    cls_weight = shm_config.get('consistency_cls_weight', 1.0)
    box_weight = shm_config.get('consistency_box_weight', 1.0)
    
    # GPCL config
    use_gpcl = use_shm and shm_config.get('gpcl_loss_weight', 0.0) > 0.0
    gpcl_weight = shm_config.get('gpcl_loss_weight', 0.0)
    gpcl_criterion = None
    if use_gpcl:
        gpcl_criterion = GeometryPreservingContrastiveLoss(
            feature_dim=shm_config.get('gpcl_feature_dim', 256),
            projection_dim=shm_config.get('gpcl_projection_dim', 128),
            temperature=shm_config.get('gpcl_temperature', 0.07),
            top_k=shm_config.get('gpcl_top_k', 100)
        ).to(device)

    # DIAGNOSTIC: Print SHM config
    if epoch == 0:
        print("\n" + "="*60)
        print("SHM DIAGNOSTIC INFO")
        print("="*60)
        print(f"SHM Enabled: {use_shm}")
        print(f"Use Consistency Loss: {use_consistency}")
        print(f"Consistency Weight: {consistency_weight}")
        print(f"Model Type: {type(model).__name__}")
        print(f"Has base_model: {hasattr(model, 'base_model')}")
        print(f"Has image_mixer: {hasattr(model, 'image_mixer')}")
        print(f"Has feature_shm: {hasattr(model, 'feature_shm')}")
        if hasattr(model, 'enable_image_level'):
            print(f"Image Level Enabled: {model.enable_image_level}")
        if hasattr(model, 'enable_feature_level'):
            print(f"Feature Level Enabled: {model.enable_feature_level}")
        print("="*60 + "\n")

    total_det_loss = 0.0
    total_consistency_loss = 0.0
    total_gpcl_loss = 0.0
    num_batches = 0
    num_nan_batches = 0
    num_consistency_computed = 0
    num_consistency_skipped = 0
    num_gpcl_computed = 0

    for i, (images, targets) in enumerate(data_loader):
        # Skip empty batches (can occur if all samples were filtered out)
        if len(images) == 0 or len(targets) == 0:
            continue
            
        # Move to device
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in t.items()} for t in targets]

        # Final safety check: validate boxes before forward pass
        valid_batch = True
        for target_idx, target in enumerate(targets):
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                box_widths = boxes[:, 2] - boxes[:, 0]
                box_heights = boxes[:, 3] - boxes[:, 1]
                
                # Check for invalid boxes
                invalid_mask = (box_widths <= 0) | (box_heights <= 0)
                if invalid_mask.any():
                    invalid_boxes = boxes[invalid_mask]
                    print(f"WARNING: Skipping batch due to invalid boxes at target {target_idx}:")
                    print(f"  Invalid boxes: {invalid_boxes.tolist()}")
                    valid_batch = False
                    break
        
        if not valid_batch:
            continue

        # DIAGNOSTIC: Check image statistics before SHM (first iteration only)
        if i == 0 and epoch == 0:
            img_means = [img.mean().item() for img in images[:2]]
            img_stds = [img.std().item() for img in images[:2]]
            print(f"\n[DIAGNOSTIC] Original image stats (first 2):")
            print(f"  Means: {img_means}")
            print(f"  Stds: {img_stds}")

        optimizer.zero_grad()

        # Forward pass with AMP
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            # Standard detection forward pass (returns loss dict in training mode)
            loss_dict = model(images, targets)
            
            # Extract and validate detection losses
            if isinstance(loss_dict, dict):
                det_loss = sum(loss for loss in loss_dict.values())
            else:
                det_loss = loss_dict

            # Check for NaN in detection loss
            if torch.isnan(det_loss) or torch.isinf(det_loss):
                print(f"WARNING: NaN/Inf detection loss at iteration {i}, skipping batch")
                num_nan_batches += 1
                continue

            # Initialize consistency and GPCL losses
            consistency_loss = torch.tensor(0.0, device=device)
            gpcl_loss = torch.tensor(0.0, device=device)

            # Compute consistency loss if SHM is enabled
            if use_consistency:
                try:
                    # DIAGNOSTIC: Log entry into consistency computation
                    if i == 0 and epoch == 0:
                        print(f"\n[DIAGNOSTIC] Attempting consistency loss computation...")
                    
                    # Temporarily switch to eval mode for consistency predictions
                    was_training = model.training
                    model.eval()
                    
                    with torch.no_grad():
                        # Get original predictions (no targets = inference mode)
                        if hasattr(model, 'base_model'):
                            preds_orig = model.base_model(images)
                        else:
                            preds_orig = model(images)
                        
                        # DIAGNOSTIC: Check prediction structure
                        if i == 0 and epoch == 0:
                            print(f"[DIAGNOSTIC] Original predictions:")
                            print(f"  Type: {type(preds_orig)}")
                            print(f"  Length: {len(preds_orig) if isinstance(preds_orig, list) else 'N/A'}")
                            if isinstance(preds_orig, list) and len(preds_orig) > 0:
                                print(f"  First pred keys: {preds_orig[0].keys()}")
                                print(f"  Num boxes in first: {len(preds_orig[0]['boxes'])}")
                                if len(preds_orig[0]['boxes']) > 0:
                                    print(f"  First box: {preds_orig[0]['boxes'][0]}")
                                    print(f"  First score: {preds_orig[0]['scores'][0].item()}")
                        
                        # Get hallucinated predictions
                        images_mixed = images  # Default to same images
                        
                        # Try to apply image-level mixing
                        if hasattr(model, 'image_mixer') and model.image_mixer is not None:
                            if i == 0 and epoch == 0:
                                print(f"[DIAGNOSTIC] Applying image-level mixing (force_mix=True)...")
                            # CRITICAL: Use force_mix=True to ensure mixing happens for consistency loss
                            _, images_mixed = model.image_mixer.mix_batch(images, force_mix=True)
                            
                            # DIAGNOSTIC: Check if mixing actually changed images
                            if i == 0 and epoch == 0:
                                orig_mean = images[0].mean().item()
                                mixed_mean = images_mixed[0].mean().item()
                                orig_std = images[0].std().item()
                                mixed_std = images_mixed[0].std().item()
                                print(f"[DIAGNOSTIC] Image mixing check:")
                                print(f"  Original: mean={orig_mean:.4f}, std={orig_std:.4f}")
                                print(f"  Mixed:    mean={mixed_mean:.4f}, std={mixed_std:.4f}")
                                print(f"  Mean difference: {abs(orig_mean - mixed_mean):.4f}")
                                print(f"  Std difference: {abs(orig_std - mixed_std):.4f}")
                        
                        # Get predictions on mixed images
                        if hasattr(model, 'base_model'):
                            preds_hall = model.base_model(images_mixed)
                        else:
                            preds_hall = model(images_mixed)
                        
                        # DIAGNOSTIC: Check hallucinated predictions
                        if i == 0 and epoch == 0:
                            print(f"[DIAGNOSTIC] Hallucinated predictions:")
                            print(f"  Type: {type(preds_hall)}")
                            print(f"  Length: {len(preds_hall) if isinstance(preds_hall, list) else 'N/A'}")
                            if isinstance(preds_hall, list) and len(preds_hall) > 0:
                                print(f"  Num boxes in first: {len(preds_hall[0]['boxes'])}")
                    
                    # Restore training mode
                    if was_training:
                        model.train()

                    # Compute consistency loss
                    # Check if predictions are actually different
                    are_different = False
                    if isinstance(preds_orig, list) and isinstance(preds_hall, list):
                        if len(preds_orig) > 0 and len(preds_hall) > 0:
                            # Check if boxes or scores differ
                            num_orig_boxes = len(preds_orig[0]['boxes'])
                            num_hall_boxes = len(preds_hall[0]['boxes'])
                            
                            if num_orig_boxes > 0 and num_hall_boxes > 0:
                                # FIXED: Truncate to minimum size before comparison
                                K_diag = min(num_orig_boxes, num_hall_boxes)
                                orig_boxes_trunc = preds_orig[0]['boxes'][:K_diag]
                                hall_boxes_trunc = preds_hall[0]['boxes'][:K_diag]
                                
                                box_diff = (orig_boxes_trunc - hall_boxes_trunc).abs().sum().item()
                                are_different = box_diff > 1e-6
                                
                                if i == 0 and epoch == 0:
                                    print(f"[DIAGNOSTIC] Comparing {K_diag} boxes (orig={num_orig_boxes}, hall={num_hall_boxes})")
                                    print(f"[DIAGNOSTIC] Box difference: {box_diff:.6f}")
                                    print(f"[DIAGNOSTIC] Predictions are different: {are_different}")
                    
                    if are_different or (isinstance(preds_orig, list) and isinstance(preds_hall, list)):
                        consistency_loss = combined_consistency(
                            preds_orig, 
                            preds_hall, 
                            box_weight=box_weight, 
                            cls_weight=cls_weight
                        )
                        
                        # DIAGNOSTIC: Log consistency loss value
                        if i == 0 and epoch == 0:
                            print(f"[DIAGNOSTIC] Computed consistency loss: {consistency_loss.item():.6f}")
                        
                        # Validate consistency loss
                        if torch.isnan(consistency_loss) or torch.isinf(consistency_loss):
                            print(f"WARNING: NaN/Inf consistency loss at iteration {i}, setting to 0")
                            consistency_loss = torch.tensor(0.0, device=device)
                            num_consistency_skipped += 1
                        else:
                            num_consistency_computed += 1
                    else:
                        if i == 0 and epoch == 0:
                            print(f"[DIAGNOSTIC] Skipping consistency loss - predictions identical")
                        num_consistency_skipped += 1
                
                except Exception as e:
                    print(f"WARNING: Error computing consistency loss at iteration {i}: {e}")
                    import traceback
                    if i == 0 and epoch == 0:
                        traceback.print_exc()
                    consistency_loss = torch.tensor(0.0, device=device)
                    num_consistency_skipped += 1
                    if 'was_training' in locals() and was_training:
                        model.train()

            # Compute GPCL loss if enabled (with real RoI features)
            if use_gpcl and gpcl_criterion is not None:
                try:
                    # Get RoI features captured during forward pass
                    roi_features = None
                    if hasattr(model, 'get_roi_features'):
                        roi_features = model.get_roi_features()
                    elif hasattr(model, 'last_roi_features'):
                        roi_features = model.last_roi_features
                    
                    if roi_features is not None and 'box_features' in roi_features:
                        # Compute GPCL using actual RoI features
                        gpcl_loss = gpcl_criterion(roi_features)
                        
                        if torch.isnan(gpcl_loss) or torch.isinf(gpcl_loss):
                            if i == 0 and epoch == 0:
                                print(f"[DIAGNOSTIC] GPCL loss is NaN/Inf, setting to 0")
                            gpcl_loss = torch.tensor(0.0, device=device)
                        else:
                            num_gpcl_computed += 1
                            if i == 0 and epoch == 0:
                                num_features = roi_features['box_features'].shape[0]
                                feature_dim = roi_features['box_features'].shape[1]
                                print(f"[DIAGNOSTIC] Computed GPCL loss: {gpcl_loss.item():.6f}")
                                print(f"[DIAGNOSTIC] Using {num_features} RoI features of dim {feature_dim}")
                    else:
                        if i == 0 and epoch == 0:
                            print(f"[DIAGNOSTIC] No RoI features available for GPCL, setting to 0")
                        gpcl_loss = torch.tensor(0.0, device=device)
                    
                except Exception as e:
                    if i == 0 and epoch == 0:
                        print(f"[DIAGNOSTIC] Error computing GPCL loss: {e}")
                        import traceback
                        traceback.print_exc()
                    gpcl_loss = torch.tensor(0.0, device=device)

            # Total loss with gradient clipping prevention
            total_loss = det_loss + consistency_weight * consistency_loss + gpcl_weight * gpcl_loss

            # Final NaN check
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"WARNING: NaN/Inf total loss at iteration {i}, skipping batch")
                num_nan_batches += 1
                continue

        # Backward pass
        if scaler is not None:
            scaler.scale(total_loss).backward()
            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        # Accumulate losses for logging
        total_det_loss += det_loss.item()
        total_consistency_loss += consistency_loss.item()
        total_gpcl_loss += gpcl_loss.item()
        num_batches += 1

        # Print progress
        if i % print_freq == 0:
            # Extract individual detection losses for detailed logging
            loss_str = ""
            if isinstance(loss_dict, dict):
                for k, v in loss_dict.items():
                    loss_str += f"{k}={v.item():.4f}, "
            
            print(f"Epoch [{epoch}] Iter [{i}/{len(data_loader)}]: "
                  f"{loss_str}"
                  f"det_loss={det_loss.item():.4f}, "
                  f"consistency_loss={consistency_loss.item():.4f}, "
                  f"gpcl_loss={gpcl_loss.item():.4f}, "
                  f"total_loss={total_loss.item():.4f}")

    # Report NaN batches
    if num_nan_batches > 0:
        print(f"WARNING: {num_nan_batches} batches skipped due to NaN/Inf losses")

    # Report consistency computation stats
    print(f"\n[DIAGNOSTIC] Consistency Loss Stats:")
    print(f"  Computed: {num_consistency_computed}/{num_batches}")
    print(f"  Skipped: {num_consistency_skipped}/{num_batches}")
    
    if use_gpcl:
        print(f"[DIAGNOSTIC] GPCL Loss Stats:")
        print(f"  Computed: {num_gpcl_computed}/{num_batches}")

    # Return average losses
    avg_det_loss = total_det_loss / num_batches if num_batches > 0 else 0.0
    avg_consistency_loss = total_consistency_loss / num_batches if num_batches > 0 else 0.0
    avg_gpcl_loss = total_gpcl_loss / num_batches if num_batches > 0 else 0.0

    print(f"\n[Epoch {epoch} Summary] "
          f"Avg Detection Loss: {avg_det_loss:.4f}, "
          f"Avg Consistency Loss: {avg_consistency_loss:.4f}, "
          f"Avg GPCL Loss: {avg_gpcl_loss:.4f}, "
          f"Batches processed: {num_batches}, Batches skipped: {num_nan_batches}")

    return {
        "det_loss": avg_det_loss,
        "consistency_loss": avg_consistency_loss,
        "gpcl_loss": avg_gpcl_loss,
        "total_loss": avg_det_loss + consistency_weight * avg_consistency_loss + gpcl_weight * avg_gpcl_loss,
        "num_batches": num_batches,
        "num_nan_batches": num_nan_batches,
        "num_consistency_computed": num_consistency_computed,
        "num_gpcl_computed": num_gpcl_computed
    }