#!/usr/bin/env python3
"""
Quick test script to verify SHM components are working.

Usage:
    python scripts/test_shm_components.py
"""

import sys
import torch

print("="*60)
print("SHM Component Testing Script")
print("="*60)

# Test 1: Import checks
print("\n[Test 1/5] Checking imports...")
try:
    from src.augmentation.style_hallucination import ImageStyleMixer, StyleHallucination
    from src.models.shm_detector import SHMDetector, wrap_model_with_shm
    from src.losses.consistency_losses import classification_consistency, box_consistency
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: ImageStyleMixer
print("\n[Test 2/5] Testing ImageStyleMixer...")
try:
    mixer = ImageStyleMixer(mix_prob=1.0, alpha=0.2)
    images = [torch.randn(3, 256, 256) for _ in range(4)]
    orig, mixed = mixer.mix_batch(images)
    
    assert len(orig) == 4, "Original batch size mismatch"
    assert len(mixed) == 4, "Mixed batch size mismatch"
    assert orig[0].shape == (3, 256, 256), "Image shape mismatch"
    
    # Check that mixing actually changed statistics (check all images, at least one should change)
    any_changed = False
    for i in range(len(orig)):
        orig_mean = orig[i].mean().item()
        mixed_mean = mixed[i].mean().item()
        if abs(orig_mean - mixed_mean) > 0.001:  # More lenient threshold
            any_changed = True
            print(f"  Image {i}: orig mean={orig_mean:.4f}, mixed mean={mixed_mean:.4f}")
    
    assert any_changed, "Style mixing didn't change any image statistics"
    
    print(f"✅ Image mixer works (verified statistics changed)")
except Exception as e:
    print(f"❌ ImageStyleMixer test failed: {e}")
    sys.exit(1)

# Test 3: StyleHallucination
print("\n[Test 3/5] Testing StyleHallucination...")
try:
    shm = StyleHallucination(style_dim=256, base_style_num=16)
    features = torch.randn(2, 256, 32, 32)
    f_orig, f_hall = shm(features)
    
    assert f_orig.shape == (2, 256, 32, 32), "Original features shape mismatch"
    assert f_hall.shape == (2, 256, 32, 32), "Hallucinated features shape mismatch"
    
    # Check that hallucination changed features
    diff = (f_orig - f_hall).abs().mean().item()
    assert diff > 0.01, "Feature hallucination didn't change features"
    
    print(f"✅ Feature hallucination works (mean diff: {diff:.3f})")
except Exception as e:
    print(f"❌ StyleHallucination test failed: {e}")
    sys.exit(1)

# Test 4: Consistency losses
print("\n[Test 4/5] Testing consistency losses...")
try:
    # Classification consistency
    p1 = torch.softmax(torch.randn(10, 5), dim=1)
    p2 = torch.softmax(torch.randn(10, 5), dim=1)
    cls_loss = classification_consistency(p1, p2)
    assert cls_loss.item() >= 0, "Classification loss should be non-negative"
    
    # Box consistency
    boxes1 = torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=torch.float32)
    boxes2 = torch.tensor([[12, 12, 52, 52], [98, 98, 148, 148]], dtype=torch.float32)
    box_loss = box_consistency(boxes1, boxes2)
    assert box_loss.item() >= 0, "Box loss should be non-negative"
    assert box_loss.item() < 1.0, "Box loss should be < 1 for similar boxes"
    
    print(f"✅ Consistency losses work (cls: {cls_loss.item():.3f}, box: {box_loss.item():.3f})")
except Exception as e:
    print(f"❌ Consistency loss test failed: {e}")
    sys.exit(1)

# Test 5: Model wrapping
print("\n[Test 5/5] Testing model wrapping...")
try:
    from src.models.model_builder import build_model
    
    # Build base model
    model = build_model('fasterrcnn_resnet50_fpn', num_classes=6, pretrained_backbone=False)
    
    # Test with SHM disabled
    config_off = {'shm': {'enable': False}}
    wrapped_off = wrap_model_with_shm(model, config_off)
    assert type(wrapped_off).__name__ == 'FasterRCNN', "Should return original model when disabled"
    
    # Test with SHM enabled
    config_on = {
        'shm': {
            'enable': True,
            'image_level': True,
            'feature_level': False,
            'image_mix_prob': 0.5,
            'image_alpha': 0.2
        }
    }
    wrapped_on = wrap_model_with_shm(model, config_on)
    assert type(wrapped_on).__name__ == 'SHMDetector', "Should return SHMDetector when enabled"
    
    print(f"✅ Model wrapping works (off: {type(wrapped_off).__name__}, on: {type(wrapped_on).__name__})")
except Exception as e:
    print(f"❌ Model wrapping test failed: {e}")
    sys.exit(1)

# All tests passed
print("\n" + "="*60)
print("🎉 ALL TESTS PASSED!")
print("="*60)
print("\nYou can now:")
print("1. Run training with SHM disabled (baseline)")
print("2. Enable SHM in configs/train_config.yaml")
print("3. Compare results using scripts/benchmark_shm.py")
print("\nSee docs/SHM_USAGE_EXAMPLES.md for detailed usage.")
