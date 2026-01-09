"""Generate model diagram images showing BOTH same-domain and cross-domain style mixing.

This helps explain that training uses BOTH strategies:
- Same-domain: subtle intra-domain variations  
- Cross-domain: aggressive domain shifts for generalization
"""
import os
import sys
import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.augmentation.style_hallucination import ImageStyleMixer


def load_random_images(indra_path, visdrone_path, split='train', num_each=2):
    """Load multiple random images from each dataset."""
    # Load IndraEye images
    indra_dir = os.path.join(indra_path, split)
    if not os.path.exists(indra_dir):
        indra_dir = os.path.join(indra_path, 'images', split)
    
    indra_imgs = [f for f in os.listdir(indra_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    selected_indra = random.sample(indra_imgs, num_each)
    
    # Load VisDrone images
    visdrone_dir = os.path.join(visdrone_path, split)
    if not os.path.exists(visdrone_dir):
        visdrone_dir = os.path.join(visdrone_path, 'images', split)
    
    visdrone_imgs = [f for f in os.listdir(visdrone_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    selected_visdrone = random.sample(visdrone_imgs, num_each)
    
    # Load actual images
    indra_pil_imgs = [Image.open(os.path.join(indra_dir, f)).convert('RGB') for f in selected_indra]
    visdrone_pil_imgs = [Image.open(os.path.join(visdrone_dir, f)).convert('RGB') for f in selected_visdrone]
    
    return indra_pil_imgs, visdrone_pil_imgs, selected_indra, selected_visdrone


def create_comprehensive_comparison(output_dir, indra_imgs, visdrone_imgs):
    """Create comparison showing both same-domain and cross-domain mixing."""
    
    target_size = (640, 640)
    mixer_strong = ImageStyleMixer(mix_prob=1.0, alpha=0.8)
    
    # Convert to tensors and resize
    indra1_t = TF.resize(TF.to_tensor(indra_imgs[0]), target_size)
    indra2_t = TF.resize(TF.to_tensor(indra_imgs[1]), target_size)
    visdrone1_t = TF.resize(TF.to_tensor(visdrone_imgs[0]), target_size)
    visdrone2_t = TF.resize(TF.to_tensor(visdrone_imgs[1]), target_size)
    
    # Same-domain mixing
    indra_same = mixer_strong._style_mix_single(indra1_t, indra2_t)  # IndraEye + IndraEye style
    visdrone_same = mixer_strong._style_mix_single(visdrone1_t, visdrone2_t)  # VisDrone + VisDrone style
    
    # Cross-domain mixing
    indra_cross = mixer_strong._style_mix_single(indra1_t, visdrone1_t)  # IndraEye + VisDrone style
    visdrone_cross = mixer_strong._style_mix_single(visdrone1_t, indra1_t)  # VisDrone + IndraEye style
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Style Mixing Strategies: Same-Domain vs Cross-Domain (α=0.8)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Row 1: IndraEye content
    axes[0, 0].imshow(TF.to_pil_image(indra1_t))
    axes[0, 0].set_title('Original\n(IndraEye)', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(TF.to_pil_image(indra2_t))
    axes[0, 1].set_title('Style Ref\n(IndraEye)', fontweight='bold', fontsize=12)
    axes[0, 1].axis('off')
    axes[0, 1].text(0.5, -0.1, 'Same Domain', ha='center', transform=axes[0, 1].transAxes,
                    fontsize=11, color='blue', fontweight='bold')
    
    axes[0, 2].imshow(TF.to_pil_image(indra_same))
    axes[0, 2].set_title('Mixed (Same)\nSubtle variation', fontweight='bold', fontsize=12)
    axes[0, 2].axis('off')
    add_border(axes[0, 2], 'blue', 3)
    
    axes[0, 3].imshow(TF.to_pil_image(visdrone1_t))
    axes[0, 3].set_title('Style Ref\n(VisDrone)', fontweight='bold', fontsize=12)
    axes[0, 3].axis('off')
    axes[0, 3].text(0.5, -0.1, 'Cross Domain', ha='center', transform=axes[0, 3].transAxes,
                    fontsize=11, color='red', fontweight='bold')
    
    # Add arrow and mixed result
    axes[0, 3].text(1.1, 0.5, '→', transform=axes[0, 3].transAxes, fontsize=30, 
                    ha='center', va='center')
    
    # Create inset for cross-domain mixed result
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(axes[0, 3], width="100%", height="100%", 
                       bbox_to_anchor=(1.2, 0, 1, 1), bbox_transform=axes[0, 3].transAxes)
    axins.imshow(TF.to_pil_image(indra_cross))
    axins.set_title('Mixed (Cross)\nDramatic shift', fontweight='bold', fontsize=12)
    axins.axis('off')
    add_border(axins, 'red', 3)
    
    # Row 2: VisDrone content
    axes[1, 0].imshow(TF.to_pil_image(visdrone1_t))
    axes[1, 0].set_title('Original\n(VisDrone)', fontweight='bold', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(TF.to_pil_image(visdrone2_t))
    axes[1, 1].set_title('Style Ref\n(VisDrone)', fontweight='bold', fontsize=12)
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, -0.1, 'Same Domain', ha='center', transform=axes[1, 1].transAxes,
                    fontsize=11, color='blue', fontweight='bold')
    
    axes[1, 2].imshow(TF.to_pil_image(visdrone_same))
    axes[1, 2].set_title('Mixed (Same)\nSubtle variation', fontweight='bold', fontsize=12)
    axes[1, 2].axis('off')
    add_border(axes[1, 2], 'blue', 3)
    
    axes[1, 3].imshow(TF.to_pil_image(indra1_t))
    axes[1, 3].set_title('Style Ref\n(IndraEye)', fontweight='bold', fontsize=12)
    axes[1, 3].axis('off')
    axes[1, 3].text(0.5, -0.1, 'Cross Domain', ha='center', transform=axes[1, 3].transAxes,
                    fontsize=11, color='red', fontweight='bold')
    
    # Add arrow and mixed result
    axes[1, 3].text(1.1, 0.5, '→', transform=axes[1, 3].transAxes, fontsize=30, 
                    ha='center', va='center')
    
    # Create inset for cross-domain mixed result
    axins2 = inset_axes(axes[1, 3], width="100%", height="100%", 
                        bbox_to_anchor=(1.2, 0, 1, 1), bbox_transform=axes[1, 3].transAxes)
    axins2.imshow(TF.to_pil_image(visdrone_cross))
    axins2.set_title('Mixed (Cross)\nDramatic shift', fontweight='bold', fontsize=12)
    axins2.axis('off')
    add_border(axins2, 'red', 3)
    
    # Add explanation
    fig.text(0.5, 0.02, 
             'Training uses BOTH strategies: Same-domain (subtle) + Cross-domain (aggressive) = Robust generalization', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '8_SAME_vs_CROSS_DOMAIN_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: 8_SAME_vs_CROSS_DOMAIN_comparison.png")


def add_border(ax, color, linewidth):
    """Add colored border to axis."""
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(linewidth)


def main():
    """Main execution."""
    print("="*80)
    print("GENERATING SAME-DOMAIN vs CROSS-DOMAIN COMPARISON")
    print("="*80)
    
    # Use random seed
    seed = int(time.time())
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\nRandom seed: {seed}")
    
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    indra_path = os.path.join(project_root, 'data', 'indraEye_dataset')
    visdrone_path = os.path.join(project_root, 'data', 'visdrone_dataset')
    output_dir = os.path.join(project_root, 'outputs', 'model_diagram_images')
    
    # Load images
    print("\nLoading 2 images from each domain...")
    indra_imgs, visdrone_imgs, indra_names, visdrone_names = load_random_images(
        indra_path, visdrone_path, num_each=2
    )
    print(f"  ✓ IndraEye: {indra_names[0]}, {indra_names[1]}")
    print(f"  ✓ VisDrone: {visdrone_names[0]}, {visdrone_names[1]}")
    
    # Create comparison
    print("\nGenerating same-domain vs cross-domain comparison...")
    create_comprehensive_comparison(output_dir, indra_imgs, visdrone_imgs)
    
    print("\n" + "="*80)
    print("✓ DONE!")
    print("="*80)
    print(f"\nGenerated: 8_SAME_vs_CROSS_DOMAIN_comparison.png")
    print(f"Location: {output_dir}")
    print("\nThis figure shows:")
    print("  - Blue borders: Same-domain mixing (subtle)")
    print("  - Red borders: Cross-domain mixing (dramatic)")
    print("\nUse this to explain that training uses BOTH strategies!")
    print("="*80)


if __name__ == '__main__':
    main()
