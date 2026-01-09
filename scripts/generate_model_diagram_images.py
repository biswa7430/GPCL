"""Generate images for model diagram visualization.

This script extracts images at different stages of the SHM model pipeline:
1. Original images from two domains (IndraEye and VisDrone)
2. Style reference images (randomly selected partners)
3. Style-mixed images (after AdaIN)
4. Final input images to detector

Output: Saved images for creating model architecture diagram.
"""
import os
import sys
import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.augmentation.style_hallucination import ImageStyleMixer


def load_random_image(dataset_path, split='train'):
    """Load a random image from dataset."""
    img_dir = os.path.join(dataset_path, split)
    if not os.path.exists(img_dir):
        img_dir = os.path.join(dataset_path, 'images', split)
    
    if not os.path.exists(img_dir):
        raise ValueError(f"Image directory not found: {img_dir}")
    
    # Get all jpg/png images
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        raise ValueError(f"No images found in {img_dir}")
    
    # Select random image
    img_name = random.choice(images)
    img_path = os.path.join(img_dir, img_name)
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    return img, img_name


def pil_to_tensor(img):
    """Convert PIL image to tensor [C, H, W] normalized to [0, 1]."""
    return TF.to_tensor(img)


def tensor_to_pil(tensor):
    """Convert tensor [C, H, W] to PIL image."""
    return TF.to_pil_image(tensor.cpu().clamp(0, 1))


def visualize_style_mixing_process(indra_img, visdrone_img, output_dir):
    """Visualize the complete style mixing process."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to tensors
    indra_tensor = pil_to_tensor(indra_img)
    visdrone_tensor = pil_to_tensor(visdrone_img)
    
    # Resize to similar sizes for better visualization (optional)
    target_size = (640, 640)
    indra_resized = TF.resize(indra_tensor, target_size)
    visdrone_resized = TF.resize(visdrone_tensor, target_size)
    
    # Save original images (Stage 1: Input)
    print("\n1. Saving INPUT images (original from datasets)...")
    indra_pil = tensor_to_pil(indra_resized)
    visdrone_pil = tensor_to_pil(visdrone_resized)
    
    # Clear naming for the 4 stages
    indra_pil.save(os.path.join(output_dir, 'Stage1_INPUT_IndraEye_Original.png'))
    visdrone_pil.save(os.path.join(output_dir, 'Stage1_INPUT_VisDrone_Original.png'))
    
    # Keep old names for backward compatibility
    indra_pil.save(os.path.join(output_dir, '1_indra_original.png'))
    visdrone_pil.save(os.path.join(output_dir, '1_visdrone_original.png'))
    print(f"   Saved: Stage1_INPUT_*.png")
    
    # Create style mixers with different alpha values for better visualization
    mixer_strong = ImageStyleMixer(mix_prob=1.0, alpha=0.8)  # Strong effect for diagram
    mixer_medium = ImageStyleMixer(mix_prob=1.0, alpha=0.6)  # Medium effect
    mixer_training = ImageStyleMixer(mix_prob=1.0, alpha=0.5)  # Training setting
    
    # Scenario 1: IndraEye as content, VisDrone as style
    print("\n2. Creating STYLE-MIXED images with different intensities...")
    print("   Scenario 1: IndraEye (content) + VisDrone (style)")
    
    # Create multiple versions with different alpha values
    indra_mixed_strong = mixer_strong._style_mix_single(indra_resized, visdrone_resized)
    indra_mixed_medium = mixer_medium._style_mix_single(indra_resized, visdrone_resized)
    indra_mixed_training = mixer_training._style_mix_single(indra_resized, visdrone_resized)
    
    indra_mixed_strong_pil = tensor_to_pil(indra_mixed_strong)
    indra_mixed_medium_pil = tensor_to_pil(indra_mixed_medium)
    indra_mixed_training_pil = tensor_to_pil(indra_mixed_training)
    
    # Save with clear stage names
    indra_mixed_strong_pil.save(os.path.join(output_dir, 'Stage3_STYLE_MIXED_IndraEye_with_VisDrone_style_STRONG.png'))
    indra_mixed_medium_pil.save(os.path.join(output_dir, 'Stage3_STYLE_MIXED_IndraEye_with_VisDrone_style_MEDIUM.png'))
    indra_mixed_training_pil.save(os.path.join(output_dir, 'Stage3_STYLE_MIXED_IndraEye_with_VisDrone_style.png'))
    
    # Keep old names for backward compatibility
    indra_mixed_strong_pil.save(os.path.join(output_dir, '2_indra_style_from_visdrone_strong.png'))
    indra_mixed_medium_pil.save(os.path.join(output_dir, '2_indra_style_from_visdrone_medium.png'))
    indra_mixed_training_pil.save(os.path.join(output_dir, '2_indra_style_from_visdrone.png'))
    
    # Scenario 2: VisDrone as content, IndraEye as style
    print("   Scenario 2: VisDrone (content) + IndraEye (style)")
    
    visdrone_mixed_strong = mixer_strong._style_mix_single(visdrone_resized, indra_resized)
    visdrone_mixed_medium = mixer_medium._style_mix_single(visdrone_resized, indra_resized)
    visdrone_mixed_training = mixer_training._style_mix_single(visdrone_resized, indra_resized)
    
    visdrone_mixed_strong_pil = tensor_to_pil(visdrone_mixed_strong)
    visdrone_mixed_medium_pil = tensor_to_pil(visdrone_mixed_medium)
    visdrone_mixed_training_pil = tensor_to_pil(visdrone_mixed_training)
    
    # Save with clear stage names
    visdrone_mixed_strong_pil.save(os.path.join(output_dir, 'Stage3_STYLE_MIXED_VisDrone_with_IndraEye_style_STRONG.png'))
    visdrone_mixed_medium_pil.save(os.path.join(output_dir, 'Stage3_STYLE_MIXED_VisDrone_with_IndraEye_style_MEDIUM.png'))
    visdrone_mixed_training_pil.save(os.path.join(output_dir, 'Stage3_STYLE_MIXED_VisDrone_with_IndraEye_style.png'))
    
    # Keep old names for backward compatibility
    visdrone_mixed_strong_pil.save(os.path.join(output_dir, '2_visdrone_style_from_indra_strong.png'))
    visdrone_mixed_medium_pil.save(os.path.join(output_dir, '2_visdrone_style_from_indra_medium.png'))
    visdrone_mixed_training_pil.save(os.path.join(output_dir, '2_visdrone_style_from_indra.png'))
    
    print(f"   Saved: Stage3_STYLE_MIXED_* (Strong α=0.8, Medium α=0.6, Training α=0.5)")
    
    # Save style reference images (Stage 2: showing which image provided the style)
    print("\n3. Saving STYLE REFERENCE images...")
    visdrone_pil.save(os.path.join(output_dir, 'Stage2_STYLE_REFERENCE_VisDrone_for_IndraEye.png'))
    indra_pil.save(os.path.join(output_dir, 'Stage2_STYLE_REFERENCE_IndraEye_for_VisDrone.png'))
    
    # Keep old names for backward compatibility
    visdrone_pil.save(os.path.join(output_dir, '3_style_ref_visdrone.png'))
    indra_pil.save(os.path.join(output_dir, '3_style_ref_indra.png'))
    print(f"   Saved: Stage2_STYLE_REFERENCE_*.png")
    
    # Save final detector input images (Stage 4: same as style-mixed, but labeled as detector input)
    print("\n4. Saving DETECTOR INPUT images (augmented/mixed images)...")
    indra_mixed_strong_pil.save(os.path.join(output_dir, 'Stage4_DETECTOR_INPUT_IndraEye_augmented_STRONG.png'))
    visdrone_mixed_strong_pil.save(os.path.join(output_dir, 'Stage4_DETECTOR_INPUT_VisDrone_augmented_STRONG.png'))
    print(f"   Saved: Stage4_DETECTOR_INPUT_*.png")
    
    # Create comparison visualizations showing different alpha effects
    print("\n5. Creating alpha comparison visualizations...")
    create_alpha_comparison(
        indra_pil, visdrone_pil,
        indra_mixed_training_pil, indra_mixed_medium_pil, indra_mixed_strong_pil,
        visdrone_mixed_training_pil, visdrone_mixed_medium_pil, visdrone_mixed_strong_pil,
        output_dir
    )
    
    # Create a comprehensive visualization (using strong effect for clarity)
    print("\n5. Creating comprehensive visualization...")
    create_comprehensive_figure(
        indra_pil, visdrone_pil,
        indra_mixed_strong_pil, visdrone_mixed_strong_pil,
        output_dir
    )
    
    # Create statistical visualization
    print("\n6. Creating statistical analysis visualization...")
    create_statistical_visualization(
        indra_pil, visdrone_pil,
        indra_mixed_strong_pil, visdrone_mixed_strong_pil,
        output_dir
    )
    
    # Create individual process diagrams (using strong effect)
    print("\n7. Creating process flow diagrams...")
    create_process_diagram_indra(indra_pil, visdrone_pil, indra_mixed_strong_pil, output_dir)
    create_process_diagram_visdrone(visdrone_pil, indra_pil, visdrone_mixed_strong_pil, output_dir)
    create_comprehensive_figure(
        indra_pil, visdrone_pil,
        indra_mixed_strong_pil, visdrone_mixed_strong_pil,
        output_dir
    )
    
    # Create statistical visualization
    print("\n6. Creating statistical analysis visualization...")
    create_statistical_visualization(
        indra_pil, visdrone_pil,
        indra_mixed_strong_pil, visdrone_mixed_strong_pil,
        output_dir
    )
    
    # Create individual process diagrams (using strong effect)
    print("\n7. Creating process flow diagrams...")
    create_process_diagram_indra(indra_pil, visdrone_pil, indra_mixed_strong_pil, output_dir)
    create_process_diagram_visdrone(visdrone_pil, indra_pil, visdrone_mixed_strong_pil, output_dir)
    
    print(f"\n✓ All images saved to: {output_dir}")
    return output_dir


def create_alpha_comparison(indra_orig, visdrone_orig, 
                            indra_weak, indra_medium, indra_strong,
                            visdrone_weak, visdrone_medium, visdrone_strong,
                            output_dir):
    """Create comparison showing different alpha mixing strengths."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Style Mixing Intensity Comparison (Different α values)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Row 1: IndraEye mixing progression
    axes[0, 0].imshow(indra_orig)
    axes[0, 0].set_title('Original\n(IndraEye)', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(indra_weak)
    axes[0, 1].set_title('Weak Mixing\n(α=0.5)', fontweight='bold', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(indra_medium)
    axes[0, 2].set_title('Medium Mixing\n(α=0.6)', fontweight='bold', fontsize=11)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(indra_strong)
    axes[0, 3].set_title('Strong Mixing\n(α=0.8)', fontweight='bold', fontsize=11)
    axes[0, 3].axis('off')
    add_border(axes[0, 3], 'red', 3)
    
    # Row 2: VisDrone mixing progression
    axes[1, 0].imshow(visdrone_orig)
    axes[1, 0].set_title('Original\n(VisDrone)', fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(visdrone_weak)
    axes[1, 1].set_title('Weak Mixing\n(α=0.5)', fontweight='bold', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(visdrone_medium)
    axes[1, 2].set_title('Medium Mixing\n(α=0.6)', fontweight='bold', fontsize=11)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(visdrone_strong)
    axes[1, 3].set_title('Strong Mixing\n(α=0.8)', fontweight='bold', fontsize=11)
    axes[1, 3].axis('off')
    add_border(axes[1, 3], 'red', 3)
    
    # Add text annotation
    fig.text(0.5, 0.02, 
             'Note: α=0.8 (strong) used for diagram visualization clarity. Training uses α=0.5.', 
             ha='center', fontsize=12, style='italic', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '6_alpha_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 6_alpha_comparison.png")


def create_statistical_visualization(indra_orig, visdrone_orig, indra_mixed, visdrone_mixed, output_dir):
    """Create visualization showing statistical changes (mean and std)."""
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('Statistical Analysis: Color Distribution Transfer via AdaIN', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Convert PIL to numpy for histogram computation
    indra_np = np.array(indra_orig)
    visdrone_np = np.array(visdrone_orig)
    indra_mixed_np = np.array(indra_mixed)
    visdrone_mixed_np = np.array(visdrone_mixed)
    
    colors = ['red', 'green', 'blue']
    color_names = ['Red', 'Green', 'Blue']
    
    # Row 1: IndraEye → mixed
    axes[0, 0].imshow(indra_orig)
    axes[0, 0].set_title('Original (IndraEye)', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Histogram of original IndraEye
    axes[0, 1].set_title('Color Distribution', fontweight='bold')
    for i, (color, name) in enumerate(zip(colors, color_names)):
        axes[0, 1].hist(indra_np[:,:,i].ravel(), bins=50, alpha=0.5, color=color, label=name)
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[0, 2].imshow(visdrone_orig)
    axes[0, 2].set_title('Style Source (VisDrone)', fontweight='bold')
    axes[0, 2].axis('off')
    add_border(axes[0, 2], 'green', 3)
    
    axes[0, 3].imshow(indra_mixed)
    axes[0, 3].set_title('Style-Mixed Result', fontweight='bold')
    axes[0, 3].axis('off')
    add_border(axes[0, 3], 'red', 3)
    
    # Histogram of mixed IndraEye
    axes[0, 4].set_title('Color Distribution (After)', fontweight='bold')
    for i, (color, name) in enumerate(zip(colors, color_names)):
        axes[0, 4].hist(indra_mixed_np[:,:,i].ravel(), bins=50, alpha=0.5, color=color, label=name)
    axes[0, 4].legend()
    axes[0, 4].set_xlabel('Pixel Value')
    axes[0, 4].set_ylabel('Frequency')
    
    # Row 2: VisDrone → mixed
    axes[1, 0].imshow(visdrone_orig)
    axes[1, 0].set_title('Original (VisDrone)', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Histogram of original VisDrone
    axes[1, 1].set_title('Color Distribution', fontweight='bold')
    for i, (color, name) in enumerate(zip(colors, color_names)):
        axes[1, 1].hist(visdrone_np[:,:,i].ravel(), bins=50, alpha=0.5, color=color, label=name)
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    
    axes[1, 2].imshow(indra_orig)
    axes[1, 2].set_title('Style Source (IndraEye)', fontweight='bold')
    axes[1, 2].axis('off')
    add_border(axes[1, 2], 'green', 3)
    
    axes[1, 3].imshow(visdrone_mixed)
    axes[1, 3].set_title('Style-Mixed Result', fontweight='bold')
    axes[1, 3].axis('off')
    add_border(axes[1, 3], 'red', 3)
    
    # Histogram of mixed VisDrone
    axes[1, 4].set_title('Color Distribution (After)', fontweight='bold')
    for i, (color, name) in enumerate(zip(colors, color_names)):
        axes[1, 4].hist(visdrone_mixed_np[:,:,i].ravel(), bins=50, alpha=0.5, color=color, label=name)
    axes[1, 4].legend()
    axes[1, 4].set_xlabel('Pixel Value')
    axes[1, 4].set_ylabel('Frequency')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '7_statistical_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 7_statistical_analysis.png")


def create_comprehensive_figure(indra_orig, visdrone_orig, indra_mixed, visdrone_mixed, output_dir):
    """Create a comprehensive 2x4 figure showing the complete pipeline."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Style Hallucination Module (SHM) - Image-Level Mixing Process (α=0.8 for visualization)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Row 1: IndraEye domain
    axes[0, 0].imshow(indra_orig)
    axes[0, 0].set_title('Original\n(IndraEye Domain)', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    add_border(axes[0, 0], 'blue', 3)
    
    axes[0, 1].imshow(visdrone_orig)
    axes[0, 1].set_title('Style Reference\n(VisDrone Domain)', fontweight='bold', fontsize=12)
    axes[0, 1].axis('off')
    add_border(axes[0, 1], 'green', 3)
    
    axes[0, 2].imshow(indra_mixed)
    axes[0, 2].set_title('Style-Mixed Image\n(AdaIN α=0.8)', fontweight='bold', fontsize=12)
    axes[0, 2].axis('off')
    add_border(axes[0, 2], 'red', 3)
    
    axes[0, 3].imshow(indra_mixed)
    axes[0, 3].set_title('Input to Detector\n(Augmented)', fontweight='bold', fontsize=12)
    axes[0, 3].axis('off')
    add_border(axes[0, 3], 'purple', 3)
    
    # Row 2: VisDrone domain
    axes[1, 0].imshow(visdrone_orig)
    axes[1, 0].set_title('Original\n(VisDrone Domain)', fontweight='bold', fontsize=12)
    axes[1, 0].axis('off')
    add_border(axes[1, 0], 'blue', 3)
    
    axes[1, 1].imshow(indra_orig)
    axes[1, 1].set_title('Style Reference\n(IndraEye Domain)', fontweight='bold', fontsize=12)
    axes[1, 1].axis('off')
    add_border(axes[1, 1], 'green', 3)
    
    axes[1, 2].imshow(visdrone_mixed)
    axes[1, 2].set_title('Style-Mixed Image\n(AdaIN α=0.8)', fontweight='bold', fontsize=12)
    axes[1, 2].axis('off')
    add_border(axes[1, 2], 'red', 3)
    
    axes[1, 3].imshow(visdrone_mixed)
    axes[1, 3].set_title('Input to Detector\n(Augmented)', fontweight='bold', fontsize=12)
    axes[1, 3].axis('off')
    add_border(axes[1, 3], 'purple', 3)
    
    # Add arrows between stages
    add_arrow_annotations(fig, axes)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '4_comprehensive_pipeline.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 4_comprehensive_pipeline.png")


def create_process_diagram_indra(indra_orig, style_ref, mixed, output_dir):
    """Create process flow for IndraEye image."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('SHM Process Flow: IndraEye → Mixed Image', 
                 fontsize=14, fontweight='bold')
    
    axes[0].imshow(indra_orig)
    axes[0].set_title('1. Input Image\n(IndraEye)', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(style_ref)
    axes[1].set_title('2. Random Style Ref\n(VisDrone)', fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(mixed)
    axes[2].set_title('3. AdaIN Mixing\n(μ, σ transfer)', fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(mixed)
    axes[3].set_title('4. Augmented Input\n(to Detector)', fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '5_process_indra.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 5_process_indra.png")


def create_process_diagram_visdrone(visdrone_orig, style_ref, mixed, output_dir):
    """Create process flow for VisDrone image."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('SHM Process Flow: VisDrone → Mixed Image', 
                 fontsize=14, fontweight='bold')
    
    axes[0].imshow(visdrone_orig)
    axes[0].set_title('1. Input Image\n(VisDrone)', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(style_ref)
    axes[1].set_title('2. Random Style Ref\n(IndraEye)', fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(mixed)
    axes[2].set_title('3. AdaIN Mixing\n(μ, σ transfer)', fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(mixed)
    axes[3].set_title('4. Augmented Input\n(to Detector)', fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '5_process_visdrone.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: 5_process_visdrone.png")


def add_border(ax, color, linewidth):
    """Add colored border to axis."""
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(linewidth)


def add_arrow_annotations(fig, axes):
    """Add arrows showing data flow between stages."""
    # This is a simplified version - matplotlib arrows between subplots are tricky
    # In practice, you might want to add these manually in your paper figure
    pass


def create_latex_figure_code(output_dir):
    """Generate LaTeX code for including these images in paper."""
    latex_code = r"""
% LaTeX code for model diagram figure
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/4_comprehensive_pipeline.png}
    \caption{Style Hallucination Module (SHM) pipeline showing image-level style mixing. 
    Top row: IndraEye domain image mixed with VisDrone style. 
    Bottom row: VisDrone domain image mixed with IndraEye style. 
    The Adaptive Instance Normalization (AdaIN) transfers mean and standard deviation 
    statistics across domains, creating synthetic style variations for domain generalization.}
    \label{fig:shm_pipeline}
\end{figure*}

% Individual process flows
\begin{figure}[t]
    \centering
    \begin{subfigure}{\textwidth}
        \includegraphics[width=\textwidth]{figures/5_process_indra.png}
        \caption{IndraEye domain processing}
    \end{subfigure}
    \begin{subfigure}{\textwidth}
        \includegraphics[width=\textwidth]{figures/5_process_visdrone.png}
        \caption{VisDrone domain processing}
    \end{subfigure}
    \caption{Step-by-step visualization of SHM image-level mixing process.}
    \label{fig:shm_process}
\end{figure}
"""
    
    latex_file = os.path.join(output_dir, 'latex_figure_code.tex')
    with open(latex_file, 'w') as f:
        f.write(latex_code)
    print(f"\n✓ LaTeX figure code saved to: latex_figure_code.tex")


def main():
    """Main execution."""
    print("="*80)
    print("GENERATING MODEL DIAGRAM IMAGES")
    print("="*80)
    
    # Use random seed based on current time for different images each run
    import time
    seed = int(time.time())
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\nRandom seed: {seed} (different images each run)")
    
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    indra_path = os.path.join(project_root, 'data', 'indraEye_dataset')
    visdrone_path = os.path.join(project_root, 'data', 'visdrone_dataset')
    output_dir = os.path.join(project_root, 'outputs', 'model_diagram_images')
    
    # Check if datasets exist
    if not os.path.exists(indra_path):
        print(f"ERROR: IndraEye dataset not found at {indra_path}")
        return
    if not os.path.exists(visdrone_path):
        print(f"ERROR: VisDrone dataset not found at {visdrone_path}")
        return
    
    # Load random images from each domain
    print("\nLoading images...")
    print(f"  IndraEye dataset: {indra_path}")
    indra_img, indra_name = load_random_image(indra_path, split='train')
    print(f"  ✓ Loaded IndraEye image: {indra_name} ({indra_img.size})")
    
    print(f"  VisDrone dataset: {visdrone_path}")
    visdrone_img, visdrone_name = load_random_image(visdrone_path, split='train')
    print(f"  ✓ Loaded VisDrone image: {visdrone_name} ({visdrone_img.size})")
    
    # Visualize style mixing process
    print("\nGenerating visualizations...")
    output_path = visualize_style_mixing_process(indra_img, visdrone_img, output_dir)
    
    # Generate LaTeX code
    print("\nGenerating LaTeX figure code...")
    create_latex_figure_code(output_dir)
    
    print("\n" + "="*80)
    print("SUMMARY - 4 KEY STAGES FOR YOUR MODEL DIAGRAM")
    print("="*80)
    print(f"\n🎯 MAIN IMAGES (Clear Stage Names):")
    print(f"\n  Stage 1: INPUT (Original Images)")
    print(f"    - Stage1_INPUT_IndraEye_Original.png")
    print(f"    - Stage1_INPUT_VisDrone_Original.png")
    print(f"\n  Stage 2: STYLE REFERENCE (Randomly Selected)")
    print(f"    - Stage2_STYLE_REFERENCE_VisDrone_for_IndraEye.png")
    print(f"    - Stage2_STYLE_REFERENCE_IndraEye_for_VisDrone.png")
    print(f"\n  Stage 3: STYLE MIXED (AdaIN Result) ⭐ MOST IMPORTANT")
    print(f"    - Stage3_STYLE_MIXED_IndraEye_with_VisDrone_style_STRONG.png")
    print(f"    - Stage3_STYLE_MIXED_VisDrone_with_IndraEye_style_STRONG.png")
    print(f"    [Also available: _MEDIUM.png (α=0.6), no suffix (α=0.5)]")
    print(f"\n  Stage 4: DETECTOR INPUT (Final Augmented Images)")
    print(f"    - Stage4_DETECTOR_INPUT_IndraEye_augmented_STRONG.png")
    print(f"    - Stage4_DETECTOR_INPUT_VisDrone_augmented_STRONG.png")
    print(f"\n📊 Comprehensive Visualizations:")
    print(f"  - 4_comprehensive_pipeline.png (All 4 stages in one figure)")
    print(f"  - 5_process_indra.png (Step-by-step flow: IndraEye)")
    print(f"  - 5_process_visdrone.png (Step-by-step flow: VisDrone)")
    print(f"  - 6_alpha_comparison.png (α effect comparison)")
    print(f"  - 7_statistical_analysis.png (Color histogram changes)")
    print(f"\n📁 All files saved to: {output_path}")
    print(f"\n✨ RECOMMENDATION:")
    print(f"  • Use Stage*_*_STRONG.png files (α=0.8) for best visual clarity")
    print(f"  • File names now clearly indicate: INPUT → STYLE REF → STYLE MIXED → DETECTOR INPUT")
    print(f"  • Note in paper caption: visualization uses α=0.8, training uses α=0.5")
    print(f"\n💡 Each run generates different random images from the datasets!")
    print(f"   Run multiple times to find the best visual examples.")
    print("="*80)


if __name__ == '__main__':
    main()
