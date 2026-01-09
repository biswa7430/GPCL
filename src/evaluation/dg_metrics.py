"""
Novel Evaluation Metrics for Domain Generalization in Object Detection

These metrics provide comprehensive evaluation beyond standard mAP:
1. Domain Generalization Gap (DG-GAP)
2. Style-Invariance Score (SI-Score)
3. Cross-Domain Consistency Index (CDCI)
4. Scale-Sensitive Domain Gap (SSDG)
5. Temporal Consistency Score (TCS)

Author: Your Name
Novel Contribution for ICPR 2026
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from typing import List, Dict, Tuple


class DomainGeneralizationMetrics:
    """
    Novel metrics for evaluating domain generalization in object detection
    """
    
    @staticmethod
    def compute_dg_gap(source_perf: float, target_perf: float) -> float:
        """
        Domain Generalization Gap (DG-GAP)
        
        Measures how well model generalizes relative to source performance.
        Normalized metric that accounts for source performance level.
        
        DG-GAP = 1 - (P_target / P_source)
        
        Range: [0, ∞)
        - 0.0 = perfect generalization (target equals source)
        - 0.1 = 10% performance drop
        - 0.5 = 50% performance drop
        
        Lower is better!
        
        Args:
            source_perf: mAP on source (training) domain
            target_perf: mAP on target (test) domain
        
        Returns:
            DG-GAP score
        
        Example:
            Source: 55% mAP, Target: 46.5% mAP
            DG-GAP = 1 - (46.5/55) = 0.154 = 15.4% gap
        """
        if source_perf == 0:
            return float('inf')
        
        dg_gap = 1.0 - (target_perf / source_perf)
        return dg_gap
    
    @staticmethod
    def compute_relative_dg_improvement(baseline_gap: float, method_gap: float) -> float:
        """
        Compute relative improvement in DG-GAP
        
        Args:
            baseline_gap: DG-GAP of baseline method
            method_gap: DG-GAP of proposed method
        
        Returns:
            Relative improvement (positive = better)
        
        Example:
            Baseline DG-GAP: 0.380 (38%)
            Your method DG-GAP: 0.154 (15.4%)
            Improvement: (0.380 - 0.154) = 0.226 = 22.6% absolute reduction
        """
        improvement = baseline_gap - method_gap
        relative_improvement = (improvement / baseline_gap) * 100
        return relative_improvement
    
    @staticmethod
    def compute_style_invariance_score(predictions_orig: List[Dict], 
                                      predictions_augmented: List[Dict]) -> float:
        """
        Style-Invariance Score (SI-Score)
        
        Measures prediction stability across style variations.
        Combines geometric (box) and semantic (class) stability.
        
        SI-Score = mean(IoU(box_orig, box_aug)) × mean(sim(class_orig, class_aug))
        
        Range: [0, 1]
        - 1.0 = perfect invariance
        - 0.5 = moderate invariance
        - 0.0 = no invariance
        
        Higher is better!
        
        Args:
            predictions_orig: Original predictions (list of dicts with 'boxes', 'scores')
            predictions_augmented: Style-augmented predictions
        
        Returns:
            SI-Score value
        """
        box_stabilities = []
        class_stabilities = []
        
        for pred_orig, pred_aug in zip(predictions_orig, predictions_augmented):
            boxes_orig = pred_orig['boxes']
            boxes_aug = pred_aug['boxes']
            scores_orig = pred_orig['scores']
            scores_aug = pred_aug['scores']
            
            if len(boxes_orig) == 0 or len(boxes_aug) == 0:
                continue
            
            # Match boxes (top-K approach)
            K = min(len(boxes_orig), len(boxes_aug))
            boxes_orig = boxes_orig[:K]
            boxes_aug = boxes_aug[:K]
            scores_orig = scores_orig[:K]
            scores_aug = scores_aug[:K]
            
            # Box stability via IoU
            iou = box_iou(boxes_orig, boxes_aug)
            box_stability = torch.diag(iou).mean().item()
            box_stabilities.append(box_stability)
            
            # Class stability via cosine similarity
            if len(scores_orig) > 0 and len(scores_aug) > 0:
                # Handle potential shape mismatches
                if scores_orig.dim() == 1:
                    scores_orig = scores_orig.unsqueeze(1)
                if scores_aug.dim() == 1:
                    scores_aug = scores_aug.unsqueeze(1)
                
                class_sim = F.cosine_similarity(scores_orig, scores_aug, dim=0)
                class_stability = class_sim.mean().item()
                class_stabilities.append(class_stability)
        
        if len(box_stabilities) == 0:
            return 0.0
        
        # Combined score
        box_stability_mean = np.mean(box_stabilities)
        class_stability_mean = np.mean(class_stabilities) if class_stabilities else 1.0
        si_score = box_stability_mean * class_stability_mean
        
        return si_score
    
    @staticmethod
    def compute_cross_domain_consistency_index(performances: List[float]) -> float:
        """
        Cross-Domain Consistency Index (CDCI)
        
        Measures how consistent performance is across multiple domains.
        Lower variance = more consistent = better generalization.
        
        CDCI = 1 - (std(performances) / mean(performances))
        
        Range: [0, 1]
        - 1.0 = perfectly consistent across domains
        - 0.5 = moderate consistency
        - 0.0 = highly inconsistent
        
        Higher is better!
        
        Args:
            performances: List of mAP scores across different domains
        
        Returns:
            CDCI value
        
        Example:
            Your method: [46.5, 37.6, 48.2, 40.1] mAP
            Mean = 43.1, Std = 4.8
            CDCI = 1 - (4.8/43.1) = 0.889
            
            Baseline: [34.2, 28.9, 41.8, 35.4] mAP
            Mean = 35.1, Std = 5.2
            CDCI = 1 - (5.2/35.1) = 0.852
            
            Your method is more consistent!
        """
        if len(performances) < 2:
            return 1.0
        
        perfs = np.array(performances)
        mean_perf = perfs.mean()
        std_perf = perfs.std()
        
        if mean_perf == 0:
            return 0.0
        
        cdci = 1.0 - (std_perf / mean_perf)
        return max(cdci, 0.0)  # Clamp to [0, 1]
    
    @staticmethod
    def compute_scale_sensitive_domain_gap(source_aps: Dict[str, float],
                                          target_aps: Dict[str, float],
                                          weights: Dict[str, float] = None) -> float:
        """
        Scale-Sensitive Domain Gap (SSDG)
        
        Measures domain gap separately for small/medium/large objects.
        Weighted by importance (small objects typically more challenging).
        
        SSDG = Σ weight_scale × |AP_source_scale - AP_target_scale|
        
        Range: [0, 100]
        - 0.0 = no gap across all scales
        - 50.0 = large gap
        
        Lower is better!
        
        Args:
            source_aps: Dict with keys 'AP_S', 'AP_M', 'AP_L' (source domain)
            target_aps: Dict with keys 'AP_S', 'AP_M', 'AP_L' (target domain)
            weights: Optional custom weights for each scale
        
        Returns:
            SSDG value
        
        Example:
            Source: AP_S=18.4, AP_M=42.7, AP_L=51.2
            Target: AP_S=24.6, AP_M=54.3, AP_L=63.8
            
            With default weights [0.5, 0.3, 0.2]:
            SSDG = 0.5×|18.4-24.6| + 0.3×|42.7-54.3| + 0.2×|51.2-63.8|
                 = 0.5×6.2 + 0.3×11.6 + 0.2×12.6
                 = 3.1 + 3.48 + 2.52 = 9.1
        """
        if weights is None:
            # Default: emphasize small objects (harder to detect)
            weights = {'AP_S': 0.5, 'AP_M': 0.3, 'AP_L': 0.2}
        
        ssdg = 0.0
        for scale in ['AP_S', 'AP_M', 'AP_L']:
            if scale in source_aps and scale in target_aps:
                gap = abs(source_aps[scale] - target_aps[scale])
                ssdg += weights.get(scale, 0.33) * gap
        
        return ssdg
    
    @staticmethod
    def compute_temporal_consistency_score(detections_sequence: List[List[Dict]]) -> float:
        """
        Temporal Consistency Score (TCS)
        
        For video sequences: measures detection stability across frames.
        Important for video object detection in drone/surveillance scenarios.
        
        TCS = mean(IoU(frame_t, frame_{t+1}))
        
        Range: [0, 1]
        - 1.0 = perfectly stable across frames
        - 0.5 = moderate stability
        - 0.0 = highly unstable
        
        Higher is better!
        
        Args:
            detections_sequence: List of detections per frame
                                Each frame is a list of dicts with 'boxes', 'labels'
        
        Returns:
            TCS value
        """
        if len(detections_sequence) < 2:
            return 1.0
        
        tcs_scores = []
        
        for t in range(len(detections_sequence) - 1):
            curr_frame_dets = detections_sequence[t]
            next_frame_dets = detections_sequence[t + 1]
            
            # Aggregate boxes from all images in current frame
            curr_boxes = torch.cat([d['boxes'] for d in curr_frame_dets if len(d['boxes']) > 0])
            next_boxes = torch.cat([d['boxes'] for d in next_frame_dets if len(d['boxes']) > 0])
            
            if len(curr_boxes) == 0 or len(next_boxes) == 0:
                continue
            
            # Compute IoU matrix
            iou_matrix = box_iou(curr_boxes, next_boxes)
            
            # Match boxes: for each box in t, find best match in t+1
            best_ious, _ = iou_matrix.max(dim=1)
            frame_tcs = best_ious.mean().item()
            tcs_scores.append(frame_tcs)
        
        if len(tcs_scores) == 0:
            return 0.0
        
        tcs = np.mean(tcs_scores)
        return tcs


# ============================================================================
# COMPREHENSIVE EVALUATION REPORT
# ============================================================================

class DGEvaluationReport:
    """
    Generate comprehensive domain generalization evaluation report
    """
    
    def __init__(self):
        self.metrics = DomainGeneralizationMetrics()
    
    def generate_report(self, 
                       source_perf: Dict[str, float],
                       target_perf: Dict[str, float],
                       baseline_source: Dict[str, float] = None,
                       baseline_target: Dict[str, float] = None) -> Dict:
        """
        Generate comprehensive DG evaluation report
        
        Args:
            source_perf: Your method's source domain performance
            target_perf: Your method's target domain performance
            baseline_source: Baseline method's source performance
            baseline_target: Baseline method's target performance
        
        Returns:
            Dictionary with all metrics and comparisons
        """
        report = {}
        
        # 1. DG-GAP
        report['dg_gap'] = self.metrics.compute_dg_gap(
            source_perf['mAP'], target_perf['mAP']
        )
        
        if baseline_source and baseline_target:
            baseline_gap = self.metrics.compute_dg_gap(
                baseline_source['mAP'], baseline_target['mAP']
            )
            report['baseline_dg_gap'] = baseline_gap
            report['dg_gap_improvement'] = self.metrics.compute_relative_dg_improvement(
                baseline_gap, report['dg_gap']
            )
        
        # 2. SSDG
        if all(k in source_perf for k in ['AP_S', 'AP_M', 'AP_L']):
            report['ssdg'] = self.metrics.compute_scale_sensitive_domain_gap(
                source_perf, target_perf
            )
        
        # 3. CDCI (requires multiple domain results)
        # You can add this when testing on multiple target domains
        
        return report
    
    def print_report(self, report: Dict):
        """
        Pretty-print evaluation report
        """
        print("\n" + "="*70)
        print("DOMAIN GENERALIZATION EVALUATION REPORT")
        print("="*70)
        
        print(f"\n📊 Domain Generalization Gap (DG-GAP)")
        print(f"   Your method: {report['dg_gap']:.3f} ({report['dg_gap']*100:.1f}% performance drop)")
        
        if 'baseline_dg_gap' in report:
            print(f"   Baseline:    {report['baseline_dg_gap']:.3f} ({report['baseline_dg_gap']*100:.1f}% performance drop)")
            print(f"   ✅ Improvement: {report['dg_gap_improvement']:.1f}% reduction in gap")
        
        if 'ssdg' in report:
            print(f"\n📏 Scale-Sensitive Domain Gap (SSDG)")
            print(f"   SSDG: {report['ssdg']:.2f} (lower is better)")
        
        print("\n" + "="*70)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """
    Example of how to use novel metrics
    """
    metrics = DomainGeneralizationMetrics()
    
    # Example 1: DG-GAP
    print("="*70)
    print("Example 1: Domain Generalization Gap (DG-GAP)")
    print("="*70)
    
    source_map = 55.3  # Source domain (VisDrone)
    target_map_baseline = 34.2  # Baseline on target (IndraEye)
    target_map_yours = 46.5  # Your method on target
    
    dg_gap_baseline = metrics.compute_dg_gap(source_map, target_map_baseline)
    dg_gap_yours = metrics.compute_dg_gap(source_map, target_map_yours)
    
    print(f"Baseline DG-GAP: {dg_gap_baseline:.3f} ({dg_gap_baseline*100:.1f}% drop)")
    print(f"Your method DG-GAP: {dg_gap_yours:.3f} ({dg_gap_yours*100:.1f}% drop)")
    print(f"Improvement: {(dg_gap_baseline - dg_gap_yours)*100:.1f}% absolute reduction")
    
    # Example 2: CDCI
    print("\n" + "="*70)
    print("Example 2: Cross-Domain Consistency Index (CDCI)")
    print("="*70)
    
    perfs_baseline = [34.2, 28.9, 41.8, 35.4]
    perfs_yours = [46.5, 37.6, 48.2, 40.1]
    
    cdci_baseline = metrics.compute_cross_domain_consistency_index(perfs_baseline)
    cdci_yours = metrics.compute_cross_domain_consistency_index(perfs_yours)
    
    print(f"Baseline CDCI: {cdci_baseline:.3f}")
    print(f"Your method CDCI: {cdci_yours:.3f}")
    print(f"Your method is more consistent!" if cdci_yours > cdci_baseline else "Baseline is more consistent")
    
    # Example 3: SSDG
    print("\n" + "="*70)
    print("Example 3: Scale-Sensitive Domain Gap (SSDG)")
    print("="*70)
    
    source_aps = {'AP_S': 18.4, 'AP_M': 42.7, 'AP_L': 51.2}
    target_aps = {'AP_S': 24.6, 'AP_M': 54.3, 'AP_L': 63.8}
    
    ssdg = metrics.compute_scale_sensitive_domain_gap(source_aps, target_aps)
    print(f"SSDG: {ssdg:.2f}")
    print("Lower SSDG = better scale-invariant generalization")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("Novel Domain Generalization Metrics")
    print("For ICPR 2026 Submission")
    print()
    
    example_usage()
