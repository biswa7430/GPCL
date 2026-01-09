"""Model builder for object detection models."""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(model_name, num_classes, pretrained_backbone=True, 
                trainable_backbone_layers=None, rpn_score_thresh=None):
    """
    Build an object detection model.
    
    Args:
        model_name (str): Name of the model architecture
        num_classes (int): Number of classes (including background)
        pretrained_backbone (bool): Whether to use pretrained backbone
        trainable_backbone_layers (int): Number of trainable backbone layers
        rpn_score_thresh (float): RPN score threshold for Faster R-CNN
    
    Returns:
        torch.nn.Module: The detection model
    """
    if model_name == "fasterrcnn_resnet50_fpn":
        # Load Faster R-CNN with ResNet50 backbone
        if pretrained_backbone:
            weights_backbone = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights_backbone = None
        
        # Build kwargs for model
        model_kwargs = {
            'weights': None,
            'weights_backbone': weights_backbone,
            'num_classes': num_classes,
        }
        
        # Only add optional parameters if they're not None
        if trainable_backbone_layers is not None:
            model_kwargs['trainable_backbone_layers'] = trainable_backbone_layers
        if rpn_score_thresh is not None:
            model_kwargs['rpn_score_thresh'] = rpn_score_thresh
        
        # Build model with custom number of classes
        model = fasterrcnn_resnet50_fpn(**model_kwargs)
        
    elif model_name == "fasterrcnn_mobilenet_v3_large_fpn":
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        
        if pretrained_backbone:
            weights_backbone = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        else:
            weights_backbone = None
            
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers
        )
        
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model


def get_model_instance(checkpoint_path, num_classes, device='cuda', config=None):
    """
    Load a trained model from checkpoint.
    
    Handles both regular and SHM-wrapped models automatically.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        num_classes (int): Number of classes
        device (str): Device to load model on
        config (dict): Optional config dict for SHM settings
    
    Returns:
        tuple: (model, checkpoint_data)
            - model: Loaded model in eval mode (unwrapped from SHM)
            - checkpoint_data: Full checkpoint dictionary
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get state dict
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Check if this is an SHM-wrapped checkpoint
    is_shm_wrapped = any(k.startswith('base_model.') for k in state_dict.keys())
    
    if is_shm_wrapped:
        print("Detected SHM-wrapped checkpoint. Extracting base model...")
        
        # Extract base_model weights by removing 'base_model.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('base_model.'):
                # Remove 'base_model.' prefix
                new_key = k[11:]  # len('base_model.') = 11
                
                # Also handle RoI wrapper: 'roi_heads.original_roi_heads.' -> 'roi_heads.'
                if new_key.startswith('roi_heads.original_roi_heads.'):
                    new_key = 'roi_heads.' + new_key[29:]  # len('roi_heads.original_roi_heads.') = 29
                
                new_state_dict[new_key] = v
            elif k.startswith('module.base_model.'):
                # Handle DDP + SHM: 'module.base_model.' -> ''
                new_key = k[18:]  # len('module.base_model.') = 18
                
                # Also handle RoI wrapper
                if new_key.startswith('roi_heads.original_roi_heads.'):
                    new_key = 'roi_heads.' + new_key[29:]
                
                new_state_dict[new_key] = v
            # Skip other keys (image_mixer, feature_shm, etc.)
        
        state_dict = new_state_dict
    else:
        # Handle regular DDP wrapped models
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # Create model architecture
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, checkpoint
