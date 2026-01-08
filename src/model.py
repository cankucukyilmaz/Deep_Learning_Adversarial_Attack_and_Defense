import torch
import torch.nn as nn
from torchvision import models

def get_celeba_model(num_classes=40, pretrained=True):
    """
    Returns a ResNet18 model configured for CelebA.
    
    Args:
        num_classes (int): Number of target labels (default 40).
        pretrained (bool): If True, loads ImageNet weights for the backbone.
                           If False, initializes with random weights.
    """
    # 1. Load the Backbone (The "Eye")
    if pretrained:
        # standard way to load best available weights in modern torchvision
        weights = models.ResNet18_Weights.DEFAULT 
    else:
        weights = None
        
    model = models.resnet18(weights=weights)
    
    # 2. Modify the Head (The "Brain")
    # Original: Linear(512 -> 1000)
    # New:      Linear(512 -> 40)
    
    in_features = model.fc.in_features
    
    # We replace the final fully connected layer.
    # Note: The new layer is automatically initialized with random weights.
    model.fc = nn.Linear(in_features, num_classes)
    
    return model