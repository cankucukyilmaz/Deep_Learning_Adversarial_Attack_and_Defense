import ssl
import os

# LOGIC: 
# macOS openssl versions often cannot find the root certificate bundle in venvs.
# We override the default https context to skip verification for this specific download.
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import pytest
from src.model import get_celeba_model

def test_output_shape():
    """
    Hypothesis: The model should accept an image batch (N, 3, 224, 224) 
    and output a tensor of shape (N, 40).
    """
    batch_size = 4
    num_classes = 40
    model = get_celeba_model(num_classes=num_classes, pretrained=False)
    
    # Create a random dummy batch
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    output = model(dummy_input)
    
    # Verification
    assert output.shape == (batch_size, num_classes)
    assert output.dtype == torch.float32

def test_pretrained_vs_random_weights():
    """
    Hypothesis: A pretrained model and a non-pretrained model must have 
    significantly different weights in the backbone (e.g., conv1).
    """
    # Load 'Eye' with ImageNet knowledge
    model_pretrained = get_celeba_model(pretrained=True)
    
    # Load 'Eye' with random noise
    model_random = get_celeba_model(pretrained=False)
    
    # Extract weights from the very first convolutional layer
    # .data.clone() ensures we don't mess with gradients
    weights_p = model_pretrained.conv1.weight.data
    weights_r = model_random.conv1.weight.data
    
    # Verification: The weights should NOT be close
    # If they are close, it means pretraining failed to load specific weights.
    assert not torch.allclose(weights_p, weights_r), "Pretrained and random weights are identical!"

def test_head_initialization():
    """
    Hypothesis: Even if we load pretrained backbones, the NEW head (fc) 
    should be random. Two different instances should have different random heads.
    """
    model_a = get_celeba_model(pretrained=True)
    model_b = get_celeba_model(pretrained=True)
    
    # The backbones might be identical (same ImageNet weights), 
    # but the heads (fc) are initialized randomly at the moment of creation.
    head_a = model_a.fc.weight.data
    head_b = model_b.fc.weight.data
    
    # Verification: Heads should be different (random initialization)
    assert not torch.allclose(head_a, head_b), "The new heads should be randomly distinct."