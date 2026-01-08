import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# Internal Imports
from src.dataset import CelebADataset
from src.model import get_celeba_model
from src.metrics import calculate_per_attribute_metrics

def get_dataloader(config, split='test'):
    """Standardizes data loading for all engine functions."""
    # ImageNet Normalization (Required for Pretrained ResNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CelebADataset(
        partition_file=config['partition'], 
        attr_csv=config['attr_csv'], 
        data_root=config['data_root'], 
        split=split, 
        transform=transform
    )
    
    return DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=(split == 'train'), 
        num_workers=0, # Set to 2 or 4 if on Linux/Mac for speed
        pin_memory=True if torch.cuda.is_available() else False
    )

def run_baseline_evaluation(config):
    print("\n" + "="*40)
    print("   ENGINE START: Baseline Evaluation")
    print("="*40)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load Data
    test_loader = get_dataloader(config, split='test')
    print(f"Data Loaded: {len(test_loader.dataset)} test images.")

    # 2. Load Model
    # We load Pretrained=True to get the good backbone
    # But the head (fc) will be random (untrained)
    print("Loading ResNet18 (Pretrained Backbone, Random Head)...")
    model = get_celeba_model(num_classes=40, pretrained=True).to(device)
    model.eval() # Freeze BatchNorm/Dropout

    # 3. Inference Loop
    all_probs = []
    all_targets = []
    
    print("Running Inference...")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            
            # Forward Pass
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            
            # Move to CPU to save GPU memory
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())
            
    # 4. Compute Metrics
    print("Calculating scientific metrics...")
    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_targets)
    
    # Calculate F1, AUC, etc. per attribute
    df_metrics = calculate_per_attribute_metrics(
        y_true, 
        y_probs, 
        attribute_names=test_loader.dataset.label_cols, 
        stage_prefix="Base"
    )
    
    # 5. Save Report
    os.makedirs(os.path.dirname(config['report_path']), exist_ok=True)
    df_metrics.to_csv(config['report_path'])
    
    print("\n" + "-"*40)
    print(f"SUCCESS: Report saved to {config['report_path']}")
    print(f"Mean F1 Score (Baseline): {df_metrics['Base_F1'].mean():.4f}")
    print("-"*40 + "\n")