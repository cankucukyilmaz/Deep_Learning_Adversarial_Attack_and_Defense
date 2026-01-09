import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score

# --- Internal Module Imports ---
from src.dataset import CelebADataset
from src.model import get_celeba_model
from src.metrics import calculate_per_attribute_metrics
from src.attack import BIMAttack
from src.defense import JPEGDefense

# --- Helper Functions ---

def get_dataloader(config, split='test'):
    """
    Creates a DataLoader with standard ImageNet normalization.
    """
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
    
    # Use shuffle=True only for training
    should_shuffle = (split == 'train')
    
    return DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=should_shuffle, 
        num_workers=0, # Increase to 2 or 4 on Linux/Mac
        pin_memory=True if torch.cuda.is_available() else False
    )

# --- 1. Baseline Evaluation Engine ---

def run_baseline_evaluation(config):
    print("\n" + "="*40)
    print("   ENGINE START: Baseline Evaluation")
    print("="*40)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load Data
    test_loader = get_dataloader(config, split='test')
    
    # 2. Load Model (Pretrained Backbone, Random Head)
    print("Loading ResNet18 (Untrained Head)...")
    model = get_celeba_model(num_classes=40, pretrained=True).to(device)
    model.eval()

    # 3. Inference
    all_probs = []
    all_targets = []
    
    print("Running Inference...")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())
            
    # 4. Metrics
    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_targets)
    
    df_metrics = calculate_per_attribute_metrics(
        y_true, y_probs, 
        test_loader.dataset.label_cols, 
        stage_prefix="Base"
    )
    
    # 5. Save Report
    os.makedirs(os.path.dirname(config['report_path']), exist_ok=True)
    df_metrics.to_csv(config['report_path'])
    
    print(f"Baseline F1 Score (Mean): {df_metrics['Base_F1'].mean():.4f}")
    print(f"Report saved to: {config['report_path']}")


# --- 2. Training Engine ---

def run_training(config):
    print("\n" + "="*40)
    print("   ENGINE START: Training Phase")
    print("="*40)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Data
    train_loader = get_dataloader(config, split='train')
    val_loader   = get_dataloader(config, split='val')
    
    # 2. Setup Model
    print("Initializing Model...")
    model = get_celeba_model(num_classes=40, pretrained=True).to(device)
    
    # 3. Setup Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-4))
    
    num_epochs = config.get('epochs', 5)
    best_val_loss = float('inf')
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(config['model_path']), exist_ok=True)
    
    # 4. Training Loop
    for epoch in range(num_epochs):
        # --- Train Step ---
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train = train_loss / len(train_loader)
        
        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        # --- Checkpointing ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), config['model_path'])
            print(f"--> Best model saved to {config['model_path']}")
            
    print("\nTraining Complete.")


# --- 3. Attack & Defense Engine ---

def run_targeted_attack(config, target_label_name):
    print("\n" + "="*40)
    print(f"   ENGINE START: Targeted Attack on '{target_label_name}'")
    print("="*40)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Test Data
    test_loader = get_dataloader(config, split='test')
    attr_names = test_loader.dataset.label_cols
    
    # Verify label existence
    if target_label_name not in attr_names:
        print(f"ERROR: Label '{target_label_name}' not found.")
        return
    target_idx = attr_names.index(target_label_name)
    
    # 2. Load TRAINED Model
    if not os.path.exists(config['model_path']):
        print(f"ERROR: No model found at {config['model_path']}. Please run --mode train first.")
        return
        
    print(f"Loading trained model from {config['model_path']}...")
    model = get_celeba_model(num_classes=40, pretrained=False) # Architecture
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Initialize Adversaries
    attacker = BIMAttack(model, epsilon=0.05, alpha=0.01, steps=10, device=device)
    defender = JPEGDefense(quality=75, device=device)
    
    # 4. Experiment Loop
    results = {'Original': [], 'Adversarial': [], 'Defended': [], 'GroundTruth': []}
    
    print("Running Attack/Defense Pipeline...")
    for imgs, labels in tqdm(test_loader, desc="Attacking"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Get target values (Inverse of Truth)
        true_vals = labels[:, target_idx]
        target_vals = 1.0 - true_vals 
        
        # A. Original Prediction
        with torch.no_grad():
            base_logits = model(imgs)
            base_probs = torch.sigmoid(base_logits[:, target_idx])
            
        # B. Generate Attack
        # The attacker tries to force the model to output 'target_vals'
        adv_imgs = attacker(imgs, target_idx, target_vals)
        
        with torch.no_grad():
            adv_logits = model(adv_imgs)
            adv_probs = torch.sigmoid(adv_logits[:, target_idx])
            
        # C. Apply Defense
        def_imgs = defender(adv_imgs)
        
        with torch.no_grad():
            def_logits = model(def_imgs)
            def_probs = torch.sigmoid(def_logits[:, target_idx])
            
        # Store batch results
        results['Original'].extend(base_probs.cpu().numpy())
        results['Adversarial'].extend(adv_probs.cpu().numpy())
        results['Defended'].extend(def_probs.cpu().numpy())
        results['GroundTruth'].extend(true_vals.cpu().numpy())
        
    # 5. Display Results
    y_true = np.array(results['GroundTruth'])
    
    print(f"\nResults for Targeted Attack on '{target_label_name}':")
    print(f"{'State':<15} | {'Acc':<8} | {'F1 Score':<8} | {'ASR (Success)':<15}")
    print("-" * 60)
    
    for state in ['Original', 'Adversarial', 'Defended']:
        probs = np.array(results[state])
        preds = (probs > 0.5).astype(float)
        
        # Metrics
        acc = (preds == y_true).mean()
        f1 = f1_score(y_true, preds, zero_division=0)
        
        # Attack Success Rate (ASR): Percentage of samples that are WRONG (flipped)
        # Note: For Original/Defended, we want ASR low. For Adversarial, we want ASR high.
        asr = (preds != y_true).mean()
        
        print(f"{state:<15} | {acc:.4f}   | {f1:.4f}   | {asr:.4f}")
    
    print("-" * 60)