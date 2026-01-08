import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def calculate_per_attribute_metrics(y_true, y_probs, attribute_names, stage_prefix="Base"):
    """
    Computes metrics for each of the 40 attributes individually.
    
    Args:
        y_true (np.array): Ground truth (N, 40) with 0s and 1s.
        y_probs (np.array): Model output probabilities (N, 40) range [0, 1].
        attribute_names (list): List of strings for the index.
        stage_prefix (str): Prefix for columns (e.g., 'Base', 'Attack', 'Def').
        
    Returns:
        pd.DataFrame: Rows are attributes, columns are metrics.
    """
    # Convert probabilities to binary predictions based on standard threshold
    y_pred = (y_probs > 0.5).astype(int)
    
    metrics_list = []
    
    for i, name in enumerate(attribute_names):
        # Slice the i-th column for all samples
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        prob_i = y_probs[:, i]
        
        # Handle edge case: If a batch has only one class, AUC crashes
        try:
            auc = roc_auc_score(true_i, prob_i)
        except ValueError:
            auc = 0.5 # Default for undefined
            
        metrics_list.append({
            'Attribute': name,
            f'{stage_prefix}_Acc': accuracy_score(true_i, pred_i),
            f'{stage_prefix}_F1': f1_score(true_i, pred_i, zero_division=0),
            f'{stage_prefix}_Prec': precision_score(true_i, pred_i, zero_division=0),
            f'{stage_prefix}_Rec': recall_score(true_i, pred_i, zero_division=0),
            f'{stage_prefix}_AUC': auc
        })
        
    df = pd.DataFrame(metrics_list)
    df.set_index('Attribute', inplace=True)
    return df