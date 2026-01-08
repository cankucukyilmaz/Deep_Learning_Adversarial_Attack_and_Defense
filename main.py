import argparse
import sys
import os

# Import the engine function
from src.engine import run_baseline_evaluation

# --- Global Configuration ---
# Update these paths if your folder structure differs!
CONFIG = {
    'data_root': "data/raw",  
    'partition': "data/external/list_eval_partition.txt",
    'attr_csv': "data/processed/attr_celeba.csv",
    'report_path': "reports/model_comparison.csv",
    'batch_size': 64
}

def check_paths():
    """Safety check to ensure files exist before crashing deep in the code."""
    required = [CONFIG['data_root'], CONFIG['partition'], CONFIG['attr_csv']]
    for p in required:
        if not os.path.exists(p):
            print(f"ERROR: Could not find required path: {p}")
            print("Please check your folder structure or CONFIG in main.py")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="CelebA Multi-Label Project Pipeline")
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['baseline', 'train', 'attack'], 
        required=True,
        help="Action to perform: 'baseline' (test untrained), 'train' (finetune), or 'attack' (adversarial)."
    )
    
    args = parser.parse_args()
    
    # Pre-flight check
    check_paths()
    
    if args.mode == 'baseline':
        run_baseline_evaluation(CONFIG)
        
    elif args.mode == 'train':
        print("Training mode is coming next...")
        # run_training(CONFIG)
        
    elif args.mode == 'attack':
        print("Attack mode is coming later...")

if __name__ == "__main__":
    main()