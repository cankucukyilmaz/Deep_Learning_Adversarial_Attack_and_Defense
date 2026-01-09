import argparse
import sys
import os

# Internal Imports
# We import functions conditionally inside the blocks or at top-level
# Importing here ensures we catch syntax errors in engine.py early
from src.engine import (
    run_baseline_evaluation,
    run_training,
    run_targeted_attack
)

# --- Global Configuration ---
# This dictionary controls the paths for the entire project.
# Ensure your folder structure matches these paths.
CONFIG = {
    # Data Paths
    'data_root': "data/raw",                  # Folder containing .jpg files
    'partition': "data/external/list_eval_partition.txt",
    'attr_csv': "data/processed/attr_celeba.csv",
    
    # Output Paths
    'report_path': "reports/model_comparison.csv",
    'model_path': "models/best_model.pth",       # Where trained model is saved/loaded
    
    # Hyperparameters
    'batch_size': 64,
    'epochs': 5,                                 # Number of training epochs
    'learning_rate': 1e-4
}

def check_paths():
    """
    Pre-flight check: verification that essential data files exist 
    before starting expensive operations.
    """
    required_files = [
        ('Partition File', CONFIG['partition']),
        ('Attribute CSV', CONFIG['attr_csv'])
    ]
    
    required_dirs = [
        ('Data Root', CONFIG['data_root'])
    ]
    
    missing = []
    
    for name, path in required_files:
        if not os.path.isfile(path):
            missing.append(f"{name} (File not found: {path})")
            
    for name, path in required_dirs:
        if not os.path.isdir(path):
            missing.append(f"{name} (Directory not found: {path})")
            
    if missing:
        print("\n" + "!"*50)
        print("CRITICAL ERROR: Missing Resources")
        print("!"*50)
        for m in missing:
            print(f"- {m}")
        print("\nPlease check your folder structure or update CONFIG in main.py")
        sys.exit(1)

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="CelebA Multi-Label Adversarial Robustness Project",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # The 'Mode' Switch
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['baseline', 'train', 'attack'], 
        required=True,
        help=("Select the operation to perform:\n"
              "  baseline : Evaluate random/untrained model performance.\n"
              "  train    : Train (fine-tune) ResNet18 on the dataset.\n"
              "  attack   : Run Targeted BIM Attack + JPEG Defense on a specific label.")
    )
    
    # Optional Argument for Attack Mode
    parser.add_argument(
        '--target_label', 
        type=str, 
        default=None,
        help="[Attack Mode Only] The specific attribute to attack (e.g., 'Smiling', 'Eyeglasses')."
    )
    
    args = parser.parse_args()
    
    # 2. Run Safety Checks
    check_paths()
    
    # 3. Execution Logic
    try:
        if args.mode == 'baseline':
            # Run the initial sanity check
            run_baseline_evaluation(CONFIG)
            
        elif args.mode == 'train':
            # Train the model to get high accuracy
            run_training(CONFIG)
            
        elif args.mode == 'attack':
            # Run the Adversarial Experiment
            if not args.target_label:
                print("\nError: You must provide --target_label when using attack mode.")
                print("Example: python main.py --mode attack --target_label Smiling")
                sys.exit(1)
                
            run_targeted_attack(CONFIG, args.target_label)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # In a real app, we might log the full traceback here
        raise e

if __name__ == "__main__":
    main()