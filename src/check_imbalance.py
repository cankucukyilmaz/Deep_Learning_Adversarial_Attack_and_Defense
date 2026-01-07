import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Update this path to where your processed 0/1 CSV is located
CSV_PATH = "data/processed/attr_celeba.csv"
OUTPUT_DIR = "plots"
OUTPUT_FILE = "class_imbalance.png"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def plot_imbalance(csv_path, output_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. logical check: Drop non-label columns
    # We assume 'filename' or 'partition' might be there. We only want the 40 attributes.
    # Select only numeric columns (0s and 1s)
    label_df = df.select_dtypes(include=['number'])
    
    # If partition column exists (0,1,2), drop it from calculation
    if 'partition' in label_df.columns:
        label_df = label_df.drop(columns=['partition'])
        
    # 2. Calculate Statistics
    # Mean of a 0/1 binary vector is exactly the proportion of positives
    # Count of 1s / Total Count
    positive_ratio = label_df.mean().sort_values(ascending=True) # Sort for cleaner plot
    
    # 3. Plotting
    plt.figure(figsize=(10, 12)) # Tall figure for 40 labels
    
    # Create horizontal bar plot
    # We use a threshold line at 0.5 (perfect balance)
    bars = plt.barh(positive_ratio.index, positive_ratio.values, color='skyblue')
    
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=1, label='Perfect Balance (50%)')
    
    # Aesthetics
    plt.title(f"Class Imbalance across {len(positive_ratio)} Attributes", fontsize=15)
    plt.xlabel("Proportion of Positive Samples (1s)", fontsize=12)
    plt.xlim(0, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Annotate bars with specific percentages for clarity
    for bar in bars:
        width = bar.get_width()
        label_text = f'{width:.1%}' 
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, label_text, 
                 va='center', fontsize=9, color='black')

    plt.tight_layout()
    
    # 4. Save
    save_loc = os.path.join(output_path, OUTPUT_FILE)
    plt.savefig(save_loc)
    print(f"Plot saved to: {save_loc}")
    plt.show()

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    plot_imbalance(CSV_PATH, OUTPUT_DIR)