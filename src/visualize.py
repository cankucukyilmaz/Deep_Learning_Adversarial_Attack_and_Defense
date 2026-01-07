import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import sys

# --- Configuration ---
DATA_ROOT = "data/raw"  # Folder containing actual .jpg files
PARTITION_FILE = "data/external/list_eval_partition.txt"
LABEL_FILE = "data/external/list_attr_celeba.txt" # Adjust this filename if yours differs
TARGET_LABEL = "Black_Hair" # Replace with the specific column name you are analyzing (0 or 1)

def load_and_merge_metadata():
    """
    Loads partition info and labels, then merges them into a single DataFrame.
    """
    # 1. Load Partitions
    if not os.path.exists(PARTITION_FILE):
        sys.exit(f"Error: Partition file not found at {PARTITION_FILE}")
        
    df_part = pd.read_csv(PARTITION_FILE, delim_whitespace=True, header=None, names=['filename', 'partition'])

    # 2. Load Labels
    # Sci-check: We assume headers exist in the label file (common in CelebA). 
    # If your file has no headers, remove 'header=0' and manually set names.
    if not os.path.exists(LABEL_FILE):
        sys.exit(f"Error: Label file not found at {LABEL_FILE}")

    # Note: skiprows might be needed if there are metadata lines (CelebA has 1 metadata line usually)
    df_attr = pd.read_csv(LABEL_FILE, sep=r'\s+', header=0, skiprows=1) 
    
    # In some formats, the filename is the index. Let's ensure 'filename' is a column to merge on.
    if 'filename' not in df_attr.columns and df_attr.index.name != 'filename':
        df_attr = df_attr.reset_index()
        df_attr.rename(columns={'index': 'filename'}, inplace=True)

    # 3. Merge (Inner Join)
    # We only want images that exist in both lists.
    df_merged = pd.merge(df_part, df_attr, on='filename')
    
    return df_merged

def plot_verification(df, label_col):
    """
    Plots a grid of images with their assigned labels.
    """
    print(f"Sampling 10 images to verify label '{label_col}'...")
    
    # We take 5 positives (1) and 5 negatives (-1 or 0) to ensure we see both classes
    # Adjust logic if your labels are strictly 0/1. CelebA uses -1/1 usually.
    pos_samples = df[df[label_col] == 1].sample(5, replace=True)
    neg_samples = df[df[label_col] != 1].sample(5, replace=True)
    samples = pd.concat([pos_samples, neg_samples])

    plt.figure(figsize=(20, 8))
    
    for idx, (i, row) in enumerate(samples.iterrows()):
        img_path = os.path.join(DATA_ROOT, row['filename'])
        
        if not os.path.exists(img_path):
            print(f"MISSING: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"CORRUPT: {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(2, 5, idx + 1)
        plt.imshow(img)
        
        # Color code the title: Green for 1, Red for 0/-1
        color = 'green' if row[label_col] == 1 else 'red'
        plt.title(f"{row['filename']}\nLabel: {row[label_col]}", color=color, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Execute pipeline
    df_data = load_and_merge_metadata()
    
    # Filter for Train set just to test (partition 0)
    train_df = df_data[df_data['partition'] == 0]
    
    print(f"Training set size: {len(train_df)}")
    
    try:
        plot_verification(train_df, TARGET_LABEL)
    except KeyError as e:
        print(f"Error: The label column '{TARGET_LABEL}' was not found in your file.")
        print(f"Available columns: {list(train_df.columns)}")