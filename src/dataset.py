import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class CelebADataset(Dataset):
    def __init__(self, partition_file, attr_csv, data_root, split='test', transform=None):
        """
        Args:
            partition_file (str): Path to list_eval_partition.txt
            attr_csv (str): Path to your processed 0/1 attributes CSV.
            data_root (str): Path to the folder containing .jpg images.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): PyTorch transforms.
        """
        self.data_root = data_root
        self.transform = transform
        
        # 1. Load Partitions (Format: filename.jpg partition_id)
        # partition_id: 0=train, 1=val, 2=test
        df_part = pd.read_csv(partition_file, delim_whitespace=True, header=None, names=['filename', 'partition'])
        
        split_map = {'train': 0, 'val': 1, 'test': 2}
        target_split = split_map.get(split.lower())
        
        if target_split is None:
            raise ValueError(f"Invalid split: {split}. Choose train, val, or test.")
            
        # 2. Load Attributes (The 0/1 CSV)
        df_attr = pd.read_csv(attr_csv)
        
        # Ensure we can merge on filename
        if 'filename' not in df_attr.columns:
            # Check if filename is in the index
            if df_attr.index.name == 'filename' or df_attr.index.name == 'index':
                df_attr = df_attr.reset_index()
                # If it was named 'index', rename to 'filename'
                if 'filename' not in df_attr.columns:
                     df_attr.rename(columns={'index': 'filename'}, inplace=True)
            else:
                 # Fallback: Assume first column is filename
                 df_attr.rename(columns={df_attr.columns[0]: 'filename'}, inplace=True)

        # 3. Merge: Keep only images present in THIS split
        # Inner join ensures we only get rows that exist in both files
        self.df = pd.merge(df_part[df_part['partition'] == target_split], df_attr, on='filename')
        
        # 4. Identify Label Columns (Everything that isn't metadata)
        self.label_cols = [c for c in self.df.columns if c not in ['filename', 'partition']]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load Image
        img_name = row['filename']
        img_path = os.path.join(self.data_root, img_name)
        
        # Open and convert to RGB (strips alpha channel if present)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Get labels as Float Tensor (needed for BCE Loss later)
        labels = torch.tensor(row[self.label_cols].values.astype('float32'))
        
        return image, labels