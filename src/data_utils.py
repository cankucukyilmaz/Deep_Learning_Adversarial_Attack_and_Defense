import pandas as pd
import os

def load_partition_data(partition_path: str) -> pd.DataFrame:
    """
    Reads the partition file and returns a DataFrame.
    Assumes space-separated values: 'filename.jpg partition_id'
    """
    if not os.path.exists(partition_path):
        raise FileNotFoundError(f"{partition_path} does not exist.")

    # logical check: we expect a specific schema
    df = pd.read_csv(partition_path, sep=r'\s+', header=None, names=['filename', 'partition'])
    
    return df

def get_data_splits(df: pd.DataFrame):
    """
    Splits the DataFrame into train, val, test based on partition ID.
    0: Train, 1: Validation, 2: Test
    """
    train = df[df['partition'] == 0].copy()
    val = df[df['partition'] == 1].copy()
    test = df[df['partition'] == 2].copy()
    
    return train, val, test