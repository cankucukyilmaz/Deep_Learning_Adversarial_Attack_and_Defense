import pandas as pd
from pathlib import Path

def process_celeba_attributes(input_path=None, output_path=None):
    """
    Reads CelebA attributes, converts -1 to 0, and saves as CSV.
    If paths are not provided, uses the default project structure.
    """
    # Define default paths relative to this file if not provided
    if input_path is None or output_path is None:
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
        
        if input_path is None:
            input_path = project_root / 'data' / 'external' / 'list_attr_celeba.txt'
        if output_path is None:
            output_path = project_root / 'data' / 'processed' / 'attr_celeba.csv'

    # Ensure we are working with Path objects
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw attributes from: {input_path}")

    # Read the text file
    try:
        df = pd.read_csv(input_path, skiprows=1, sep=r'\s+')
    except FileNotFoundError:
        print(f"Error: Could not find file at {input_path}")
        return

    # Fix indexing: The text file often treats the filename as the index
    df.index.name = 'image_id'
    df = df.reset_index()

    # Replace -1 with 0
    df = df.replace(-1, 0)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Processed attributes saved to: {output_path}")

if __name__ == "__main__":
    process_celeba_attributes()