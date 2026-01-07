import pandas as pd
from src.transform_attr import process_celeba_attributes

def test_process_celeba_attributes_converts_negatives(tmp_path):
    # 1. Setup: Create a dummy input file in a temporary folder
    d_input = tmp_path / "list_attr_celeba.txt"
    d_output = tmp_path / "attr_celeba.csv"

    # Mock content mimicking the real CelebA file (Count line, Header, Data)
    # Note: 000001.jpg has -1 for Attribute_A, which should become 0
    content = """2
Attribute_A Attribute_B
000001.jpg -1  1
000002.jpg  1 -1
"""
    d_input.write_text(content)

    # 2. Action: Run the function using these temporary paths
    process_celeba_attributes(input_path=d_input, output_path=d_output)

    # 3. Assertion: Check if output exists and values are correct
    assert d_output.exists()
    
    df = pd.read_csv(d_output)
    
    # Check if -1 was converted to 0 for the first image
    row_1 = df.loc[df['image_id'] == '000001.jpg'].iloc[0]
    assert row_1['Attribute_A'] == 0, "Expected -1 to be converted to 0"
    assert row_1['Attribute_B'] == 1, "Expected 1 to remain 1"

    # Check the second image
    row_2 = df.loc[df['image_id'] == '000002.jpg'].iloc[0]
    assert row_2['Attribute_A'] == 1
    assert row_2['Attribute_B'] == 0