import pytest
import pandas as pd
from io import StringIO
from src.data_utils import load_partition_data, get_data_splits

@pytest.fixture
def mock_partition_file(tmp_path):
    # Create a temporary dummy partition file
    content = """image_01.jpg 0
image_02.jpg 0
image_03.jpg 1
image_04.jpg 2"""
    p = tmp_path / "list_eval_partition.txt"
    p.write_text(content)
    return str(p)

def test_load_partition_data(mock_partition_file):
    df = load_partition_data(mock_partition_file)
    assert len(df) == 4
    assert df.iloc[0]['filename'] == 'image_01.jpg'

def test_get_data_splits(mock_partition_file):
    df = load_partition_data(mock_partition_file)
    train, val, test = get_data_splits(df)
    
    # Assert logical correctness of the split
    assert len(train) == 2
    assert len(val) == 1
    assert len(test) == 1
    
    # Verify strict separation (no leakage)
    assert set(train['filename']).isdisjoint(set(val['filename']))