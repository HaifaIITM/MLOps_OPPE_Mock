import pandas as pd

def test_data_exists():
    df = pd.read_csv("data/data_v1.csv")
    assert df.shape[0] > 0