import pandas as pd

def load_data(path: str):
    """
    Loads dataset from given path.
    """
    data = pd.read_csv(path)
    return data