import pandas as pd


def load_dataset(path):
    """
    Load the mall customer dataset
    """
    df = pd.read_csv(path)

    print("Dataset loaded successfully")
    print(df.head())

    return df