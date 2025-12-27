import pandas as pd

def create_dataframe(data: dict) -> pd.DataFrame:
    """Create a pandas DataFrame from a dictionary."""
    return pd.DataFrame(data)

def combine_vertical(df_top: pd.DataFrame, df_bottom: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two DataFrames vertically."""
    return pd.concat([df_top, df_bottom])

train_set = create_dataframe({'A': [1, 2], 'B': [3, 4]})
test_set = create_dataframe({'A': [5, 6], 'B': [7, 8]})
combined_set = combine_vertical(train_set, test_set)