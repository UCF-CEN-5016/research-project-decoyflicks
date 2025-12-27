import pandas as pd


def build_dataframe(column_a, column_b):
    """Create a DataFrame with columns 'A' and 'B' from provided sequences."""
    return pd.DataFrame({'A': column_a, 'B': column_b})


def combine_dataframes(df_first, df_second):
    """Concatenate two DataFrames vertically."""
    return pd.concat([df_first, df_second])


# Minimal example to reproduce the bug fix
training_df = build_dataframe([1, 2], ['a', 'b'])
testing_df = build_dataframe([3, 4], ['c', 'd'])

combined_df = combine_dataframes(training_df, testing_df)