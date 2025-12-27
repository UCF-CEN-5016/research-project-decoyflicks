import pandas as pd
from typing import List


def create_sample_dataframe(values: List[int], column_name: str = "a") -> pd.DataFrame:
    """Create a simple DataFrame from a list of integers with a single column."""
    return pd.DataFrame({column_name: values})


def combine_dataframes(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of DataFrames along the row axis."""
    return pd.concat(dataframes)


def main() -> None:
    train_data = create_sample_dataframe([1, 2])
    test_data = create_sample_dataframe([3, 4])

    combined_df = combine_dataframes([train_data, test_data])

    print("Combined DataFrame:")
    print(combined_df)


if __name__ == "__main__":
    main()