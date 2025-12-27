import pandas as pd
from pandas import to_datetime
from typing import List


def create_example_dataframe() -> pd.DataFrame:
    data = {
        'date': ['01.01.2009', '02.01.2009', '03.01.2009'],
        'temp': [5, 6, 7],
        'rain': [0.5, 0.4, 0.3]
    }
    return pd.DataFrame(data)


def convert_date_column_to_datetime(df: pd.DataFrame, column: str = 'date') -> pd.DataFrame:
    df[column] = df[column].apply(lambda x: to_datetime(x))
    return df


def attempt_correlation(df: pd.DataFrame, error_prefix: str = "Error:") -> None:
    try:
        print(df.corr())
    except ValueError as e:
        print(f"{error_prefix} {e}")


def load_dataframe_with_parsed_dates(path: str = 'your_data.csv', date_columns: List[str] = None) -> pd.DataFrame:
    if date_columns is None:
        date_columns = ['date']
    return pd.read_csv(path, parse_dates=date_columns)


if __name__ == '__main__':
    # Example DataFrame with string dates causing correlation error
    example_df = create_example_dataframe()

    # Attempt to compute correlation (this will fail)
    attempt_correlation(example_df, error_prefix="Error:")

    # Fixing the date column by converting it to datetime
    example_df = convert_date_column_to_datetime(example_df, column='date')

    # Now compute correlation without error
    print("\nAfter fixing dates:")
    attempt_correlation(example_df, error_prefix="Error (if any after fix):")

    # Assuming 'date' should be treated as a datetime column when loading real data
    loaded_df = load_dataframe_with_parsed_dates('your_data.csv', date_columns=['date'])

    # Compute the correlation matrix safely
    attempt_correlation(loaded_df, error_prefix="Error:")