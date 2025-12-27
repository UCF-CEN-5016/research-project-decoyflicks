import pandas as pd
import matplotlib.pyplot as plt

"""
Bug Type: Data Type Mismatch
Bug Description: DataFrame contains non-numeric columns causing ValueError when computing correlations
Reproduction: build a DataFrame with mixed types, select numeric columns, compute and display correlation matrix
"""


def create_sample_dataframe() -> pd.DataFrame:
    """Create and return a sample DataFrame with mixed data types."""
    return pd.DataFrame({
        'date': ['01.01.2009 00:10:00', '01.01.2009 00:20:00'],
        'temp': [20, 22],
        'humidity': [60, 65]
    })


def get_numeric_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and return only the numeric columns from the given DataFrame."""
    return frame.select_dtypes(include=['int64', 'float64'])


def plot_correlation_matrix(numeric_frame: pd.DataFrame) -> None:
    """Compute correlation matrix of numeric_frame and display it as a matrix plot."""
    corr_matrix = numeric_frame.corr()
    plt.matshow(corr_matrix)
    plt.show()


def main() -> None:
    data_frame = create_sample_dataframe()
    numeric_frame = get_numeric_dataframe(data_frame)
    plot_correlation_matrix(numeric_frame)


if __name__ == '__main__':
    main()