# Minimal reproduction of ParquetConfig initialization error

class ParquetConfig:
    def __init__(self, filepath, format="parquet"):
        self.filepath = filepath
        self.format = format

# The caller incorrectly passes a 'token' argument which is not supported
try:
    config = ParquetConfig(filepath="data.parquet", token="abc123")
except TypeError as e:
    print(f"Caught expected error: {e}")