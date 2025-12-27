from datasets import ParquetConfig

try:
    config = ParquetConfig(token="test")
    print("Successfully created ParquetConfig")
except Exception as e:
    print(f"Error: {e}")