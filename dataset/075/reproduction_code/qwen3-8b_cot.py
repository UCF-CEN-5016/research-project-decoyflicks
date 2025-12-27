# Minimal reproduction of the bug: ParquetConfig.__init__() got an unexpected keyword argument 'token'

class ParquetConfig:
    def __init__(self, some_param):
        pass  # ParquetConfig does not accept 'token' parameter

# Attempting to initialize ParquetConfig with 'token' argument
parquet_config = ParquetConfig(token="some_token")