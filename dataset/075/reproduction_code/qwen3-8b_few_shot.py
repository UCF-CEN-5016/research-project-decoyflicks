from transformers import ParquetConfig
config = ParquetConfig(token="dummy_token")

from transformers import ParquetConfig

# Attempt to initialize ParquetConfig with an unsupported 'token' argument
config = ParquetConfig(token="dummy_token")