from transformers import ParquetConfig

def _create_default_parquet_config() -> ParquetConfig:
    """Create and return a ParquetConfig instance with default settings."""
    return ParquetConfig()

DEFAULT_PARQUET_CONFIG = _create_default_parquet_config()