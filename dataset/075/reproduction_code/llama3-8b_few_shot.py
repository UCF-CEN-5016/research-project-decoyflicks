import deepspeed as ds
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

# Initialize DeepSpeed configuration with unexpected keyword argument
ds_config = ds.DeepSpeedConfig(parquet_config=ParquetConfig(token='test'))

print("DeepSpeed configuration:", ds_config)