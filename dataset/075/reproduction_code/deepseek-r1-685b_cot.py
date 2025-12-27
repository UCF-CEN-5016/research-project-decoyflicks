from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import deepspeed

# Minimal setup to trigger the same error
def reproduce_bug():
    # Load model similar to what autotuning would do
    model_name = "gpt2-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Try to load a dataset with token parameter (what autotuning might do)
    try:
        # This mimics what DeepSpeed autotuning might do internally
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", 
                             split="train",
                             token=True)  # This is the problematic parameter
        print("Dataset loaded successfully")
    except Exception as e:
        print(f"Error: {e}")  # Should show the ParquetConfig error

    # Initialize DeepSpeed (not strictly needed to reproduce but shows full context)
    ds_engine = deepspeed.init_inference(model, mp_size=1)
    
if __name__ == "__main__":
    reproduce_bug()