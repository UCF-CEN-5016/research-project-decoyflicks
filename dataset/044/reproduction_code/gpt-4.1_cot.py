import os
from omegaconf import OmegaConf
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.data import Dictionary

# Minimal example: simulate loading the model and dictionary,
# tokenize Japanese text, filter OOV tokens, and show empty output.

def main():
    # Example Japanese text (can be any Japanese sentence)
    text = "こんにちは、元気ですか？"  # "Hello, how are you?"

    # Load model and task from the MMS repository
    # Note: Replace with actual MMS loading if needed
    repo_id = "facebook/mms-jvn"
    model_tag = "tts_jvn"
    # This example assumes you have fairseq installed and can load from HF hub.
    # If not, loading local model and dictionary is required.
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        repo_id,
        [model_tag],
    )

    # Get source dictionary (text tokens)
    src_dict: Dictionary = task.source_dictionary

    # Tokenize input text
    # The MMS TTS expects tokenization compatible with the model
    # Here we simulate tokenization by simple split or character splitting
    # (In MMS, tokenization might be more complex)

    # For demonstration, split text by characters (Japanese tokenization)
    tokens = list(text)

    print(f"Original tokens: {tokens}")

    # Filter out tokens not in dictionary (simulate OOV filtering)
    filtered_tokens = [t for t in tokens if t in src_dict]

    print(f"Tokens after filtering OOV: {filtered_tokens}")

    # If filtered_tokens is empty, it reproduces the issue:
    if not filtered_tokens:
        print("Filtered tokens are empty after OOV filtering - reproducing bug.")

if __name__ == "__main__":
    main()