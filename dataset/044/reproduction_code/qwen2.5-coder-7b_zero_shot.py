import torch
from typing import List, Any
from transformers import JvNTTSForConditionalGeneration, JvNTTSPipeline


def load_tts_model(model_name: str) -> JvNTTSForConditionalGeneration:
    """Load and return the JvNTTS TTS model from pretrained weights."""
    return JvNTTSForConditionalGeneration.from_pretrained(model_name)


def build_tts_pipeline(model: JvNTTSForConditionalGeneration) -> JvNTTSPipeline:
    """Create and return a TTS pipeline for the given model."""
    return JvNTTSPipeline(model)


def synthesize_text(
    pipeline: JvNTTSPipeline,
    text: str,
    input_ids: List[str],
    max_length: int = 50,
    early_stopping: bool = True,
) -> Any:
    """Run the pipeline to synthesize text/audio (preserves original pipeline behavior)."""
    return pipeline(text, input_ids=input_ids, max_length=max_length, early_stopping=early_stopping)


def main() -> None:
    model_name = "jvn"
    source_text = "これはテストの文です"
    token_ids = ["5418", "1114", "2345", "5419"]

    tts_model = load_tts_model(model_name)
    tts_pipeline = build_tts_pipeline(tts_model)

    result = synthesize_text(tts_pipeline, source_text, token_ids, max_length=50, early_stopping=True)
    print(result)


if __name__ == "__main__":
    main()