from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.models.text_to_speech import load_vocoder
import soundfile as sf

def load_and_configure_model():
    models, cfg, task = load_model_ensemble_and_task(
        ["jvn/checkpoint_best.pt"],
        arg_overrides={"vocoder": "hifigan", "data": "jvn"}
    )
    model = models[0]
    vocoder = load_vocoder(cfg.vocoder, model)
    return model, vocoder, task

def generate_speech(model, task, vocoder, text):
    sample = task.build_generator([model], cfg).generate(model, {"src_text": text})
    return sample[0] if sample else None

def save_output(wav):
    if wav:
        sf.write("output.wav", wav, 16000)
    else:
        print("No audio generated - empty output after filtering")

if __name__ == "__main__":
    model, vocoder, task = load_and_configure_model()
    text = "こんにちは世界"  # Japanese "Hello world"
    sample = generate_speech(model, task, vocoder, text)

    if sample:
        wav = vocoder(sample["waveform"])
        print(f"Text after filtering OOV: {sample['text']}")
        print(f"Waveform length: {len(wav)}")
        save_output(wav)
    else:
        print("Failed to generate speech.")