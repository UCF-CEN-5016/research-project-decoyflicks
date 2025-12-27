from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.models.text_to_speech import load_vocoder
import soundfile as sf

def load_jvn_model_and_vocoder():
    # Load the model (assuming jvn.tar.gz is extracted in current directory)
    models, cfg, task = load_model_ensemble_and_task(
        ["jvn/checkpoint_best.pt"],
        arg_overrides={"vocoder": "hifigan", "data": "jvn"}
    )
    model = models[0]
    vocoder = load_vocoder(cfg.vocoder, model)
    return model, vocoder, task

def generate_speech(model, vocoder, task, text):
    sample = task.build_generator([model], cfg).generate(model, {"src_text": text})
    wav = vocoder(sample[0]["waveform"])
    return sample, wav

def save_output(wav):
    if len(wav) > 0:
        sf.write("output.wav", wav, 16000)
    else:
        print("No audio generated - empty output after filtering")

if __name__ == "__main__":
    model, vocoder, task = load_jvn_model_and_vocoder()
    
    # Text that might trigger the issue
    text = "こんにちは世界"  # Japanese "Hello world"
    
    sample, wav = generate_speech(model, vocoder, task, text)
    
    # Check for empty output
    print(f"Text after filtering OOV: {sample[0]['text']}")  # Likely empty
    print(f"Waveform length: {len(wav)}")  # Check if audio was generated
    
    # Save output (if any)
    save_output(wav)