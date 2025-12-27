import torch
import torchaudio
import sox
import json
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 1.0

def load_model_dict():
    # Placeholder for loading model and dictionary
    return None, {}  # Return a dummy model and empty dictionary for testing

def text_normalize(text, lang):
    # Placeholder for text normalization
    return text

def get_uroman_tokens(norm_transcripts, uroman_path, lang):
    # Placeholder for getting uroman tokens
    return norm_transcripts

def generate_emissions(model, audio_file):
    # Placeholder for generating emissions
    return torch.zeros((100, 10)), 1  # Dummy emissions and stride for testing

def merge_repeats(path, dictionary):
    # Placeholder for merging repeated tokens
    return path  # Return the path as is for testing

def get_alignments(audio_file, tokens, model, dictionary, use_star):
    emissions, stride = generate_emissions(model, audio_file)
    T, N = emissions.size()
    if use_star:
        emissions = torch.cat([emissions, torch.zeros(T, 1).to(DEVICE)], dim=1)

    if tokens:
        token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    else:
        print(f"Empty transcript!!!!! for audio file {audio_file}")
        token_indices = []

    blank = dictionary.get("<blank>", -1)  # Use -1 if <blank> is not in dictionary
    targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)
    input_lengths = torch.tensor(emissions.shape[0])
    target_lengths = torch.tensor(targets.shape[0])

    path, _ = F.forced_align(emissions, targets, input_lengths, target_lengths, blank=blank)
    path = path.to("cpu").tolist()
    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return segments, stride

def main(args):
    assert not os.path.exists(args.outdir), f"Error: Output path exists already {args.outdir}"

    transcripts = []
    with open(args.text_filepath) as f:
        transcripts = [line.strip() for line in f]
    print("Read {} lines from {}".format(len(transcripts), args.text_filepath))

    norm_transcripts = [text_normalize(line.strip(), args.lang) for line in transcripts]
    tokens = get_uroman_tokens(norm_transcripts, args.uroman_path, args.lang)

    model, dictionary = load_model_dict()
    model = model.to(DEVICE) if model is not None else None  # Check if model is loaded
    if args.use_star:
        dictionary["<star>"] = len(dictionary)
        tokens = ["<star>"] + tokens
        transcripts = ["<star>"] + transcripts
        norm_transcripts = ["<star>"] + norm_transcripts

    segments, stride = get_alignments(args.audio_filepath, tokens, model, dictionary, args.use_star)
    print("text after filtering OOV:", tokens)  # This line is crucial for reproducing the bug

if __name__ == "__main__":
    class Args:
        outdir = "output"
        text_filepath = "test_transcripts.txt"
        audio_filepath = "test_audio.flac"
        lang = "ja"
        uroman_path = "path/to/uroman"
        use_star = False

    main(Args())