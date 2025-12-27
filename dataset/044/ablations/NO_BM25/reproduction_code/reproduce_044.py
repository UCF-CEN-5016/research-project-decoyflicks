import torch
import torchaudio
import sox
import json
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 1.0

def load_model_dict():
    # Placeholder for loading the model and dictionary
    return None, {}  # Return a dummy model and empty dictionary for the sake of completeness

def text_normalize(text, lang):
    # Placeholder for text normalization
    return text

def get_uroman_tokens(norm_transcripts, uroman_path, lang):
    # Placeholder for getting uroman tokens
    return []

def generate_emissions(model, audio_file):
    # Placeholder for generating emissions
    return torch.zeros((100, 10)), 10  # Dummy emissions and stride

def merge_repeats(path, dictionary):
    # Placeholder for merging repeated tokens
    return path  # Return the path as is for the sake of completeness

def get_spans(tokens, segments):
    # Placeholder for getting spans
    return [(slice(0, len(tokens)),)] * len(tokens)  # Dummy spans for the sake of completeness

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

    blank = dictionary.get("<blank>", -1)  # Use get to avoid KeyError
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
    model = model.to(DEVICE)
    if args.use_star:
        dictionary["<star>"] = len(dictionary)
        tokens = ["<star>"] + tokens
        transcripts = ["<star>"] + transcripts
        norm_transcripts = ["<star>"] + norm_transcripts

    segments, stride = get_alignments(args.audio_filepath, tokens, model, dictionary, args.use_star)
    spans = get_spans(tokens, segments)

    os.makedirs(args.outdir, exist_ok=True)  # Use exist_ok to avoid race conditions
    with open(f"{args.outdir}/manifest.json", "w") as f:
        for i, t in enumerate(transcripts):
            span = spans[i]
            seg_start_idx = span[0].start
            seg_end_idx = span[-1].end

            output_file = f"{args.outdir}/segment{i}.flac"

            audio_start_sec = seg_start_idx * stride / 1000
            audio_end_sec = seg_end_idx * stride / 1000 

            tfm = sox.Transformer()
            tfm.trim(audio_start_sec, audio_end_sec)
            tfm.build_file(args.audio_filepath, output_file)

            sample = {
                "audio_start_sec": audio_start_sec,
                "audio_filepath": str(output_file),
                "duration": audio_end_sec - audio_start_sec,
                "text": t,
                "normalized_text": norm_transcripts[i],
                "uroman_tokens": tokens[i],
            }
            f.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    class Args:
        audio_filepath = "path/to/test_audio.flac"
        text_filepath = "path/to/test_transcript.txt"
        outdir = "output_directory"
        use_star = False
        lang = "ja"
        uroman_path = "path/to/uroman"

    args = Args()
    main(args)