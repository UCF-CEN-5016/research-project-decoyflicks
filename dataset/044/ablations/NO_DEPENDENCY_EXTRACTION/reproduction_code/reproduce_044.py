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
    pass

def get_uroman_tokens(norm_transcripts, uroman_path, lang):
    # Placeholder for getting uroman tokens
    pass

def text_normalize(line, lang):
    # Placeholder for text normalization
    return line

def time_to_frame(time):
    # Placeholder for converting time to frame index
    return int(time * SAMPLING_FREQ)

def generate_emissions(model, audio_file):
    waveform, _ = torchaudio.load(audio_file)
    waveform = waveform.to(DEVICE)
    total_duration = sox.file_info.duration(audio_file)
    audio_sf = sox.file_info.sample_rate(audio_file)
    assert audio_sf == SAMPLING_FREQ
    emissions_arr = []
    with torch.inference_mode():
        i = 0
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)
            context = EMISSION_INTERVAL * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[:, int(SAMPLING_FREQ * input_start_time): int(SAMPLING_FREQ * (input_end_time))]
            model_outs, _ = model(waveform_split)
            emissions_ = model_outs[0]
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)
            emissions_ = emissions_[emission_start_frame - offset: emission_end_frame - offset, :]
            emissions_arr.append(emissions_)
            i += EMISSION_INTERVAL
    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)
    stride = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)
    return emissions, stride

def merge_repeats(path, dictionary):
    # Placeholder for merging repeated segments
    return path

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
    print("Tokens before filtering OOV:", tokens)
    blank = dictionary["<blank>"]
    targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)
    input_lengths = torch.tensor(emissions.shape[0])
    target_lengths = torch.tensor(targets.shape[0])
    
    # Assuming F is a module that contains the forced alignment function
    import some_forced_alignment_library as F  # Placeholder for the actual library
    path, _ = F.forced_align(emissions, targets, input_lengths, target_lengths, blank=blank)
    path = path.to("cpu").tolist()
    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return segments, stride

def main():
    model, dictionary = load_model_dict()
    model = model.to(DEVICE)
    audio_filepath = "<path_to_audio_file>"
    text_filepath = "<path_to_test_transcript.txt>"
    outdir = "<output_directory>"
    use_star = False

    transcripts = []
    with open(text_filepath) as f:
        transcripts = [line.strip() for line in f]
    print("Read {} lines from {}".format(len(transcripts), text_filepath))

    norm_transcripts = [text_normalize(line.strip(), 'ja') for line in transcripts]
    tokens = get_uroman_tokens(norm_transcripts, '<uroman_path>', 'ja')

    segments, stride = get_alignments(audio_filepath, tokens, model, dictionary, use_star)
    print("text after filtering OOV:", tokens)  # This line is crucial for reproducing the bug

if __name__ == "__main__":
    main()