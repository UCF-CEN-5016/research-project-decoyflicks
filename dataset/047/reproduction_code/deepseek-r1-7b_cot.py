import argparse
from torchaudio import load

def main():
    parser = argparse.ArgumentParser()
    # ... (existing arguments parsing code)

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file {args.model} not found.")

    dataset = ASVspoof2019Dataset(root=args.data_dir, split=args.split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    model = ASVspoof2019Model(args.model)

    # Main inference loop with correct output formatting
    outputs = []  # To store each file's input and output in order

    for audio_path in args.audio:
        print(f"Processing {audio_path}...")
        wav, sample_rate = torchaudio.load(audio_path)
        with torch.no_grad():
            output = model(wav, sample_rate)
        
        # Append the Input (path) followed by Output (text) to outputs
        outputs.append( (audio_path, output) )

    if len(outputs) > 0:
        import pandas as pd
        df = pd.DataFrame(outputs, columns=['Input', 'Output'])
        print("\nOutputs:")
        for index, row in df.iterrows():
            print(f"Input: {row['Input']}")
            print(f"Output: {row['Output'].text}")  # Assuming Output is an object with text attribute