import argparse
import os
from typing import List, Tuple, Any

import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASVspoof2019 model on audio files.")
    parser.add_argument("--model", required=True, help="Path to the model file.")
    parser.add_argument("--data_dir", default=".", help="Root directory of the dataset.")
    parser.add_argument("--split", default="eval", help="Dataset split to use.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset.")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for DataLoader.")
    parser.add_argument("--audio", nargs="+", required=True, help="One or more audio file paths to process.")
    return parser.parse_args()


def validate_args(opts: argparse.Namespace) -> None:
    if not os.path.exists(opts.model):
        raise FileNotFoundError(f"Model file {opts.model} not found.")


def prepare_dataloader(opts: argparse.Namespace):
    dataset = ASVspoof2019Dataset(root=opts.data_dir, split=opts.split)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        shuffle=opts.shuffle,
        pin_memory=opts.pin_memory,
        drop_last=False,
    )
    return data_loader


def run_inference_on_files(model: Any, audio_paths: List[str]) -> List[Tuple[str, Any]]:
    results: List[Tuple[str, Any]] = []
    for audio_path in audio_paths:
        print(f"Processing {audio_path}...")
        wav, sample_rate = torchaudio.load(audio_path)
        with torch.no_grad():
            output = model(wav, sample_rate)
        results.append((audio_path, output))
    return results


def main():
    opts = parse_args()
    validate_args(opts)

    data_loader = prepare_dataloader(opts)

    model = ASVspoof2019Model(opts.model)

    results = run_inference_on_files(model, opts.audio)

    if len(results) > 0:
        import pandas as pd

        df = pd.DataFrame(results, columns=["Input", "Output"])
        print("\nOutputs:")
        for _, row in df.iterrows():
            print(f"Input: {row['Input']}")
            print(f"Output: {row['Output'].text}")


if __name__ == "__main__":
    main()