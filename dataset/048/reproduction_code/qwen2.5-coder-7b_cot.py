import os
import subprocess
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from fairseq import data_prep  # kept for compatibility with original code


BASE_DIR = Path("data_prep")


def ensure_directories(base: Path) -> None:
    (base / "alignments").mkdir(parents=True, exist_ok=True)
    (base / "text").mkdir(parents=True, exist_ok=True)
    (base / "models").mkdir(parents=True, exist_ok=True)


def write_text_file(base: Path, filename: str, content: str) -> Path:
    file_path = base / "text" / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fh:
        fh.write(content)
    return file_path


def download_audio(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        # fallback to a no-op if download fails in this environment
        if not dest.exists():
            dest.write_bytes(b"")


def write_model_download_script(base: Path) -> Path:
    script_path = base / "models" / "download_model.sh"
    script_content = """#!/bin/bash
wget --content-type=application/x-tar.gz --no-check-certificate \
    "https://dl.fbaipublicfiles.com/fairseq/models/w2v-Large-en.tar.gz" \
    -O data_prep/models/w2v-Large-en.tar.gz

tar xzf data_prep/models/w2v-Large-en.tar.gz
rm data_prep/models/w2v-Large-en.tar.gz
"""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with script_path.open("w", encoding="utf-8") as fh:
        fh.write(script_content)
    try:
        script_path.chmod(0o755)
    except Exception:
        pass
    return script_path


def run_alignment_script(audio_path: Path, text_path: Path, lang: str, outdir: Path, uroman_bin: str) -> None:
    cmd = (
        f"python data_prep/align_and_segment.py --audio_filepath {str(audio_path)} "
        f"--text_filepath {str(text_path)} --lang {lang} --outdir {str(outdir)} --uroman {uroman_bin}"
    )
    # Run but do not fail hard if the script does not exist in the environment
    try:
        subprocess.run(cmd, shell=True, check=False)
    except Exception:
        pass


def generate_emissions_from_outputs(model_outputs: list[torch.Tensor]) -> torch.Tensor:
    """
    Given a list of emission tensors (shape: [batch, time_features]),
    perform per-step padding per the original logic and concatenate them.
    The function:
      - computes a reference max length across all outputs,
      - for each step > 0 it checks current_length = time - 1 and pads to reference if needed,
      - after collecting all emissions pads them to the same max length and concatenates along dim=1,
      - squeezes the result.
    """
    if not model_outputs:
        return torch.tensor([])

    # Reference maximum (used for in-loop padding)
    reference_max = max(t.size(1) for t in model_outputs)

    emissions_list = []
    for step_idx, emissions in enumerate(model_outputs):
        # Keep emissions as-is for step 0; for subsequent steps apply the described adjustment
        if step_idx > 0:
            previous_emission = model_outputs[step_idx - 1]
            current_length = emissions.size(1) - 1
            if current_length < reference_max:
                pad_amount = reference_max - current_length
                # pad on the last dimension (time/features) to the right
                emissions = F.pad(emissions, (0, pad_amount))
        emissions_list.append(emissions)

    # Ensure final uniform length across all collected emissions
    max_len = max(t.size(1) for t in emissions_list)
    padded_emissions = [F.pad(t, (0, max_len - t.size(1))) for t in emissions_list]
    concatenated = torch.cat(padded_emissions, dim=1).squeeze()
    return concatenated


def main() -> None:
    ensure_directories(BASE_DIR)

    # Write example text file (kept content identical to original)
    text_filename = "B27_20_Apocalypse.txt"
    write_text_file(BASE_DIR, text_filename, "ful\nB27_20_Apocalypse")

    # Download audio (original used curl; here we use urllib with a safe fallback)
    audio_dest = BASE_DIR / "audio.wav"
    download_audio("https://example.com/audio.wav", audio_dest)

    # Write model downloading script
    write_model_download_script(BASE_DIR)

    # Run alignment script (attempt, but don't fail if unavailable)
    run_alignment_script(
        audio_path=audio_dest,
        text_path=BASE_DIR / "text" / text_filename,
        lang="ful",
        outdir=BASE_DIR / "alignments",
        uroman_bin="uroman/bin",
    )

    # Simulate model outputs for emission generation.
    # These represent tensors from each processing step; batch size 1 and varying time lengths.
    torch.manual_seed(0)
    simulated_outputs = [
        torch.randn(1, L) for L in [8, 6, 10, 7, 9]
    ]

    concatenated = generate_emissions_from_outputs(simulated_outputs)
    # Print shape to mimic usage
    print("Concatenated emissions shape:", tuple(concatenated.shape) if concatenated.numel() else concatenated.shape)


if __name__ == "__main__":
    main()