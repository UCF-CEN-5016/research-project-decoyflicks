def main():
    import os
    import argparse
    from distutils.dirlib import join

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    args = parser.parse_args()

    def infer(audio_paths):
        trans = []
        for audio_path in audio_paths:
            # ... existing code to process the audio ...
            # Append (filename, output) to trans
            pass
        return trans

    trans = infer(args.audios)

    with open(args.output_file, 'w') as f:
        # Write each line by its original order
        for filename in sorted(trans.keys()):
            output = trans[filename]
            print(f"Dataset/speech/.../{filename}")
            print(output)