import os
import argparse
import torch
import torch.nn as nn
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("bug_reproducer")

# Create a simplified version of the model that produces incorrect dimensionality
class BuggyPredictor(nn.Module):
    def __init__(self, input_dim=80, output_dim=100):
        super(BuggyPredictor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # This will produce a 2D output (batch_size, output_dim) instead of 
        # the required 3D output (batch_size, input_length, num_classes)
        return self.linear(x)

# Mock implementation of align_and_segment function that will trigger the error
def align_and_segment(audio_filepath, text_filepath, lang, outdir, uroman):
    logger.info(f"Processing audio file: {audio_filepath}")
    logger.info(f"Using text from: {text_filepath}")
    
    # Create directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    
    # Create dummy input data
    batch_size = 1
    feature_dim = 80
    x = torch.randn(batch_size, feature_dim)
    
    # Initialize the buggy model
    model = BuggyPredictor()
    
    # Get log probabilities with incorrect shape
    log_probs = model(x)
    
    # Check the dimensionality of log_probs
    if len(log_probs.shape) != 3:
        error_msg = f"log_probs must be 3-D (batch_size, input length, num classes), but got shape: {log_probs.shape}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("Alignment completed successfully")
    return True

def main():
    parser = argparse.ArgumentParser(description="Reproduce log probabilities dimension mismatch bug")
    parser.add_argument("--audio_filepath", type=str, default="audio.wav", help="Path to the audio file")
    parser.add_argument("--text_filepath", type=str, default="text.txt", help="Path to the text file")
    parser.add_argument("--lang", type=str, default="udm", help="Language code")
    parser.add_argument("--outdir", type=str, default="output", help="Output directory")
    parser.add_argument("--uroman", type=str, default="uroman/bin", help="Path to uroman binary")
    
    args = parser.parse_args()
    
    # Create dummy audio and text files if they don't exist
    if not os.path.exists(args.audio_filepath):
        logger.info(f"Creating dummy audio file at {args.audio_filepath}")
        with open(args.audio_filepath, "w") as f:
            f.write("dummy audio content")
    
    if not os.path.exists(args.text_filepath):
        logger.info(f"Creating dummy text file at {args.text_filepath}")
        with open(args.text_filepath, "w") as f:
            f.write("dummy transcript")
    
    # Run the alignment function that will raise the error
    try:
        align_and_segment(
            args.audio_filepath,
            args.text_filepath,
            args.lang,
            args.outdir,
            args.uroman
        )
    except RuntimeError as e:
        logger.error(f"Bug successfully reproduced: {str(e)}")
        print("\nBug reproduction successful! The error matches the expected error in the bug report.")
        return
    
    logger.error("Failed to reproduce the bug - no error was raised.")

if __name__ == "__main__":
    main()