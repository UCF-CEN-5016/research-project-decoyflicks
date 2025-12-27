import os
import logging
from fairseq import _build_optimizer, build_lr_scheduler

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Function to load and prepare model
def prepare_model():
    # Download model
    os.system("wget https://dl.fbaipublicfiles.com/mms/tts/jvn.tar.gz")
    # Extract model
    os.system("tar -xzf jvn.tar.gz")
    # Navigate into model directory
    os.chdir("jvn")

# Function to run inference
def run_inference(input_text):
    # Run infer.py with input text
    output = os.popen(f"python infer.py --input {input_text}").read()
    return output

# Main function to reproduce the bug
def reproduce_bug():
    prepare_model()
    input_text = "input_text.txt"
    output = run_inference(input_text)
    logging.debug("Inference Output: {}".format(output))
    if not output.strip():
        logging.error("Empty output after OOV filtering")
    else:
        logging.info("OOV filtering successful")

# Run the bug reproduction
reproduce_bug()