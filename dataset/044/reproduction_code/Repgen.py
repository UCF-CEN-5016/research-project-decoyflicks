# reproduce_047_all_steps.py
import os
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model():
    """Download the Japanese TTS model if not already present"""
    if not os.path.exists("jvn.tar.gz") and not os.path.exists("jvn"):
        logger.info("Downloading Japanese TTS model...")
        subprocess.run(["wget", "https://dl.fbaipublicfiles.com/mms/tts/jvn.tar.gz"], check=True)
    else:
        logger.info("Model already downloaded")

def extract_model():
    """Extract the model if not already extracted"""
    if not os.path.exists("jvn") and os.path.exists("jvn.tar.gz"):
        logger.info("Extracting model...")
        subprocess.run(["tar", "-xzf", "jvn.tar.gz"], check=True)
    else:
        logger.info("Model already extracted or download failed")

def run_inference(input_text):
    """Run the inference script with the given input text"""
    if not os.path.exists("jvn"):
        logger.error("Model directory not found. Cannot run inference.")
        return None
    
    os.chdir("jvn")
    logger.info(f"Running inference with input text: {input_text}")
    
    try:
        # Run the inference script
        result = subprocess.run(
            ["python", "infer.py", "--input_text", input_text],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Return to original directory
        os.chdir("..")
        
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Inference failed with error: {e}")
        logger.error(f"Error output: {e.stderr}")
        # Return to original directory
        os.chdir("..")
        return None

def check_output(output):
    """Check if the output contains filtered text after OOV filtering"""
    if output is None:
        logger.error("No output to check")
        return False
    
    # Look for the line with "text after filtering OOV:"
    lines = output.split("\n")
    for i, line in enumerate(lines):
        if "text after filtering OOV:" in line:
            # Check if there's any text after this line
            if i + 1 < len(lines) and lines[i + 1].strip():
                logger.info(f"Found non-empty filtered text: {lines[i + 1]}")
                return True
            else:
                logger.error("Empty output after OOV filtering")
                return False
    
    logger.error("Could not find 'text after filtering OOV:' in output")
    return False

def main():
    """Main function to run the bug reproduction steps"""
    logger.info("Starting bug reproduction process")
    
    # Step 1 & 2: Download and extract the model
    download_model()
    extract_model()
    
    # Step 3: Navigate to the model directory (done in run_inference)
    
    # Step 4: Run inference with Japanese test text
    test_text = "これはテストです"  # "This is a test" in Japanese
    output = run_inference(test_text)
    
    # Step 5: Check if the output after OOV filtering is empty
    if output:
        logger.info("Output received from inference")
        success = check_output(output)
        if success:
            logger.info("Bug not reproduced - output after OOV filtering is not empty")
        else:
            logger.info("Bug reproduced - output after OOV filtering is empty")
    else:
        logger.error("Failed to get output from inference")

if __name__ == "__main__":
    main()