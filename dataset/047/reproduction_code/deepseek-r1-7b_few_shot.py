import argparse
import threading

# Define model path and language
model_path = '../fairseq_resource/mms1b_all.pt'
lang = 'eng'

def process_audio(input_path):
    """Run inference on an audio file and return output."""
    try:
        # Simulate ASR inference (replace with actual model execution)
        output = "Processed audio file: {}".format(input_path)
        results.append((input_path, output))
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
    
def main():
    parser = argparse.ArgumentParser(description='Run MMS ASR inference')
    parser.add_argument('--model', type=str, default=model_path,
                        help='Path to the MMS model (default: %(default)s)')
    parser.add_argument('--lang', type=str, default=lang,
                        help='Language for ASR (default: %(default)s)')
    args = parser.parse_args()
    
    # List of audio files to process
    audio_files = [
        'audio1.wav', 'audio2.wav', 'audio3.wav', ..., 'audio10.wav'
    ]
    
    # Create a list to hold results with their input paths
    results = []
    
    # Start threads for each audio file
    threads = []
    for file_path in audio_files:
        thread = threading.Thread(target=process_audio, args=(file_path,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to finish before proceeding
    for thread in threads:
        thread.join(timeout=10)  # Timeout after 10 seconds
    
    # Print results in order
    for input_path, output in sorted(results):
        print(f"Input: {input_path}")
        print(f"Output: {output}\n")
        
if __name__ == '__main__':
    main()