import argparse
import joblib
import logging
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Setup logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")

# Dummy function to simulate inference
def mms_infer(audio_files, model_path):
    # Simulate loading the model and running inference
    km_model = MiniBatchKMeans(n_clusters=10)  # Example parameters
    km_model.fit(np.random.rand(100, 32))  # Random feature matrix

    results = []
    for audio_file in audio_files:
        result = f"Transcription for {audio_file}"
        results.append(result)
    
    return results

# Main function to simulate the bug reproduction
def reproduce_bug():
    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("nshard", type=int)
    parser.add_argument("km_path", type=str)
    parser.add_argument("n_clusters", type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--percent", default=-1, type=float, help="sample a subset; -1 for all"
    )
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    args = parser.parse_args()

    audio_files = [f"audio{i}.wav" for i in range(1, 11)]
    model_path = "mms1b_all.pt"

    mms_log_output = mms_infer(audio_files, model_path)

    expected_text_order = ["audio{}.wav".format(i) for i in range(1, 11)]
    mms_audio_paths = [result.split(" ")[-1] for result in mms_log_output]

    assert mms_audio_paths != expected_text_order, "The output log order is correct."

if __name__ == "__main__":
    reproduce_bug()