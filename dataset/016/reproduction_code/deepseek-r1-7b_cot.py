import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.distribute import tf as tftf
tftf.set_seed(0)

def main():
    # Cluster information for two workers across multiple nodes.
    cluster = {
        "worker": [
            "ip1:6655.0",
            "ip2:6655.0"
        ],
        "task": {"type": "worker", "index": 0},
        "cluster": {
            "IP address for node 0": "ip1",
            "IP address for node 1": "ip2"
        },
        "num_accelerators": {
            "per_worker": ["GPU:0", "GPU:1"]
        }
    }

    tftf.set_cluster(cluster)
    
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: ", strategy.num_replicas_in_sequence())

if __name__ == "__main__":
    main()