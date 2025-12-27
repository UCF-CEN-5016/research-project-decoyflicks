import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

np.random.seed(42)
url = 'https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/dataset/PeMSD7_Full.zip'
data_dir = keras.utils.get_file(origin=url, extract=True, archive_format="zip")
data_dir = data_dir.rstrip("PeMSD7_Full.zip")

route_distances = pd.read_csv(os.path.join(data_dir, 'PeMSD7_W_228.csv'), header=None).to_numpy()
speeds_array = pd.read_csv(os.path.join(data_dir, 'PeMSD7_V_228.csv'), header=None).to_numpy()

print(f'route_distances shape={route_distances.shape}')
print(f'speeds_array shape={speeds_array.shape}')

sample_routes = [0, 1, 4, 7, 8, 11, 15, 108, 109, 114, 115, 118, 120, 123, 124, 126, 127, 129, 130, 132, 133, 136, 139, 144, 147, 216]
route_distances = route_distances[np.ix_(sample_routes, sample_routes)]
speeds_array = speeds_array[:, sample_routes]

print(f'route_distances shape={route_distances.shape}')
print(f'speeds_array shape={speeds_array.shape}')

def preprocess(data_array: np.ndarray, train_size: float, val_size: float):
    num_time_steps = data_array.shape[0]
    num_train, num_val = (int(num_time_steps * train_size), int(num_time_steps * val_size))
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train : (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val) :] - mean) / std

    return train_array, val_array, test_array

train_array, val_array, test_array = preprocess(speeds_array, 0.5, 0.2)

print(f'train set size: {train_array.shape}')
print(f'validation set size: {val_array.shape}')
print(f'test set size: {test_array.shape}')

def create_tf_dataset(data_array: np.ndarray, input_sequence_length: int, forecast_horizon: int, batch_size: int = 128, shuffle=True, multi_horizon=True):
    inputs = keras.utils.timeseries_dataset_from_array(
        np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    target_offset = (input_sequence_length if multi_horizon else input_sequence_length + forecast_horizon - 1)
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = keras.utils.timeseries_dataset_from_array(
        data_array[target_offset:],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()

input_sequence_length = 12
forecast_horizon = 3
batch_size = 64

train_dataset, val_dataset = (create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size) for data_array in [train_array, val_array])
test_dataset = create_tf_dataset(test_array, input_sequence_length, forecast_horizon, batch_size=test_array.shape[0], shuffle=False, multi_horizon=False)

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xlabel("road number")
    plt.ylabel("road number")

show_heatmap(pd.DataFrame(train_array))