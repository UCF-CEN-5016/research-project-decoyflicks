import os
import pickle
import tensorflow as tf
from delf.python.datasets import tuples_dataset
from delf.python.datasets import utils

def id2filename(image_id, prefix):
    if prefix:
        return os.path.join(prefix, image_id[-2:], image_id[-4:-2], image_id[-6:-4], image_id)
    else:
        return os.path.join(image_id[-2:], image_id[-4:-2], image_id[-6:-4], image_id)

class _Sfm120k(tuples_dataset.TuplesDataset):
    def __init__(self, mode, data_root, imsize=None, num_negatives=5, num_queries=2000, pool_size=20000, loader=utils.default_loader, eccv2020=False):
        if mode not in ['train', 'val']:
            raise ValueError("`mode` argument should be either 'train' or 'val', passed as a String.")
        if eccv2020:
            name = "retrieval-SfM-120k-val-eccv2020"
        else:
            name = "retrieval-SfM-120k"
        db_root = os.path.join(data_root, 'train/retrieval-SfM-120k')
        ims_root = os.path.join(db_root, 'ims/')
        db_filename = os.path.join(db_root, '{}.pkl'.format(name))
        with tf.io.gfile.GFile(db_filename, 'rb') as f:
            db = pickle.load(f)[mode]
        self.images = [id2filename(img_name, None) for img_name in db['cids']]
        super().__init__(name, mode, db_root, imsize, num_negatives, num_queries, pool_size, loader, ims_root)

def CreateDataset(mode, data_root, imsize=None, num_negatives=5, num_queries=2000, pool_size=20000, loader=utils.default_loader, eccv2020=False):
    return _Sfm120k(mode, data_root, imsize, num_negatives, num_queries, pool_size, loader, eccv2020)

tf.random.set_seed(42)
data_root = '/path/to/sfm120k'
dataset = CreateDataset(mode='val', data_root=data_root, imsize=640, num_negatives=5, num_queries=2000, pool_size=20000)
batch_size = 8
validation_dataset = tf.data.Dataset.from_tensor_slices(dataset.images).batch(batch_size)

# Placeholder for train_dataset to maintain bug reproduction logic
# In a real scenario, train_dataset should be defined and populated with training data
train_dataset = tf.data.Dataset.from_tensor_slices(dataset.images).batch(batch_size)  # Dummy dataset for reproduction

model = tf.keras.applications.MaskRCNN(weights='coco')
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model (this is where the bug is expected to manifest)
model.fit(train_dataset, epochs=5)

# Evaluate the model on the validation dataset
metrics = model.evaluate(validation_dataset)

# Print the validation loss
print('Validation Loss:', metrics['validation_loss'])

# Assert that the validation loss is zero to reproduce the bug
assert metrics['validation_loss'] == 0.0, 'Validation loss is not zero.'