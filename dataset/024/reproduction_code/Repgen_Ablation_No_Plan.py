import os
import pickle
import tensorflow as tf
from delf import global_features_utils, global_model
import image_loading_utils

class TuplesDataset:
    def __init__(self, name, mode, data_root, imsize=None, num_negatives=5,
                 num_queries=2000, pool_size=20000,
                 loader=image_loading_utils.default_loader, ims_root=None):
        """TuplesDataset object initialization.
        
        Args:
          name: String, dataset name. I.e. 'retrieval-sfm-120k'.
          mode: 'train' or 'val' for training and validation parts of dataset.
          data_root: Path to the root directory of the dataset.
          imsize: Integer, defines the maximum size of longer image side transform.
          num_negatives: Integer, number of negative images for a query image in a
            training tuple.
          num_queries: Integer, number of query images to be processed in one epoch.
          pool_size: Integer, size of the negative image pool, from where the
            hard-negative images are re-mined.
          loader: Callable, a function to load an image given its path.
          ims_root: String, image root directory.
          
        Raises:
          ValueError: If mode is not either 'train' or 'val'.
        """
        if mode not in ['train', 'val']:
            raise ValueError("`mode` argument should be either 'train' or 'val', passed as a "
                             "String.")
        
        # Loading db.
        db_filename = os.path.join(data_root, '{}.pkl'.format(name))
        with tf.io.gfile.GFile(db_filename, 'rb') as f:
            db = pickle.load(f)[mode]
        
        # Initializing tuples dataset.
        self._ims_root = data_root if ims_root is None else ims_root
        self._name = name
        self._mode = mode
        self._imsize = imsize
        self._clusters = db['cluster']
        self._query_pool = db['qidxs']
        self._positive_pool = db['pidxs']
        
        if not hasattr(self, 'images'):
            self.images = db['ids']
        
        # Size of training subset for an epoch.
        self._num_negatives = num_negatives
        self._num_queries = min(num_queries, len(self._query_pool))
        self._pool_size = min(pool_size, len(self.images))
        self._qidxs = None
        self._pidxs = None
        self._nidxs = None
        
        self._loader = loader
        self._print_freq = 10
        # Indexer for the iterator.
        self._n = 0

    def __iter__(self):
        """Function for making TupleDataset an iterator.
        
        Returns:
          iter: The iterator object itself (TupleDataset).
        """
        return self

# Example usage:
# dataset = TuplesDataset('retrieval-sfm-120k', 'train', '/path/to/dataset')
# for batch in dataset:
#     # Process the batch