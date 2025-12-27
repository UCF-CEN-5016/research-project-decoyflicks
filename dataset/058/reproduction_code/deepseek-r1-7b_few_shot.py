import tensorflow as tf
from tensorflow.keras.applications import-retinanet
from tensorflow.keras_cv.models.object_detection import retinanet
from tensorflow.keras_cv.utils.target_gather import _gather_batched

class CustomDataset(tf.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Each sample has (image, boxes, labels)
        image, boxes, labels = self.samples[idx]
        
        if boxes is None or labels is None:
            raise ValueError("Sample must contain image, boxes, and labels")
            
        return {
            'image': image,
            'boxes': tf.expand_dims(tf.convert_to_tensor(boxes), axis=1),
            'labels': tf.expand_dims(tf.convert_to_tensor(labels), axis=1)
        }

# Create a sample with an invalid index scenario
def create_incorrect_targets():
    # One image with 63950 boxes, trying to assign label 0 (which doesn't exist)
    image = tf.random.uniform((256, 256, 3))
    boxes = [[0.1*i for i in range(63949)]]
    labels = [0]  # Label index out of bounds
    
    return {'image': image, 'boxes': boxes, 'labels': labels}

# Minimal reproduction code
def minimal_reproduction_code():
    try:
        model = retinanetRetinaNet()
        
        # Create dataset with problematic sample
        samples = [(create_incorrect_targets(), create_incorrect_targets())]
        train_dataset = tf.data.Dataset.from_generator(
            lambda: (sample['image'], sample['boxes'], sample['labels']) for sample in [samples[0]]
        )
        
        # Fit the model, this should trigger the InvalidArgumentError
        model.fit(
            train_dataset,
            epochs=1,
            verbose=2  # For visibility of error details
        )
    except InvalidArgumentError as e:
        print(f"Invalid argument error: {e}")

# Run the minimal reproduction code
minimal_reproduction_code()