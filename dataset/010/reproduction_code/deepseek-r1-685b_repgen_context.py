import tensorflow as tf
from official.vision import keras_cv
from official.vision.dataloaders import segmentation_input

def create_segmentation_model():
    model = keras_cv.models.SegmentationModel(
        backbone='resnet50',
        num_classes=2,
        backbone_weights=None
    )
    return model

def create_dummy_dataset():
    def generator():
        for _ in range(10):
            yield (tf.random.uniform((1, 256, 256, 3)),
                   tf.random.uniform((1, 256, 256), maxval=2, dtype=tf.int32))
    
    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32), output_shapes=((1, 256, 256, 3), (1, 256, 256)))
    return dataset

def compile_model(model):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

def train_model(model, train_ds, val_ds):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2,
        steps_per_epoch=5,
        validation_steps=2
    )
    return history

def main():
    model = create_segmentation_model()
    train_ds = create_dummy_dataset()
    val_ds = create_dummy_dataset()
    
    compile_model(model)
    
    history = train_model(model, train_ds, val_ds)

    print("\nTraining history:")
    print(history.history)

if __name__ == "__main__":
    main()