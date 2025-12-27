import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

TEST_IMAGE_PATHS = ['path/to/your/image']


def load_detection_model(model_name: str) -> tf.Module:
    """Download and load a TensorFlow Object Detection saved_model."""
    base_url = 'http://download.tensorflow.org/models/research/object_detection/'
    archive_name = model_name + '.tar.gz'

    downloaded_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + archive_name,
        untar=True
    )

    saved_model_dir = pathlib.Path(downloaded_dir) / "saved_model"
    print(saved_model_dir)

    model = tf.saved_model.load_v2(str(saved_model_dir))
    # Access the serving signature to ensure it's available/warmed.
    model.signatures["serving_default"]

    return model


def main():
    model = load_detection_model('ssd_mobilenet_v2_coco_2018_03_29')
    for image_path in TEST_IMAGE_PATHS:
        show_inference(model, image_path)


if __name__ == "__main__":
    main()