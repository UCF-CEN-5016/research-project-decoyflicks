import keras_cv as kcv


class COCOEvalCallback(kcv.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Intentionally empty: not used during training
        pass


def build_yolov8_model(input_shape=(640, 640, 3), num_classes=1):
    return kcv.YOLOV8(
        input_shape=input_shape,
        anchors=kcv.yolov8.anchors(),
        classes=num_classes,
    )


def make_kerascv_dataset(image_path, annotation, batch_size=16):
    return kcv.dataset.KerasCVDataset(
        image_paths=[image_path],
        annotations=[annotation],
        batch_size=batch_size,
    )


def main():
    model = build_yolov8_model()

    sample_annotation = {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "class_id": 1}

    train_dataset = make_kerascv_dataset("path/to/train/image", sample_annotation, batch_size=16)
    val_dataset = make_kerascv_dataset("path/to/validation/image", sample_annotation, batch_size=16)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=3,
        callbacks=[COCOEvalCallback()],
    )


if __name__ == "__main__":
    main()