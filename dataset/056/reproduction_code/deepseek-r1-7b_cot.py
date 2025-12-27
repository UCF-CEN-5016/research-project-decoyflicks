resized_image = ops.image.resize(
    ops.convert_to_tensor([image]), size=(image_size, image_size)
)

# Cast to float to match Conv2D expectations
resized_image = tf.cast(resized_image, dtype=tf.float32)

patches = Patches(patch_size)(resized_image)