import tensorflow as tf
from tensorflow.keras import layers

# Preprocess the image data (adjusting for float conversion)
image = x_train[np.random.choice(range(x_train.shape[0]))]
resized_image = ops.image.resize(
    ops.convert_to_tensor([image]), size=(image_size, image_size)
)
resized_image = tf.cast(resized_image, dtype=tf.float32)  # Ensure float type

# Extract patches
patches_layer = layers.Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

# Visualize patches
n = int(tf.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
    plt.axis("off")