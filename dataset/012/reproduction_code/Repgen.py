import tensorflow as tf

from research.delf.delf.python.training.model import delf_model

class BugReproducer(delf_model.Delf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_classifiers(1000)

    def train_step(self, data):
        images, labels = data
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=2)

        with tf.GradientTape() as gradient_tape:
            (desc_prelogits, attn_prelogits, _, _, _, _) = self.global_and_local_forward_pass(images)
            desc_logits = self.desc_classification(desc_prelogits)
            desc_loss = compute_loss(labels, desc_logits)
            attn_logits = self.attn_classification(attn_prelogits)
            attn_loss = compute_loss(labels, attn_logits)
            total_loss = desc_loss + attn_loss

        gradients = gradient_tape.gradient(total_loss, self.trainable_weights)
        clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        optimizer.apply_gradients(zip(clipped, self.trainable_weights))
        return {"loss": total_loss}

# Example usage
model = BugReproducer(block3_strides=True, name="bug_reproducer")
dataset = tf.data.Dataset.from_tensor_slices((tf.random.normal([2, 224, 224, 3]), tf.random.uniform([2], maxval=1000, dtype=tf.int64)))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(dataset.batch(2), epochs=1)