import tensorflow as tf
from official-resnet import resnet50_model

def model_fn(image, labels, mode, params):
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    per_worker mirrored strategy requires wrapping the optimizer:
    with strategy.scope():
        optimizer = tf.train.MomentumOptimizer(learning_rate=...)
        
    return resnet50_model-resnet50(image, labels)

def train_step(model, inputs):
    images, labels = inputs
    with tf.GradientTape() as tape:
        loss = model(images, labels)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

if __name__ == "__main__":
    tf.distribute.run(model_fn=model_fn,
                     args=(None, None),
                     strategy=tf.distribute.MultiWorkerMirroredStrategy())