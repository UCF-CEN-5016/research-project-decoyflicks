import tensorflow as tf
from official.vision.beta.modeling.backbones import efficientnet
from official.vision.beta.modeling.decoders import fpn

def create_efficientdet_d1_model():
    backbone = efficientnet.EfficientNet(model_id='efficientnet-b1')
    input_tensor = tf.keras.layers.Input(shape=(640, 640, 3))
    features = backbone(input_tensor)
    
    fpn_features = fpn.FPN(min_level=3, max_level=7)(features)
    
    classification_head = tf.keras.layers.Conv2D(90, 3, padding='same')(fpn_features[0])
    regression_head = tf.keras.layers.Conv2D(4, 3, padding='same')(fpn_features[0])
    
    model = tf.keras.Model(inputs=input_tensor, outputs=[classification_head, regression_head])
    return model

def run_training_demonstration():
    model = create_efficientdet_d1_model()

    input_data = tf.random.normal((2, 640, 640, 3))
    target_class_output = tf.random.normal((2, 80, 80, 90))
    target_reg_output = tf.random.normal((2, 80, 80, 4))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    print("Training the model - watch for missing gradient warnings")
    model.fit(input_data, [target_class_output, target_reg_output], epochs=1, verbose=1)

    print("\nVariables that should be missing gradients:")
    for var_name in ['stack_6/block_1/expand_bn/gamma:0', 'top_bn/beta:0']:
        print(f" - {var_name}")

if __name__ == "__main__":
    run_training_demonstration()
