import numpy as np
from keras_cv import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

if __name__ == '__main__':
    img_rows, img_cols = 32, 32
    num_classes = 10

    inputs = Input(shape=(img_rows, img_cols, 3))
    x = Conv2D(8, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    model = Model(inputs=inputs, outputs=x)

    X_train = np.random.rand(1000, img_rows, img_cols, 3)
    y_train = np.random.randint(0, num_classes, (1000,))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(X_train, y_train, epochs=1)