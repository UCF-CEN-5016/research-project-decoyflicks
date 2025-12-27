from keras.utils import FeatureSpace

def load_data():
    x_train = np.random.rand(1000, 20)
    y_train = np.random.randint(0, 2, (1000, 1))
    x_val = np.random.rand(200, 20)
    y_val = np.random.randint(0, 2, (200, 1))
    return x_train, y_train, x_val, y_val

x_train, y_train, x_val, y_val = load_data()

model = keras.Sequential([
    layers.Input(shape=(20,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=5)