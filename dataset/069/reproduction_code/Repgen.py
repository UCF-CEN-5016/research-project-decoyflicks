import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import seaborn as sns

# Read the TSV data from 'FordA_TRAIN.tsv' and 'FordA_TEST.tsv' using the readucr function
def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

# Visualize one timeseries example for each class in the dataset using matplotlib's plot function
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.show()
plt.close()

# Standardize the data by reshaping x_train and x_test to add a channel dimension, resulting in shapes (3601, 500, 1) and (1320, 500, 1)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Define num_classes as the number of unique classes found in y_train
num_classes = len(np.unique(y_train))

# Shuffle the training set using numpy's random.permutation function
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

# Standardize the labels to positive integers by replacing -1 with 0
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

# Define a make_model function that constructs the CNN architecture as described in the example code
def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

# Build a Fully Convolutional Neural Network model with input shape (500, 1)
model = make_model(input_shape=x_train.shape[1:])

# Compile the model using 'adam' optimizer and 'sparse_categorical_crossentropy' loss function
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])

# Train the model on x_train and y_train for 200 epochs with batch size of 32, validation split of 0.2, and appropriate callbacks
history = model.fit(x_train, y_train, batch_size=32, epochs=200, validation_split=0.2)

# Evaluate the trained model on x_test and y_test to obtain test_loss and test_acc
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy", test_acc)
print("Test loss", test_loss)

# Plot the model's training and validation accuracy using matplotlib's plot function
plt.figure()
plt.plot(history.history["sparse_categorical_accuracy"])
plt.plot(history.history["val_sparse_categorical_accuracy"])
plt.title("model sparse_categorical_accuracy")
plt.ylabel("sparse_categorical_accuracy", fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()

# Create a DataFrame with the correlation matrix
data = pd.DataFrame(np.random.rand(10, 5))  # Replace with actual data that includes string values
df = pd.DataFrame(data.corr())

def show_heatmap(df):
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# Call the show_heatmap function from timeseries_weather_forecasting.py with the dataframe df
show_heatmap(df)
