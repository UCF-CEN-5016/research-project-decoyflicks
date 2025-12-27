from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# Generate some random data
X = [[1, 2], [3, 4], [5, 6]]
y = [0, 1, 1]

# Split the data into training and test sets
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.5)

# Create a simple neural network model
model = Sequential()
model.add(Dense(2, input_dim=2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# Train the model using the training data and validation set as test set
model.fit(train_X, epochs=20, validation_data=(val_X, val_y))