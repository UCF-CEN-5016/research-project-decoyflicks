import numpy as np
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
from keras.layers import Dense

def build_classifier() -> Sequential:
    model = Sequential()
    model.add(Dense(10, activation='softmax'))
    return model

def prepare_labels() -> np.ndarray:
    return np.array([[1, 0, 0], [0, 1, 0]])

def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return sparse_categorical_crossentropy(y_true, y_pred)

def main() -> None:
    classifier = build_classifier()
    labels = prepare_labels()
    loss_values = compute_loss(labels, labels)
    print(loss_values)

if __name__ == "__main__":
    main()