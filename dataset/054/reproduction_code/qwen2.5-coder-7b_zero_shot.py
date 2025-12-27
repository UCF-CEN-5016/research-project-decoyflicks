from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any


def prepare_dataset() -> Tuple[List[List[int]], List[int]]:
    """
    Prepare the dataset.
    Returns:
        features: List of feature vectors.
        labels: Corresponding list of labels.
    """
    features = [[1, 2], [3, 4], [5, 6]]
    labels = [0, 1, 1]
    return features, labels


def split_data(features: List[List[int]], labels: List[int], test_size: float = 0.5
               ) -> Tuple[List[List[int]], List[List[int]], List[int], List[int]]:
    """
    Split features and labels into training and validation sets.
    """
    return train_test_split(features, labels, test_size=test_size)


def build_simple_model(input_dim: int = 2, units: int = 2, activation: str = 'softmax',
                       loss: str = 'binary_crossentropy', optimizer: str = 'adam') -> Sequential:
    """
    Construct and compile a simple Sequential model.
    """
    model = Sequential()
    model.add(Dense(units, input_dim=input_dim, activation=activation))
    model.compile(loss=loss, optimizer=optimizer)
    return model


def train_model(model: Sequential, x_train: Any, x_val: Any, y_val: Any, epochs: int = 20) -> None:
    """
    Train the model using provided training data and validation data.
    Note: The training call mirrors the original logic and uses x_train as the first positional argument.
    """
    model.fit(x_train, epochs=epochs, validation_data=(x_val, y_val))


if __name__ == '__main__':
    features, labels = prepare_dataset()
    x_train, x_val, y_train, y_val = split_data(features, labels, test_size=0.5)

    model = build_simple_model(input_dim=2, units=2, activation='softmax',
                               loss='binary_crossentropy', optimizer='adam')

    train_model(model, x_train, x_val, y_val, epochs=20)