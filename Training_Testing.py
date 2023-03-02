"""Training_Testing.py: Facilitates the training and testing of a neural network using the data stored in Train_Data and Test_Data."""

import os
import json

import tensorflow
import numpy

from Data_Collection import folder_names


def main():

    """Controls the operations to be completed. Uncomment all to train and test a model."""

    # train_model()
    # test_model()
    pass


def train_model():

    """These functions represent the historical progression of the training."""

    train_data, train_labels, test_data, test_labels = load_data_set

    # Clean up everything behind the scenes
    tensorflow.keras.backend.clear_session()
    tensorflow.random.set_seed(0)

    # Layer outline
    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Conv1D(64, 18, activation="relu", padding="same", input_shape=(numpy.shape(train_data[0])[0], 1), strides=1),
        tensorflow.keras.layers.MaxPooling1D(1),
        tensorflow.keras.layers.Conv1D(128, 18, activation="relu", padding="same"),
        tensorflow.keras.layers.MaxPooling1D(1),
        tensorflow.keras.layers.Conv1D(128, 18, activation="relu", padding="same"),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(128, activation="relu"),
        tensorflow.keras.layers.Dropout(0.5),
        tensorflow.keras.layers.Dense(100, activation="softmax")
    ])
    print(model.summary())

    # Train it! And save it to a file
    model.compile(optimizer="nadam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_data, train_labels, epochs=10, verbose=1, validation_data=(test_data, test_labels))
    model.save("Model")


def train_model__03_02_2023():

    """Notes: This massively overfit the dataset. It can achieve upwards of 94% accuracy when
    applied to the training data, but demonstrates only 5% accuracy when applied to the test data.
    I need to clean up the dimensions of the input shape and maybe try a smaller model that is
    better suited to the task."""

    train_data, train_labels, test_data, test_labels = load_data_set

    # Clean up everything behind the scenes
    tensorflow.keras.backend.clear_session()
    tensorflow.random.set_seed(0)

    # Layer outline
    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Conv1D(64, 18, activation="relu", padding="same", input_shape=(numpy.shape(train_data[0])[0], 1), strides=1),
        tensorflow.keras.layers.MaxPooling1D(1),
        tensorflow.keras.layers.Conv1D(128, 18, activation="relu", padding="same"),
        tensorflow.keras.layers.MaxPooling1D(1),
        tensorflow.keras.layers.Conv1D(128, 18, activation="relu", padding="same"),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(128, activation="relu"),
        tensorflow.keras.layers.Dropout(0.5),
        tensorflow.keras.layers.Dense(100, activation="softmax")
    ])
    print(model.summary())

    # Train it! And save it to a file
    print("Training model.")
    model.compile(optimizer="nadam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_data, train_labels, epochs=10, verbose=1, validation_data=(test_data, test_labels))
    model.save("Model_03-02-2023")


def load_data_sets():

    """Loads training and test data into useful formats.
    Returns:
        train data, train labels, test data, test labels"""

    train_data, train_labels = load_data_set("train")
    test_data, test_labels = load_data_set("test")

    train_longest = max(train_data, key=lambda x: len(x))
    test_longest = max(test_data, key=lambda x: len(x))
    length = len(train_longest) if len(train_longest) > len(test_longest) else len(test_longest)

    for i in range(len(train_data)):
        item = numpy.array(train_data[i])
        item = numpy.resize(item, length)
        train_data[i] = item

    for i in range(len(test_data)):
        item = numpy.array(test_data[i])
        item = numpy.resize(item, length)
        test_data[i] = item

    return numpy.array(train_data), numpy.array(train_labels), numpy.array(test_data), numpy.array(test_labels)


def load_data_set(data_set):

    """Loads a specific data set.
    Args:
        data_set (str): Which data set to load. "train" or "test". Corresponds to the global dict folder_names in Data_Collection.py
    Returns:
        data, labels (both unpadded)"""

    datas = []
    labels = []

    for token in os.listdir(folder_names[data_set]):
        with open(os.path.join(folder_names[data_set], token), "r") as object:

            data = json.load(object)["tokens"]
            data = data[0] if data else []
            datas.append(data)

            label = int(token[:token.index("_")]) - 1
            labels.append(label)

    return datas, labels


def test_model(model_name="Model"):

    """Assesses the accuracy of a fitted model.
    Args:
        model_name (str): The folder name under which the model was saved. (Default is Model)"""

    # Load the data and the trained model from the filesystem
    train_data, train_labels, test_data, test_labels = load_data_sets()
    model = tensorflow.keras.models.load_model(model_name)

    print("Assessing model accuracy.")

    # Try it on the training data too, to look for overfitting
    results = model.evaluate(train_data, train_labels)
    print("Train data performance:" + "\n" + "Loss: " + str(results[0]) + "\t" "Accuracy: " + str(results[1]))

    # And of course check performance on the test set
    results = model.evaluate(test_data, test_labels)
    print("Test data performance:" + "\n" + "Loss: " + str(results[0]) + "\t" "Accuracy: " + str(results[1]))


if __name__ == "__main__":

    main()