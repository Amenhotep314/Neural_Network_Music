"""Training_Testing.py: Facilitates the training and testing of a neural network using the data stored in Train_Data and Test_Data."""

import os
import json
import random

import tensorflow
import numpy

from Data_Collection import folder_names


def main():

    """Controls the operations to be completed. Uncomment all to train and test a model."""

    # train_model()
    test_model()
    test_model_individually()
    pass


def train_model():

    """These functions represent the historical progression of the training."""

    train_data, train_labels, test_data, test_labels = load_data_sets()

    # Clean up everything behind the scenes
    tensorflow.keras.backend.clear_session()
    tensorflow.random.set_seed(0)

    # Layer outline
    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Conv1D(100, 8, activation="relu", padding="same", input_shape=(numpy.shape(train_data[0])[0], 1), strides=1),
        tensorflow.keras.layers.MaxPooling1D(2),
        tensorflow.keras.layers.Conv1D(200, 8, activation="relu", padding="same"),
        tensorflow.keras.layers.MaxPooling1D(2),
        tensorflow.keras.layers.Conv1D(200, 8, activation="relu", padding="same"),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(100, activation="relu"),
        tensorflow.keras.layers.Dropout(0.5),
        tensorflow.keras.layers.Dense(100, activation="softmax")
    ])

    print("Training model.")
    print(model.summary())

    # Train it! And save it to a file
    model.compile(optimizer="nadam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_data, train_labels, epochs=10, verbose=1, validation_split=1/5)
    model.save("Model")


def train_model__03_04_2023():

    """Notes: The change to the loss function was supremely ineffective. Not only does the model
    guess only one value almost exclusively, but this value is also wrong almost all the time.
    Repeated epochs of training cause accuracy to decrease to fractions of a percent."""

    train_data, train_labels, test_data, test_labels = load_data_sets()

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

    print("Training model.")
    print(model.summary())

    # Train it! And save it to a file
    model.compile(optimizer="nadam", loss="mean_absolute_error", metrics=["accuracy"])
    history = model.fit(train_data, train_labels, epochs=10, verbose=1, validation_split=1/5)
    model.save("Model_03-04-2023")


def train_model__03_02_2023_2():

    """Notes: This model is almost identical to the first iteration, except it has different
    validation behavior. It demonstrates a decreased tendency to overfit, but remains inaccurate.
    The validation behavior should be explored more, the loss function should be changed to mean
    absolute error instead of sparse categorical cross-entropy, and a larger dataset is needed."""

    train_data, train_labels, test_data, test_labels = load_data_sets()

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

    print("Training model.")
    print(model.summary())

    # Train it! And save it to a file
    model.compile(optimizer="nadam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_data, train_labels, epochs=10, verbose=1, validation_split=1/5)
    model.save("Model_03-02-2023-2")


def train_model__03_02_2023_1():

    """Notes: This did not overfit the data set, but managed to fall off the horse the other way
    and include no meaningful information. It only guesses 66%. Disturbingly enough, this is still
    more accurate than the previous iteration."""

    train_data, train_labels, test_data, test_labels = load_data_sets()

    # Clean up everything behind the scenes
    tensorflow.keras.backend.clear_session()
    tensorflow.random.set_seed(0)

    # Layer outline
    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Conv1D(32, 8, activation="relu", padding="same", input_shape=(numpy.shape(train_data[0])[0], 1), strides=1),
        tensorflow.keras.layers.MaxPooling1D(1),
        tensorflow.keras.layers.Conv1D(64, 8, activation="relu", padding="same"),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(64, activation="relu"),
        tensorflow.keras.layers.Dropout(0.5),
        tensorflow.keras.layers.Dense(100, activation="softmax")
    ])

    print("Training model.")
    print(model.summary())

    # Train it! And save it to a file
    model.compile(optimizer="nadam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_data, train_labels, epochs=10, verbose=1, validation_split=1/5)
    model.save("Model_03-02-2023-1")


def train_model__03_02_2023():

    """Notes: This massively overfit the dataset. It can achieve upwards of 94% accuracy when
    applied to the training data, but demonstrates only 5% accuracy when applied to the test data.
    I need to clean up the dimensions of the input shape and maybe try a smaller model that is
    better suited to the task."""

    train_data, train_labels, test_data, test_labels = load_data_sets()

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

    print("Training model.")
    print(model.summary())

    # Train it! And save it to a file
    model.compile(optimizer="nadam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_data, train_labels, epochs=10, verbose=1, validation_data=(test_data, test_labels))
    model.save("Model_03-02-2023")


def test_model(model_name="Model", test_only=False):

    """Assesses the accuracy of a fitted model.
    Args:
        model_name (str): The folder name under which the model was saved. (Default is Model)
        test_only (bool): Should the function try only the test data set, or each in turn? (Default is False)"""

    # Load the data and the trained model from the filesystem
    model = tensorflow.keras.models.load_model(model_name)
    shape = model.layers[0].input_shape[1]
    train_data, train_labels, test_data, test_labels = load_data_sets(shape=shape)

    print("Assessing model accuracy.")

    # Try it on the training data too, to look for overfitting
    if not test_only:
        results = model.evaluate(train_data, train_labels)
        print("Train data performance:\nLoss: " + str(results[0]) + "\tAccuracy: " + str(results[1]))

    # And of course check performance on the test set
    results = model.evaluate(test_data, test_labels)
    print("Test data performance:\nLoss: " + str(results[0]) + "\tAccuracy: " + str(results[1]))


def test_model_individually(model_name="Model", test_only=False):

    """Applies the model individually and verbosely to each element of the data set.
    Args:
        model_name (str): The folder name under which the model was saved. (Default is Model)
        test_only (bool): Should the function try only the test data set, or each in turn? (Default is False)"""

    # Load the data and the trained model from the filesystem
    model = tensorflow.keras.models.load_model(model_name)
    shape = model.layers[0].input_shape[1]
    train_data, train_labels, test_data, test_labels = load_data_sets(shape=shape)

    print("Assessing model accuracy individually.")

    pred_accuracies = []
    rand_accuracies = []

    # Try it on the training data too, to look for overfitting
    randomized = int(round(sum(train_labels) / len(train_labels)))

    if not test_only:
        print("\t".join(["Actual", "Predicted", "Random", "Predicted Error", "Random Error"]))
        for i in range(len(train_data)):
            actual = train_labels[i]
            predicted = numpy.argmax(model.predict(train_data[i][numpy.newaxis], verbose=0)[0])
            pred_accuracy = abs(predicted - actual)
            rand_accuracy = abs(randomized - actual)

            pred_accuracies.append(pred_accuracy)
            print("\t".join([str(actual), str(predicted), str(randomized), str(pred_accuracy), str(rand_accuracy)]))

        # This is pretty simple: add up the percent errors and average them to get an idea of how often it's close
        average_pred = sum(pred_accuracies) / len(pred_accuracies)
        average_rand = sum(rand_accuracies) / len(rand_accuracies)
        print("Train data performance:\nAverage predicted error: " + str(average_pred) + "\tAverage randomized error: " + str(average_rand))
        pred_accuracies = []
        rand_accuracies = []

    # Do this to the test data
    randomized = int(round(sum(test_labels) / len(test_labels)))
    print("\t".join(["Actual", "Predicted", "Random", "Predicted Error", "Random Error"]))

    for i in range(len(test_data)):
        actual = test_labels[i]
        predicted = numpy.argmax(model.predict(test_data[i][numpy.newaxis], verbose=0)[0])
        pred_accuracy = abs(predicted - actual)
        rand_accuracy = abs(randomized - actual)

        pred_accuracies.append(pred_accuracy)
        rand_accuracies.append(rand_accuracy)
        print("\t".join([str(actual), str(predicted), str(randomized), str(pred_accuracy), str(rand_accuracy)]))

    average_pred = sum(pred_accuracies) / len(pred_accuracies)
    average_rand = sum(rand_accuracies) / len(rand_accuracies)
    print("Test data performance:\nAverage predicted error: " + str(average_pred) + "\tAverage randomized error: " + str(average_rand))


def load_data_sets(shape=None):

    """Loads training and test data into useful formats.
    Args:
        shape (int): The length of each numpy array. If None, the shape will be drawn from the data set. Default is None
    Returns:
        train data, train labels, test data, test labels"""

    train_data, train_labels = load_data_set("train")
    test_data, test_labels = load_data_set("test")

    if not shape:
        train_longest = max(train_data, key=lambda x: len(x))
        test_longest = max(test_data, key=lambda x: len(x))
        length = len(train_longest) if len(train_longest) > len(test_longest) else len(test_longest)
    else:
        length = shape

    for i in range(len(train_data)):
        item = numpy.array(train_data[i])
        item = numpy.resize(item, length)
        train_data[i] = item

    for i in range(len(test_data)):
        item = numpy.array(test_data[i])
        item = numpy.resize(item, length)
        test_data[i] = item

    train_data = numpy.array(train_data)
    train_data = numpy.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))
    train_labels = numpy.array(train_labels)
    test_data = numpy.array(test_data)
    test_data = numpy.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
    test_labels = numpy.array(test_labels)

    return train_data, train_labels, test_data, test_labels


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


if __name__ == "__main__":

    main()