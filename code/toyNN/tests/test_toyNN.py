import os
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import h5py
from toyNN.models import BasicDeepModel
from toyNN.layers import *

def load_test_data(train_data, test_data):
    train_dataset = h5py.File(train_data, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(test_data, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def test_basic_deep():
    """
    A broad integration type test, which trains a BasicDeepModel and check if its costs are
    correct after 200 epochs.
    """
    np.random.seed(1)
    ref_costs = np.array([0.7717493284237686, 0.6720534400822914])
    training_data = os.path.abspath("toyNN/tests/test_datasets/train_catvnoncat.h5")
    test_data = os.path.abspath("toyNN/tests/test_datasets/test_catvnoncat.h5")
    train_x_orig, train_y, test_x_orig, test_y, classes = load_test_data(training_data, test_data)
    num_px = train_x_orig.shape[1]
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    input_layer_nodes = num_px * num_px * 3
    layers = [Input(input_layer_nodes), ReLU(20), ReLU(7), ReLU(5), Sigmoid(1)]

    nn = BasicDeepModel(train_x, train_y, layers, weight_initialization="custom")
    nn.train(learning_rate=0.0075, n_epochs=200)
    assert_array_almost_equal(ref_costs, nn.costs, decimal=8)


