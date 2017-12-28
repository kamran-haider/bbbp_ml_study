"""
Utility functions for ann module.
"""
import numpy as np
import h5py


def sigmoid(z):
    """Calculate activation of nodes in a layer.

    Parameters
    ----------
    z : np.ndarray
        vector of inputs

    Returns
    -------
    a : np.ndarray
        sigmoid activations
    cache : np.ndarray
        weighted inputs for caching purpose
    """

    a = 1 / (1 + np.exp(-z))
    cache = z
    return a, cache


def relu(z):
    """

    z: np.ndarray
        vector of inputs

    Returns
    -------
    a : np.ndarray
        RELU activations
    cache : np.ndarray
        weighted inputs for caching purpose
    """
    a = np.maximum(0, z)
    cache = z
    return a, cache


def relu_backward(dA, cache):
    """
    Calculate gradient of the cost with respect to relu activation function.

    Parameters
    ----------
    dA: np.ndarray
        Gradient of the cost function with respect to activations
    cache: np.ndarray
        Cached value of Z

    Returns
    -------
    dZ: np.ndarray
        Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Calculate gradient of the cost with respect to sigmoid activation function.

    Parameters
    ----------
    dA: np.ndarray
        Gradient of the cost function with respect to activations
    cache: np.ndarray
        Cached value of Z

    Returns
    -------
    dZ: np.ndarray
        Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)

    return dZ


def _load_test_data():
    train_dataset = h5py.File('tests/test_datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('tests/test_datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes