"""Utility functions for various neural network algorithms
"""
import numpy as np
import h5py


def initialize_weights(layers, scaling_method="constant", scaling_constant=0.01):
    """

    Parameters
    ----------
    layer_dims :
    scaling :
    scaling_constant :

    Returns
    -------

    """
    layer_sizes = [l.n_nodes for l in layers]
    supported_scaling = {"constant": scaling_constant, "xavier": 1.0, "he": 2.0, "custom": None}
    num_layers = len(layer_sizes)

    if scaling_method not in supported_scaling.keys() + ["custom"]:
        raise ValueError(
            "Unsupported scaling %s, please choose from the following: \n%s"
            % (scaling_method, "  ".join(supported_scaling.keys())))

    for l in range(1, num_layers):
        scaling_factor = supported_scaling["constant"]
        if scaling_method not in ["constant", "custom"]:
            scaling_factor = np.sqrt(supported_scaling[scaling_method]/layer_sizes[l])

        if scaling_method == "custom":
            layers[l].weights = np.random.randn(layer_sizes[l], layer_sizes[l - 1])/np.sqrt(layer_sizes[l-1])
            layers[l].biases = np.zeros((layer_sizes[l], 1))
        else:
            layers[l].weights = np.random.randn(layer_sizes[l], layer_sizes[l - 1]) * scaling_factor
            layers[l].biases = np.zeros((layer_sizes[l], 1))



def cross_entropy_loss(y, y_hat):
    """

    Parameters
    ----------
    y :
    probs :

    Returns
    -------

    """
    n_examples = y.shape[1]
    n_classes = y.shape[0]
    if n_classes == 1:
        loss = -np.sum(np.multiply(y, np.log(y_hat)) + np.multiply((1 - y), np.log(1 - y_hat)))
    else:
        loss = -np.sum(y_hat * np.log(y))
    loss /= n_examples
    return loss

def cross_entropy_loss_derivative(y, y_hat):
    dA = -(np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))
    return dA


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
    return a

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
    return a


def sigmoid_prime(z):
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

    sigma_prime = sigmoid(z) * (1 - sigmoid(z))
    return sigma_prime

def relu_prime(z):
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
    sigma_prime = 1.0 * (z > 0)
    return sigma_prime

