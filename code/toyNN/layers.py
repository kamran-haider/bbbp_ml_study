"""Various layers for neural network algorithms
"""
import numpy as np
import random
import sys

from utils import *


class Sigmoid(object):
    """
    Implements a layer with a sigmoid activation function
    """
    def __init__(self, n_nodes, activations=None, weights=None, biases=None):
        """

        """
        self.n_nodes = n_nodes
        self.activations = activations
        self.weights = weights
        self.biases = biases
        self.weighted_inputs = None
        self.dA = None
        self.dW = None
        self.db = None
        self.dZ = None

    def forward_pass(self, a_previous):
        """

        Parameters
        ----------
        a_previous :

        Returns
        -------

        """
        z = np.dot(self.weights, a_previous) + self.biases
        activations = sigmoid(z)
        self.weighted_inputs = z
        self.activations = activations
        return activations

    def backward_pass(self, activations_previous, dZ_next=None, weights_next=None):
        """

        Parameters
        ----------
        delta :

        Returns
        -------

        """
        m = activations_previous.shape[1]
        if dZ_next is not None and weights_next is not None:
            # hidden layers, calculate dA, dz, dW, db
            self.dA = np.dot(weights_next.T, dZ_next)
            self.dZ = self.dA * sigmoid_prime(self.weighted_inputs)
            self.dW = (1. / self.n_nodes) * np.dot(self.dZ, activations_previous.T)
            self.db = (1. / self.n_nodes) * np.sum(self.dZ, axis=1, keepdims=True)
            #print(db)
        else:
            # output layer, dA is already calculated
            dZ = self.dA * sigmoid_prime(self.weighted_inputs)
            self.dZ = dZ
            self.dW = (1. / m) * np.dot(dZ, activations_previous.T)
            self.db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)


class ReLU(object):
    """
    Implements a layer with a ReLU activation function
    """
    def __init__(self, n_nodes, activations=None, weights=None, biases=None):
        """

        """
        self.n_nodes = n_nodes
        self.activations = activations
        self.weights = weights
        self.biases = biases
        self.weighted_inputs = None
        self.dA = None
        self.dW = None
        self.db = None
        self.dZ = None

    def forward_pass(self, a_previous):
        """

        Parameters
        ----------
        a_previous :

        Returns
        -------

        """
        z = np.dot(self.weights, a_previous) + self.biases
        activations = np.maximum(0, z)
        self.weighted_inputs = z
        self.activations = activations
        return activations

    def backward_pass(self, activations_previous, dZ_next=None, weights_next=None):
        """

        Parameters
        ----------
        delta_previous :

        Returns
        -------

        """
        m = activations_previous.shape[1]
        if dZ_next is not None and weights_next is not None:
            # hidden layers, calculate dA, dz, dW, db
            self.dA = np.dot(weights_next.T, dZ_next)
            self.dZ = self.dA * relu_prime(self.weighted_inputs)
            self.dW = (1. / m) * np.dot(self.dZ, activations_previous.T)
            #print(dW)
            self.db = (1. / m) * np.sum(self.dZ, axis=1, keepdims=True)
            #print(db)
        else:
            # output layer, dA is already calculated
            dZ = self.dA * relu_prime(self.weighted_inputs)
            self.dZ = dZ
            self.dW = (1. / m) * np.dot(dZ, activations_previous.T)
            self.db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)

class Input(object):
    """
    Implements a layer class for the input to maintain consistency in layers are dealth with.
    """

    def __init__(self, n_nodes, activations=None):

        self.n_nodes = n_nodes
        self.activations = activations

    def forward_pass(self, a_previous):
        """

        Parameters
        ----------
        a_previous :

        Returns
        -------

        """
        pass

    def backward_pass(self, activations_previous, dZ_next=None, weights_next=None):
        """

        Parameters
        ----------
        delta :

        Returns
        -------

        """
        pass

