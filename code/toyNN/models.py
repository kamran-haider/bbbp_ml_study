import numpy as np

from layers import *
from utils import *

class BasicDeepModel(object):
    """
    Implements a bare-bones neural network.
    """
    def __init__(self, X, Y, layers, weight_initialization="constant"):

        self.X = X
        self.Y = Y
        self.layers = layers
        initialize_weights(self.layers, scaling_method=weight_initialization)
        self.costs = []

    def forward(self):
        for i in range(1, len(self.layers)):
            current_layer = self.layers[i]
            previous_layer = self.layers[i - 1]
            current_layer.forward_pass(previous_layer.activations)
            if i == len(self.layers) - 1:
                loss = cross_entropy_loss(self.Y, current_layer.activations)
        return loss

    def backward(self):
        for i in reversed(range(0, len(self.layers))):
            current_layer = self.layers[i]
            if i + 1 == len(self.layers):
                current_layer.backward_pass(self.layers[i-1].activations)
            else:
                current_layer.backward_pass(self.layers[i - 1].activations, dZ_next=self.layers[i + 1].dZ, weights_next=self.layers[i + 1].weights)

    def train(self, learning_rate=0.1, n_epochs=10):

        # set activation of the first layer
        self.layers[0].activations = self.X
        for i in range(n_epochs):
            loss = self.forward()
            if i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, np.squeeze(loss)))
                self.costs.append(loss)
            # initialize backprop
            self.layers[-1].dA = cross_entropy_loss_derivative(self.Y, self.layers[-1].activations)
            self.backward()
            # update
            for i in range(1, len(self.layers)):
                current_layer = self.layers[i]
                current_layer.weights -= learning_rate * current_layer.dW
                current_layer.biases -= learning_rate * current_layer.db

