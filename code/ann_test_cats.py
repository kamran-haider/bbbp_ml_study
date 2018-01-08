"""
Train ML models for BBB data and predict 
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ann.ann_utils import _load_test_data
from ann.ann import Network

def train_dnn(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Build and train L layer deep neural network, with L - 1 RELU layers and an output sigmoid layer.    

    Paramaters
    ----------
    X : np.ndarray
        Training Dataset of dimenions (number of examples x features)
    Y : np.ndarray
        Label vector with dimensions (1 x number of examples) for training dataset
    layer_dims : list
        List of layer sizes (including input layer)
    learning_rate : float
        learning rate of the gradient descent update rule
    num_iterations : int
        Number of iterations of the optimization loop
    print_cost : bool
        if True, prints the cost every 100 steps
    
    Returns
    -------
    parameters : 
        Parameters learnt by the model, used for predictions on test set.
    """
    np.random.seed(1)
    m = X.shape[1]
    costs = []
    nn = Network(layers_dims)
    parameters = nn.initialize_params()
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        AL, caches = nn.forward_propagation(X, parameters)        
        # Compute cost.
        cost = nn.compute_cost(AL, Y)
        grads = nn.backward_propagation(AL, Y, caches)
        # Update parameters.
        parameters = nn.update_parameters(parameters, grads, learning_rate)
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    #print("Accuracy: "  + str(np.sum((p == y)/m)))        
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.savefig("test.png")

    return nn

def predict_dnn(parameters, x, score_metric="accuracy"):
    """Calculates accuracy of predictions.
    
    Parameters
    ----------
    network : dict
        Trained ann model parameters
    x : np.ndarray
        Dataset to lable
    y : np.ndarray
        Target values
    
    Returns
    -------
    p : np.ndarray
        Predictions for the dataset
    """
    m = x.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    network = Network([])
    # Forward propagation
    probas, caches = network.forward_propagation(x, parameters)
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    #print("Accuracy: "  + str(np.sum((p == y)/m)))        
    return p

def main():
    # load data
    train_x_orig, train_y, test_x_orig, test_y, classes = _load_test_data("ann/tests/test_datasets/train_catvnoncat.h5", "ann/tests/test_datasets/test_catvnoncat.h5")
    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))
    n_x = 12288

    n_h = 7
    n_y = 1
    #layers_dims = (n_x, n_h, n_y)
    layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
    train_dnn(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)


if __name__ == "__main__":
    main()