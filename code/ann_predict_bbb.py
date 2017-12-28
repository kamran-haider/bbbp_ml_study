"""
Train ML models for BBB data and predict 
"""
from __future__ import division, print_function, unicode_literals
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from scipy import stats

from ann.ann import Network

PREP = False
if PREP:
    preprocess()


supported_reps = ["mol_descriptors", "fingerprints", "images"]
datasets = []
for rep in supported_reps:
    for subset in ["train", "test"]:
        data = generate_representation(rep, subset, "../data/")
        datasets.append(data)

feature_rep_train, feature_rep_test = datasets[0], datasets[1]
fingerprint_rep_train, fingerprint_rep_test = datasets[2], datasets[3]
image_rep_train, image_rep_test = datasets[4], datasets[5]


#Image data
X = np.array([i/255. for i in image_rep_train.iloc[:, -1].values])
X = X.reshape(X.shape[1], X.shape[0])
Y = image_rep_train.iloc[:, 1].values
Y = Y.reshape([1, Y.shape[0]])

print("Dataset quick summary: number of example = {0[1]}, features = {0[0]}, Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(X.shape, (np.sum(Y)/X.shape[1])*100))
x_train, y_train = X[:, :750], Y[:, :750]
print("Training Dataset quick summary: number of example = {0[1]}, features = {0[0]}, Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(x_train.shape, (np.sum(y_train)/x_train.shape[1])*100))
x_test, y_test = X[:, 750:], Y[:, 750:]
print("Test Dataset quick summary: number of example = {0[0]}, features = {0[1]} Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(x_test.shape, (np.sum(y_test)/x_test.shape[0])*100))

def train_dnn(X, Y, x_test, y_test, layers_dims, learning_rate=0.75, num_iterations=3000, print_cost=False):
    """
    Build and train L layer deep neural network, with L - 1 RELU layers and an output sigmoid layer.    

    Paramaters
    ----------
    X : np.ndarray
        Training Dataset of dimenions (number of examples x features)
    Y : np.ndarray
        Label vector with dimensions (1 x number of examples) for training dataset
    X : np.ndarray
        Test Dataset of dimenions (number of examples x features)
    Y : np.ndarray
        Label vector with dimensions (1 x number of examples) for test dataset
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
    m, n = X.shape[1], x_test.shape[1]
    costs = []
    nn = Network(layers_dims)
    parameters = nn.parameters
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        AL, caches = nn.forward_propagation(X, parameters)        
        # Compute cost.
        cost = nn.compute_cost(AL, Y)
        grads = nn.backward_propagation(AL, Y, caches)
        # Update parameters.
        parameters = nn.update_parameters(parameters, grads, learning_rate)
        # Print the cost every 100 training example
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
            p_train = predict_dnn(parameters, X) 
            print("Training Accuracy: "  + str(np.sum((p_train[0, :] == Y[0, :])/m)))        
            p_test = predict_dnn(parameters, x_test)
            print("Test Accuracy: "  + str(np.sum((p_test[0, :] == y_test[0, :])/n)))        
        if print_cost and i % 10 == 0:
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

layers = [x_train.shape[0], 20, 7, 5, 1]
model_params = train_dnn(x_train, y_train, x_test, y_test, layers, num_iterations=100, print_cost=True)
