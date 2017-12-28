"""
Train ML models for BBB data and predict 
"""
from __future__ import division, print_function, unicode_literals
from utils import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from scipy import stats

from ann import Network

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
"""

X = feature_rep_train.iloc[:, 2:].values
Y = feature_rep_train.iloc[:, 1].values
print("Dataset quick summary: number of example = {0[0]}, features = {0[1]}, Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(X.shape, (np.sum(Y)/X.shape[0])*100))
X_train, Y_train = X[:750, :], Y[:750].reshape(Y[:750].shape[0], 1)
print("Training Dataset quick summary: number of example = {0[0]}, features = {0[1]}, Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(X_train.shape, (np.sum(Y_train)/X_train.shape[0])*100))
X_test, Y_test = X[750:, :], Y[750:]
print("Test Dataset quick summary: number of example = {0[0]}, features = {0[1]} Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(X_test.shape, (np.sum(Y_test)/X_test.shape[0])*100))



model = Sequential()
model.add(Dense(X_train.shape[1], input_shape=(X_train.shape[1], )))
dropout = 0.5
hidden_dim = 80
for i in range(3):
    model.add(Dense(output_dim=hidden_dim, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout))
model.add(Dense(Y_train.shape[1]))
model.add(Activation('sigmoid'))
opt = SGD(lr=0.005)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
history = model.fit(X_train, Y_train, epochs=1000, batch_size=50)
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))
"""

################################################################################################################
#Image data
X = np.array([i for i in image_rep_train.iloc[:, -1].values])
Y = image_rep_train.iloc[:, 1].values
Y = Y.reshape([Y.shape[0], 1])
print("Dataset quick summary: number of example = {0[0]}, features = {0[1]}, Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(X.shape, (np.sum(Y)/X.shape[0])*100))
x_train, y_train = X[:750, :], Y[:750]
print("Training Dataset quick summary: number of example = {0[0]}, features = {0[1]}, Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(x_train.shape, (np.sum(y_train)/x_train.shape[0])*100))
x_test, y_test = X[750:, :], Y[750:]
print("Test Dataset quick summary: number of example = {0[0]}, features = {0[1]} Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(x_test.shape, (np.sum(y_test)/x_test.shape[0])*100))

def train_dnn(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Build and train L-layer deep neural network, with L - 1 RELU layers and an output sigmoid layer.    

    Paramaters
    ----------
    X : np.ndarray
        Training Dataset of dimenions (number of examples x features)
    Y : np.ndarray
        Label vector with dimensions (1 x number of examples)
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

    costs = []
    model = Network(layers_dims)
    """
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters    
    """
layers = [x_train.shape]
print(layers)
"""
################################################################################################################
#FP data
X = np.array([i for i in fingerprint_rep_train.iloc[:, -1].values])
Y = fingerprint_rep_train.iloc[:, 1].values
Y = Y.reshape([Y.shape[0], 1])

print("Dataset quick summary: number of example = {0[0]}, features = {0[1]}, Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(X.shape, (np.sum(Y)/X.shape[0])*100))
x_train, y_train = X[:750, :], Y[:750]
print("Training Dataset quick summary: number of example = {0[0]}, features = {0[1]}, Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(x_train.shape, (np.sum(y_train)/x_train.shape[0])*100))
x_test, y_test = X[750:, :], Y[750:]
print("Test Dataset quick summary: number of example = {0[0]}, features = {0[1]} Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(x_test.shape, (np.sum(y_test)/x_test.shape[0])*100))

model = Sequential()
model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
print(score)
#image_rep_test
"""
