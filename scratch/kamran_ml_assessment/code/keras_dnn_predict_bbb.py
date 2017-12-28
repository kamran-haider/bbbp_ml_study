"""
Train ML models for BBB data and predict 
"""
from __future__ import division, print_function, unicode_literals
from utils import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import SGD
from scipy import stats
np.random.seed(46)


PREP = False
if PREP:
    preprocess()


supported_reps = ["mol_descriptors", "fingerprints"]#, "images"]
datasets = []
for rep in supported_reps:
    for subset in ["train", "test"]:
        data = generate_representation(rep, subset, "../data/")
        datasets.append(data)

feature_rep_train, feature_rep_test = datasets[0], datasets[1]
fingerprint_rep_train, fingerprint_rep_test = datasets[2], datasets[3]
#image_rep_train, image_rep_test = datasets[4], datasets[5]

################################################################################################################
# Descriptor data


X = feature_rep_train.iloc[:, 2:].values
Y = feature_rep_train.iloc[:, 1].values

standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)


print("Dataset quick summary: number of example = {0[0]}, features = {0[1]}, Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(X.shape, (np.sum(Y)/X.shape[0])*100))
x_train, y_train = X[:750, :], Y[:750]
print("Training Dataset quick summary: number of example = {0[0]}, features = {0[1]}, Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(x_train.shape, (np.sum(y_train)/x_train.shape[0])*100))
x_test, y_test = X[750:, :], Y[750:]
print("Test Dataset quick summary: number of example = {0[0]}, features = {0[1]} Balance = {1:.2f}\
    ('%' examples belonging to the class.)".format(x_test.shape, (np.sum(y_test)/x_test.shape[0])*100))

kfold = StratifiedKFold(n_splits=10, shuffle=True)
scores = []
for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(X[train], Y[train], epochs=1000, batch_size=128)
    score = model.predict_proba(X[test])
    scores.append((roc_auc_score(Y[test], score)))

print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))

fpr, tpr, thresholds = roc_curve(Y[test], score)
plot_roc_curve([fpr], [tpr], "ANNDescriptors", ["Descriptors"])

# Make predictions on the unknown dataset
X_final_test = feature_rep_test.iloc[:, 1:].values

standard_scaler = StandardScaler()
X_final_test = standard_scaler.fit_transform(X_final_test)


predictions = model.predict_classes(X_final_test)
predict_column = pd.Series(list(np.squeeze(predictions)))
data = feature_rep_test.assign(Predictions=predict_column.values)
data.iloc[:, [0, -1]].to_csv("../predictions/predictions_ann_desc.csv")

"""
################################################################################################################
#Image data
X = np.array([i/255. for i in image_rep_train.iloc[:, -1].values])
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

kfold = StratifiedKFold(n_splits=10, shuffle=True)
scores = []
for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(X[train], Y[train], epochs=10, batch_size=128)
    score = model.predict_proba(X[test])
    scores.append((roc_auc_score(Y[test], score)))

print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))

fpr, tpr, thresholds = roc_curve(Y[test], score)
plot_roc_curve([fpr], [tpr], "ANN2DImages", ["2DImages"])

# Make predictions on the unknown dataset
X_final_test = np.array([i/255. for i in image_rep_test.iloc[:, -1].values])

model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])
model.fit(X, Y, epochs=100, batch_size=128)
predictions = model.predict_classes(X_final_test)
predict_column = pd.Series(list(predictions))
data = feature_rep_test.assign(Predictions=predict_column.values)
data.iloc[:, [0, -1]].to_csv("../predictions/predictions_ann_img.csv")
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

kfold = StratifiedKFold(n_splits=10, shuffle=True)
scores = []
for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(X[train], Y[train], epochs=1000, batch_size=128)
    score = model.predict_proba(X[test])
    scores.append((roc_auc_score(Y[test], score)))

print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))

fpr, tpr, thresholds = roc_curve(Y[test], score)
plot_roc_curve([fpr], [tpr], "ANNFingerPrints", ["Fingerprints"])

# Make predictions on the unknown dataset
#X_final_test = fingerprint_rep_test.iloc[:, 1:].values
X_final_test = np.array([i for i in fingerprint_rep_test.iloc[:, -1].values])
predictions = model.predict_classes(X_final_test)
predict_column = pd.Series(list(np.squeeze(predictions)))
data = feature_rep_test.assign(Predictions=predict_column.values)
data.iloc[:, [0, -1]].to_csv("../predictions/predictions_ann_fp.csv")
