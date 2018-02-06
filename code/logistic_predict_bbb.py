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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
np.random.seed(46)

PREP = False
if PREP:
    preprocess()

def assess_model(X, Y, data):
    print("Dataset quick summary: number of example = {0[0]}, features = {0[1]}, Balance = {1:.2f}\
        ('%' examples belonging to the class.)".format(X.shape, (np.sum(Y)/X.shape[0])*100))
    X_train, Y_train = X[:750, :], Y[:750]
    print("Training Dataset quick summary: number of example = {0[0]}, features = {0[1]}, Balance = {1:.2f}\
        ('%' examples belonging to the class.)".format(X_train.shape, (np.sum(Y_train)/X_train.shape[0])*100))
    X_test, Y_test = X[750:, :], Y[750:]
    print("Test Dataset quick summary: number of example = {0[0]}, features = {0[1]} Balance = {1:.2f}\
        ('%' examples belonging to the class.)".format(X_test.shape, (np.sum(Y_test)/X_test.shape[0])*100))
    model_scores = []
    null_scores = []
    folds = 10
    for trial in xrange(100):
        print("Model trial: ", trial + 1)
        feature_rep_train, feature_rep_test = shuffle(data[0]), shuffle(data[1])
        X = feature_rep_train.iloc[:, 2:].values
        Y = feature_rep_train.iloc[:, 1].values
        X_train, Y_train = X[:750, :], Y[:750]
        X_test, Y_test = X[750:, :], Y[750:]
        standard_scaler = StandardScaler()
        X_train = standard_scaler.fit_transform(X_train)
        X_test = standard_scaler.transform(X_test)
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        Y_test_pred = model.predict(X_test)
        model_scores.append(roc_auc_score(Y_test, Y_test_pred))
        null_model = NullModel()
        #null_model = NeverBBBPositive()
        #null_pred = cross_val_predict(null_model, X_train, Y_train, cv=folds)
        null_pred = null_model.predict(X_test)
        null_scores.append(roc_auc_score(Y_test, null_pred))

    return model_scores, null_scores
# Obtain datasets
######################################################################################
supported_reps = ["mol_descriptors", "fingerprints"]#, "images"]
datasets = []

for rep in supported_reps:
    for subset in ["train", "test"]:
        data = generate_representation(rep, subset, "../data/")
        datasets.append(data)

feature_rep_train, feature_rep_test = datasets[0], datasets[1]
fingerprint_rep_train, fingerprint_rep_test = datasets[2], datasets[3]
#image_rep_train, image_rep_test = datasets[4], datasets[5]

######################################################################################
# LogisticRegression with MolecularDescriptors
X = feature_rep_train.iloc[:, 2:].values
Y = feature_rep_train.iloc[:, 1].values

model_scores, null_scores = assess_model(X, Y, [datasets[0], datasets[1]])

print("Null Model: ", np.mean(null_scores), np.std(null_scores))

print("Logistc Model with Molecular Descriptors performance: ", np.mean(model_scores), np.std(model_scores))
t_stat, p_value = stats.ttest_ind(model_scores, null_scores)
print(p_value)

X_train, Y_train = X[:750, :], Y[:750]
X_test, Y_test = X[:750, :], Y[:750]
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)

plot_learning_curve(LogisticRegression(), "LogisticRegression_MolecularDescriptors", X_train, Y_train, cv=10)

model = LogisticRegression()
model.fit(X_train, Y_train)
y_test_scores = model.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(Y_test, y_test_scores)
plot_roc_curve([fpr], [tpr], "LogisticRegression", ["MolecularDescriptors"])

# Make predictions on the unknown dataset
X_final_test = feature_rep_test.iloc[:, 1:].values
model = LogisticRegression()
model.fit(X, Y)
predictions = model.predict(X_final_test)
predict_column = pd.Series(list(predictions))
data = feature_rep_test.assign(Predictions=predict_column.values)
data.iloc[:, [0, -1]].to_csv("../predictions/predictions.csv")
######################################################################################
