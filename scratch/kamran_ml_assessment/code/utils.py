"""
Utility funcions for the BBB dataset and predictions.
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem, Draw
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import learning_curve
from scipy import misc
import matplotlib.pyplot as plt

descriptors = ['MolLogP', 'MolMR', 'ExactMolWt', 'NHOHCount', 'NOCount', 
                'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 
                'NumValenceElectrons', 'RingCount', 'FractionCSP3', 'TPSA', 'LabuteASA']

training_images_dir = "../data/bbb_img_train/"
test_images_dir = "../data/bbb_img_test/"

def preprocess():
    """
    Pre-process Training and Test datasets
    """
    datasets = ["../data/bbb_train.csv", "../data/bbb_test.csv"]
    feature_matrix_files = ["../data/bbb_features_train.csv", "../data/bbb_features_test.csv"]
    img_dirs = [training_images_dir, test_images_dir]
    fingerprint_files = ["../data/bbb_fingerprints_train.txt", "../data/bbb_fingerprints_test.txt"]
    
    for index, dataset in enumerate(datasets):
        print("Preprocessing dataset %d\n" % index)
        data = load_data(dataset)
        print("\tGenerating feature matrix ...")
        data_updated = generate_descriptors(data, descriptors)
        data_updated.to_csv(feature_matrix_files[index])
        if not os.path.exists(img_dirs[index]):
            os.makedirs(img_dirs[index])
        print("\tGenerating Images ...")
        generate_2Dimages(data.SMILES, img_dirs[index])
        print("\tGenerating Fingerprints ...")
        fps = generate_fingerprints(data.SMILES)
        np.savetxt(fingerprint_files[index], fps, fmt="%d")

def load_data(filename):
    """
    Load dataset as pandas dataframe
    """

    df = pd.read_csv(filename)
    return df

def generate_descriptors(data, descriptor_list):
    """
    Generate molecular descripts
    """

    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_list)
    feature_matrix = []
    for i, sm in enumerate(data.SMILES):
        mol = Chem.MolFromSmiles(sm)
        descriptor_values = [v for v in calc.CalcDescriptors(mol)]
        feature_matrix.append(descriptor_values)
    feature_matrix = np.asarray(feature_matrix)
    for index, descriptor in enumerate(descriptor_list):
        data[descriptor] = feature_matrix[:, index]
    return data

def generate_2Dimages(smiles_data, save_dir):
    """
    Generate Images
    """

    size = (200, 100)
    for i, sm in enumerate(smiles_data):
        mol = Chem.MolFromSmiles(sm)
        img = Draw.MolToImage(mol, size=size)
        img.save(save_dir + "%05d.jpeg" % i)
    

def generate_fingerprints(smiles_data):
    """
    Generate Morgan Fingerprints.
    """

    fps = []
    for i, sm in enumerate(smiles_data):
        mol = Chem.MolFromSmiles(sm)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        fp_array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fingerprint, fp_array)
        fps.append(fp_array)
    return np.array(fps)

def generate_representation(rep, subset, data_dir):
    """
    Generates representation for BBB dataset

    Parameters
    ----------
    rep : str
        String corresponding to desired representation of the dataset
        possible values mol_descriptors, fingerprints, images
    
    subset : str
        Subset of data, possible values: train, test

    data_dir : str
        Path for the subdirectory containing data.

    Returns
    -------
    data : pandas.dataframe
        Pandas dataframe corresponding to data representation

    """
    file_prefix = data_dir + "bbb_"
    if rep == "mol_descriptors":
        filename = file_prefix + "features_" + subset + ".csv"
        data = pd.DataFrame.from_csv(filename)
        return data
    elif rep == "fingerprints":
        df = load_data(file_prefix + subset + ".csv")
        filename = file_prefix + "fingerprints_" + subset + ".txt"
        fp_data = np.loadtxt(filename)
        fp_matrix = []
        for i in range(fp_data.shape[0]):
            fp_matrix.append(fp_data[i, :])
        fp_column = pd.Series(fp_matrix)
        data = df.assign(Fingerprints=fp_column.values)
        return data

    elif rep == "images":
        dir_name = file_prefix + "img_" + subset + "/"
        df = load_data(file_prefix + subset + ".csv")
        image_matrix = []
        image_files = [dir_name + "%05d.jpeg" % i for i in df.index.values]
        image_vectors = [misc.imread(im) for im in image_files]
        for index, im in enumerate(image_vectors):
            x, y, z = im.shape
            im = im.reshape(x * y * z)
            image_matrix.append(im)
            #df['img_vector'] = im
        im_column = pd.Series(image_matrix)
        data = df.assign(Images=im_column.values)
        return data
    else:
        raise KeyError("Representation not supported")


def plot_learning_curve(estimator, title, X, y, acceptable_score=None, ylim=None, cv=None,
                        train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Code from: https://jmetzen.github.io/2015-01-29/ml_advice.html
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    acceptable_score : float
        Acceptable model performance

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects
    """
    
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes, scoring="roc_auc")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    if acceptable_score is not None:
        plt.plot([0, plt.xlim()[1]], [acceptable_score, acceptable_score], color='grey', linestyle='--', linewidth=1)
    plt.xlabel("Training examples")
    plt.ylabel("AUC")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.savefig(title + "_learning_curve.png")

def plot_roc_curve(fpr_list, tpr_list, name="test", labels=None):
    plt.figure()
    for i, fpr in enumerate(fpr_list):
        plt.plot(fpr, tpr_list[i], linewidth=2, label=labels[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.legend(loc="best")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(name + "_roc.png")

class NullModel(BaseEstimator):
    """
    Implements a null model for binary classifier performance, where
    binary labels are assigned randomly to the data. 
    """
    def fit(self, X, y=None):
        pass
    
    def predict(self, X):
        p = np.random.randint(2, size=X.shape[0])
        p.reshape(p.shape[0], 1)
        return p

class NeverBBBPositive(BaseEstimator):
    """
    Implements a null model for binary classifier performance, where
    binary labels are assigned randomly to the data. 
    """
    def fit(self, X, y=None):
        pass
    
    def predict(self, X):
        p = np.zeros((len(X), 1), dtype=bool)
        return p

