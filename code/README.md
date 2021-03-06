## How to Reproduce this work:
You can reproduce the predictions supplied with this work using two python programs. These programs have the folowing dependencies:
* Numpy
* Scipy
* Scikit-learn
* Keras
* RDKit (Only needed if data representations also need to be reproduced).

## Commands
These programs will work seemlessly with an Anaconda Python distribution. You can run these inside a new conda enviornment, if you wish not to install the dependencies in your main Python installation.

```
python logistic_predict_bbb.py
```
This commmand will generate file: `../predictions/predictions_logistic.csv` which contain results for logistic regression classifier for the test dataset. In addition, it will also create figures in `../notes/slides.pptx` slide # 6, 7.  

```
python keras_dnn_predict_bbb.py
```
This commmand will generate files: `../predictions/predictions_ann_fp.csv` and ``../predictions/predictions_ann_desc.csv` which contain results for artificial neural network classifier for the test dataset using molecular descriptors and morgan fingerprints. In addition, it will also create figures in `../notes/slides.pptx` slide # 9.
