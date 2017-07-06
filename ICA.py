"""
===================================================
Dimensionality reduction of Eye tracking features
===================================================
-                      ICA                        -
Applies ICA to the data in path:

Inputs:
    path: Folder containing features
    a : Index of the data to preprocess. It can be:
     0 = Initial Calibration
     1 = Final Calibration
     2 = Initial Fixation
     3 = Final Fixation
     4 = Mixing 
     5 = Reading
    tp : Percentage of the testing set over the data
    c : Penalty of the SVM
    N= Number of Principal components to train an SVM

    Outputs:

    svmt : Training time of the SVM
    atr : Training accuracy
    ate : Testing accuracy

"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from time import time
import numpy as np
import pandas as pd
import input_data_txt

def exec_(a, tp, c, n_components, path, clf):
    _data_, _labels_, features_ = input_data_txt.tune(path)

    # Increment if ICA don't coverage
    max_iter=200 
    tol=0.0001

    X=_data_[a]
    y =  _labels_[a]
    features = features_[a]
    # Number of samples
    n_samples = len(X)
    # Number of features
    n_features = len(features)
    # the label to predict is 0 or 1 by now
    #n_classes = len(y.unique())

    #print ("Total dataset size:")
    #print ("n_samples: %d" % n_samples)
    #print ("n_features: %d" % n_features)
    #print ("n_classes: %d" % n_classes)


    ###############################################################################
    # Split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tp)

    ########################################
    #      Standarizing the features       #
    ########################################

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=features)
    X_test = pd.DataFrame(sc.transform(X_test), columns=features)

    ########################################
    #             Applying ICA             #
    ########################################

    from sklearn.decomposition import FastICA

    ica = FastICA(n_components=n_components, max_iter=max_iter, tol=tol)
    X_train = ica.fit_transform(X_train, y_train)
    X_test = ica.transform(X_test)

    ########################################
    #          Classification              #
    ########################################

    import svm_clf
    import lda_clf

    if clf == 'svm':
        svmt, atr, ate = svm_clf.exec_(n_components, c, X_train, y_train, X_test, y_test)
    elif clf == 'lda':
        svmt, atr, ate = lda_clf.exec_(2, X_train, y_train, X_test, y_test)
            
    return svmt, atr, ate


