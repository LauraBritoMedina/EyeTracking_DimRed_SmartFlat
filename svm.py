"""
===================================================
Dimensionality reduction of Eye tracking features
===================================================
-                      SVM                        -
Uses SVM for classification

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
    
    Outputs:

    svmt : Training time of the SVM
    atr : Training accuracy
    ate : Testing accuracy

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from time import time
import numpy as np
import pandas as pd
import input_data_txt

def exec_(a, tp, c, path):
    _data_, _labels_, features_ = input_data_txt.tune(path)
    X=_data_[a]
    y =  _labels_[a]
    features = features_[a]
    # Number of samples
    n_samples = len(X)
    # Number of features
    n_features = len(features)
    # the label to predict is 0 or 1 by now

    ###############################################################################
    # Split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tp, random_state=42)

    ###############################################################################

    ########################################
    #      Standarizing the features       #
    ########################################

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    #########################################
    #                  SVM                  #
    #########################################
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    svm = SVC(kernel='rbf', C=c)
    t0 = time()
    svm.fit(X_train_std, y_train)
    svmt=time() - t0
    y_pred=svm.predict(X_train_std)
    atr=accuracy_score(y_train, y_pred)
    #print('Accuracy Training set: %.2f' % accuracy_score(y_train, y_pred))
    y_pred=svm.predict(X_test_std)
    ate=accuracy_score(y_test, y_pred)
    #print('Accuracy Testing set: %.2f' % accuracy_score(y_test, y_pred))

    return svmt, atr, ate
    
