"""
===================================================
Dimensionality reduction of Eye tracking features
===================================================
-                      PCA                        -
Applies PCA to the data in path:
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

#print (__doc__)

import input_data_txt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA

# Obtain the data from the tables

def exec_(a, tp, c, N, path, clf):
    _data_, _labels_, features_ = input_data_txt.tune(path)
    
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tp)#



    ########################################
    #      Standarizing the features       #
    ########################################

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=features)
    X_test = pd.DataFrame(sc.transform(X_test), columns=features)

    #######################################################################
    # Compute a PCA on the given dataset (treated as unlabeled            #
    # dataset): unsupervised feature extraction / dimensionality reduction#
    #######################################################################

    n_components = N

    #######################################
    #  Function to apply PCA eigenvalues  #
    #######################################

    def doPCA(n,data):
        from sklearn.decomposition import PCA
        pca = PCA(n)
        pca.fit(data)
        return pca

   #print ("Extracting the top %d features from %d eye tracking features" % (n_components, X_train.shape[0]))
    #t0 = time()
    ### Fit the model to the training data
    pca = doPCA(n_components, X_train)
    ##pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    #print ("done in %0.3fs" % (time() - t0))
    #print("Explained Variance Ratio:  ", pca.explained_variance_ratio_*100, '%')
    #print("Principal Components: ", pca.components_)

    #first_pc = pca.components_[0]
    #second_pc = pca.components_[1]
    ### Apply dimensionality reduction to X_train

    X_train = pca.transform(X_train)

    X_test = pca.transform(X_test)


    ########################################
    #          Classification              #
    ########################################

    import svm_clf
    import lda_clf

    if clf == 'svm':
        svmt, atr, ate = svm_clf.exec_(n_components, c, X_train, y_train, X_test, y_test)
    elif clf == 'lda':
        svmt, atr, ate = lda_clf.exec_(n_components, X_train, y_train, X_test, y_test)
            
    return svmt, atr, ate
