"""
===================================================
Dimensionality reduction of Eye tracking features
===================================================
-                  LDA classifier                 -
Uses LDA for classification

Inputs:
    n_classes : Number of categories to classify
    X_train : Training set
    y_train : Training lables
    X_test : Testing set
    y_test : Testing lables
    
    Outputs:

    ldat : Training time of the LDA
    atr : Training accuracy
    ate : Testing accuracy

"""

from time import time
import numpy as np
import pandas as pd

def exec_(n_classes, X_train, y_train, X_test, y_test):
   
    #########################################
    #                  LDA                  #
    #########################################
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score

    lda = LinearDiscriminantAnalysis(n_components=n_classes)
    t0 = time()
    lda.fit_transform(X_train, y_train)
    ldat=time() - t0
    y_pred=lda.predict(X_train)
    atr=accuracy_score(y_train, y_pred)
    #print('Accuracy Training set: %.2f' % accuracy_score(y_train, y_pred))
    y_pred=lda.predict(X_test)
    ate=accuracy_score(y_test, y_pred)
    #print('Accuracy Testing set: %.2f' % accuracy_score(y_test, y_pred))

    return ldat, atr, ate
    
