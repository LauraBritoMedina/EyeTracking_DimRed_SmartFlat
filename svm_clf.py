"""
===================================================
Dimensionality reduction of Eye tracking features
===================================================
-                  SVM classifier                 -
Uses LDA for classification

Inputs:
    c : Penalty of the classifier
    X_train : Training set
    y_train : Training lables
    X_test : Testing set
    y_test : Testing lables
    
    Outputs:

    svmt : Training time of the LDA
    atr : Training accuracy
    ate : Testing accuracy

"""

from time import time
import numpy as np
import pandas as pd

def exec_(c, X_train, y_train, X_test, y_test):
   
    #########################################
    #                  SVM                  #
    #########################################
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    svm = SVC(kernel='rbf', C=c)
    t0 = time()
    svm.fit(X_train, y_train)
    svmt=time() - t0
    #print ("SVM Training done in %0.3fs" % (time() - t0))
    y_pred=svm.predict(X_train)
    #print('Training set:')
    atr=accuracy_score(y_train, y_pred)
    #print('Accuracy of', n_components, 'principal components: %.2f' % accuracy_score(y_train, y_pred))
    y_pred=svm.predict(X_test)
    #print('Testing set:')
    ate=accuracy_score(y_test, y_pred)
    #print('Accuracy of', n_components, 'principal components: %.2f' % accuracy_score(y_test, y_pred))

    return svmt, atr, ate
    
