"""
===================================================
Dimensionality reduction of Eye tracking features
===================================================
-                 Main Script                     -

Applies all the dimensionality reduction methods to
the data in path, and classifies the data with a
specific classifier:

Inputs:
    path: Folder containing features
    clf : Index of the data to preprocess. It can be:
     svm = Initial Calibration
     lda = Final Calibration
     
    Outputs:
    ------------------------
"""

path = 'Features/'
pathR = 'Results/'
clf = 'lda'

import Main_ICA
import Main_PCA
import Main_RF

Main_PCA.exec_(path, pathR, clf)
Main_RF.exec_(path, pathR, clf)
Main_ICA.exec_(path, pathR, clf)

