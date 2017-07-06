"""

===================================================
Random Forest
===================================================
Applies random forest to the data in path:
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
    N : Number of most important features to process

Outputs:

    svmt : Training time of the SVM
    atr : Training accuracy
    ate : Testing accuracy

"""

#print (__doc__)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import input_data_txt
from time import time


def exec_(a, tp, c, N, path, clf):
# Obtain the data from the tables
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tp)

    ###############################################################################

    ########################################
    #      Standarizing the features       #
    ########################################

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=features)
    X_test = pd.DataFrame(sc.transform(X_test), columns=features)

    ########################################
    #        Applying Random Forest        #
    ########################################


    feat_labels = features
    forest = RandomForestClassifier(n_estimators=10000,
                                    random_state=10,
                                    n_jobs=-1)
    forest.fit(X_train,y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    #for f in range(X_train.shape[1]):
    #    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[f],importances[indices[f]]*100))

    plt.title('Feature Relative Importance')
    plt.bar(range(X_train.shape[1]),
            importances[indices],
            color='grey',
            align='center')

    plt.xticks(range(X_train.shape[1]),
               feat_labels, rotation=90)
    plt.xlim([-1, X_train.shape[1]])  
    plt.tight_layout()
    plt.show()


    ########################################
    #          Classification              #
    ########################################

    if clf == 'svm':
        ###############################################
        # Training SVM with N most important features #
        ###############################################

        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score

        idx=indices[:N] #Index
        svm = SVC(kernel='rbf', C=c)
        t0 = time()
        svm.fit(X_train.iloc[:, idx], y_train)
        svmt=time() - t0
        #print ("SVM Training done in %0.9fs" % svmt)
        y_pred=svm.predict(X_train.iloc[:,idx])
        #print('Training set:')
        atr=accuracy_score(y_train, y_pred)
        #print('Accuracy of', N, 'features: %.2f' % atr)
        y_pred=svm.predict(X_test.iloc[:,idx])
        #print('Testing set:')
        ate=accuracy_score(y_test, y_pred)
        #print('Accuracy of', N, 'features: %.2f' % ate)

        return svmt, atr, ate
    
    elif clf == 'lda':
        ###############################################
        # Training SVM with N most important features #
        ###############################################
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import accuracy_score

        idx=indices[:N] #Index
        lda = LinearDiscriminantAnalysis(n_components=2)
        t0 = time()
        lda.fit_transform(X_train.iloc[:, idx], y_train)
        svmt=time() - t0
        #print ("SVM Training done in %0.9fs" % svmt)
        y_pred=lda.predict(X_train.iloc[:,idx])
        #print('Training set:')
        atr=accuracy_score(y_train, y_pred)
        #print('Accuracy of', N, 'features: %.2f' % atr)
        y_pred=lda.predict(X_test.iloc[:,idx])
        #print('Testing set:')
        ate=accuracy_score(y_test, y_pred)
        #print('Accuracy of', N, 'features: %.2f' % ate)

        return svmt, atr, ate
            

