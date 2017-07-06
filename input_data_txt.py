"""

===================================================
Input data from tables for classification
===================================================
Returns data from 6 specific action releted moments:

data=[data_C1_, # Initial Calibration
    data_C2_, # Final Calibration
    data_F1_, # Initial Fixation
    data_F2_, # Final Fixation
    data_M_, # Mixing 
    data_R_] # Reading

Same for the labels

    labels= [L_C1_, L_C2_, L_F1_, L_F2_, L_R_, L_M_]
"""
#print (__doc__)

import pandas as pd
import numpy as np
import sys, os
import re
from sklearn.preprocessing import Imputer

#path = 'Features/'

def tune(path):
    
    _C1_ = []
    _C2_ = []
    _F1_ = []
    _F2_ = []
    _M_ = []
    _R_ = []
    L_C1_ = []
    L_C2_ = []
    L_F1_ = []
    L_F2_ = []
    L_M_ = []
    L_R_ = []
    data = []
    labels = []
    features = []
    aux=''
    
    
#######################################################################
##    
##    Uncomment to erase corrupted files:
##
##    dirs_ = os.listdir( path)
##    val = ["No Calibration", "No Fixation", "No Mixing", "No Reading" ]
##    for item in dirs_:
##        file = open(path+item, "r").read()
##        
##        for i in range(len(val)):
##            if val[i] in file:
##                print("Deleting invalid file: ", item)
##                os.remove(path+item, dir_fd=None)
##            else:
##                print(val[i], " Not found in file ")
##    
######################################################################                


    dirs = os.listdir( path)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=1)

    for item in dirs:
        if "C1_features" in item:
            data_ = pd.read_table(path+item, delim_whitespace=True)
            aux = data_.iloc[0]['Patient']
            if "_C" in aux:
                L_C1_.append(0)
            else:
                L_C1_.append(1)
            data_.drop(data_.columns[0],inplace=True,axis=1)
            f0= data_.columns
            _C1_.append(data_)
            data_C1_ = pd.concat(_C1_)
            
        elif "C2_features" in item:
            data_ = pd.read_table(path+item, delim_whitespace=True)
            aux = data_.iloc[0]['Patient']
            if "_C" in aux:
                L_C2_.append(0)
            else:
                L_C2_.append(1)
            data_.drop(data_.columns[0],inplace=True,axis=1)
            f1= data_.columns
            _C2_.append(data_)
            data_C2_ = pd.concat(_C2_)
            
        elif "F1_features" in item:
            data_ = pd.read_table(path+item, delim_whitespace=True)
            aux = data_.iloc[0]['Patient']
            if "_C" in aux:
                L_F1_.append(0)
            else:
                L_F1_.append(1)
            data_.drop(data_.columns[0],inplace=True,axis=1)
            f2= data_.columns
            _F1_.append(data_)
            data_F1_ = pd.concat(_F1_)
                
        elif "F2_features" in item:
            data_ = pd.read_table(path+item, delim_whitespace=True)
            aux = data_.iloc[0]['Patient']
            if "_C" in aux:
                L_F2_.append(0)
            else:
                L_F2_.append(1)
            data_.drop(data_.columns[0],inplace=True,axis=1)
            f3= data_.columns
            _F2_.append(data_)
            data_F2_ = pd.concat(_F2_)
                
        elif "M_features" in item:
            data_ = pd.read_table(path+item, delim_whitespace=True)
            aux = data_.iloc[0]['Patient']
            if "_C" in aux:
                L_M_.append(0)
            else:
                L_M_.append(1)
            data_.drop(data_.columns[0],inplace=True,axis=1)
            f4= data_.columns
            _M_.append(data_)
            data_M_ = pd.concat(_M_)
                
        elif "R_features" in item:
            data_ = pd.read_table(path+item, delim_whitespace=True)
            aux = data_.iloc[0]['Patient']
            if "_C" in aux:
                L_R_.append(0)
            else:
                L_R_.append(1)
            data_.drop(data_.columns[0],inplace=True,axis=1)
            f5= data_.columns
            _R_.append(data_)
            data_R_ = pd.concat(_R_)
            
    imp.fit(data_C1_)
    imp.fit(data_C2_)
    imp.fit(data_F1_)
    imp.fit(data_F2_)
    imp.fit(data_M_)
    imp.fit(data_R_)
    data=[imp.transform(data_C1_),imp.transform(data_C2_),imp.transform(data_F1_),imp.transform(data_F2_),imp.transform(data_M_),imp.transform(data_R_)]
    labels= [L_C1_, L_C2_, L_F1_, L_F2_, L_M_, L_R_]
    features = [f0,f1,f2,f3,f4,f5]

    return data, labels, features

#tune(path)

