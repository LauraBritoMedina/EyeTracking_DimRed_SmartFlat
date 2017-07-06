from xlsxwriter.workbook import Workbook
import Random_Forest
import pandas as pd
import numpy as np


def exec_(path, pathR, clf):
    
    tp = [0.25, 0.15, 0.15, 0.15, 0.15, 0.2] #testing set percentage
    c= [8, 1.5, 1, 1, 8, 2] #Penalty of svm
    N=15#Number of most important index

    Train_time_ = np.empty((6,0), float)
    Train_acc_ = np.empty((6,0), float)
    Test_acc_ = np.empty((6,0), float)
    name='Random_Forest_Results.xlsx'

    for a in range(0,6):
        print('Activity # ', a)
        tt = []
        tra = []
        tea = []
        for N in range(1, 44):
            auxX = []
            auxY = []
            auxZ = []
            for j in range(1,10): #Trials with the same conditions (what varies is the content of training and testing sets)
                X, Y, Z = Random_Forest.exec_(a, tp[a],c[a],N,path,clf)
                auxX.append(X)
                auxY.append(Y)   
                auxZ.append(Z)
            tt.append(np.mean(auxX))
            tra.append(np.mean(auxY))
            tea.append(np.mean(auxZ))
            
        Train_time_=np.append(Train_time_, tt)
        Train_acc_=np.append(Train_acc_, tra)
        Test_acc_= np.append(Test_acc_, tea)
        
    Train_time_=np.reshape(Train_time_, (43,6))
    Train_acc_=np.reshape(Train_acc_, (43,6))
    Test_acc_=np.reshape(Test_acc_, (43,6))

    Train_time=pd.DataFrame(Train_time_, columns=['C1', 'C2', 'F1', 'F2', 'M', 'R'])
    Train_acc=pd.DataFrame(Train_acc_, columns=['C1', 'C2', 'F1', 'F2', 'M', 'R'])
    Test_acc=pd.DataFrame(Test_acc_, columns=['C1', 'C2', 'F1', 'F2', 'M', 'R'])

    writer = pd.ExcelWriter(pathR+name)
    Train_time.to_excel(writer,'Train_time')
    Train_acc.to_excel(writer,'Train_acc')
    Test_acc.to_excel(writer,'Test_acc')
    writer.save()

    print('Random Forest Results Correctly Saved!!!! ')


