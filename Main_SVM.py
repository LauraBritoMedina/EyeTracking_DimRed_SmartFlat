from xlsxwriter.workbook import Workbook
import svm
import pandas as pd
import numpy as np


path = 'Features/'
pathR = 'Results/SVM/'
tp = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4] #testing set percentage
c= [1, 1.5, 3, 4, 8, 10] #Penalty of svm
N=15#Number of most important index


name='svm_results'
    
for a in range(0,6):
    Train_time_ = np.empty((6,0), float)
    Train_acc_ = np.empty((6,0), float)
    Test_acc_ = np.empty((6,0), float)
    print('Activity # ', a)
    
    for i in range(0, 6):
        tt = []
        tra = []
        tea = []
        for j in range(0, 6):
            X, Y, Z = svm.exec_(a, tp[i],c[j],path)
            tt.append(X)
            tra.append(Y)
            tea.append(Z)
            
        Train_time_=np.append(Train_time_, tt)
        Train_acc_=np.append(Train_acc_, tra)
        Test_acc_= np.append(Test_acc_, tea)
        
    Train_time_=np.reshape(Train_time_, (6,6))
    Train_acc_=np.reshape(Train_acc_, (6,6))
    Test_acc_=np.reshape(Test_acc_, (6,6))

    Train_time=pd.DataFrame(Train_time_, columns=['0.15','0.2', '0.25', '0.3', '0.35', '0.4'])
    Train_acc=pd.DataFrame(Train_acc_, columns=['0.15','0.2', '0.25', '0.3', '0.35', '0.4'])
    Test_acc=pd.DataFrame(Test_acc_, columns=['0.15','0.2', '0.25', '0.3', '0.35', '0.4'])

    Train_time = Train_time.set_index([c])
    Train_acc = Train_acc.set_index([c])
    Test_acc = Test_acc.set_index([c])
    
    writer = pd.ExcelWriter(pathR+name+str(a)+'.xlsx')
    Train_time.to_excel(writer,'Train_time')
    Train_acc.to_excel(writer,'Train_acc')
    Test_acc.to_excel(writer,'Test_acc')
    writer.save()
    print('SVM Results Correctly Saved for activity %f Saved!!!! ' %a)

