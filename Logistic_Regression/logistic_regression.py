import numpy as np
import pandas as pd
from sklearn.utils import shuffle
# import matplotlib.pyplot as plt

import sys

trainData = pd.read_csv("./data/train.csv", header = None,names=['x1','x2','x3','x4','y'])
testData = pd.read_csv("./data/test.csv", header = None,names=['x1','x2','x3','x4','y'])
trainData.insert(0,'ones',1)
testData.insert(0,'ones',1)

def data_shuffle(data):
    cols=data.shape[1]
    index_list=shuffle(list(range(data.shape[0])))
    dataNew=data.iloc[index_list]
    x = np.array(dataNew.iloc[:,0:cols-1].values)
    y = np.array(dataNew.iloc[:,-1].values)
    y[y==0]=-1
    return x,y

def sigmoid(z):
    y=1/(1+np.exp(-z))
    return y
#----------------------------MAP-----------------#
def sumLoss(theta,x,y,v):
    cols=x.shape[0]
    loss=np.zeros(cols)
    sumLoss=0
    for i in range(cols):
        loss[i]=np.log(1+np.exp(-y[i]*theta@x[i,:]))
        sumLoss=sumLoss+loss[i]

    sumLoss=sumLoss+theta.T@theta/v
    return sumLoss

def stochasticLR(data,v):  #v=0.01 #0.01, 0.1, 0.5, 1, 3, 5, 10, 100
    r0=0.001 #0.0002
    d=0.001
    epochs=100
    m=data.shape[0]
    theta=np.zeros(5) #5*1 array
    loss=np.zeros(epochs)
    sumObj=np.zeros(epochs)
    for epoch in range(epochs):
        x,y=data_shuffle(data)
        xCur=x[0,:] #5*1
        yCur=y[0]  #1
        learningRate=r0/(1+r0/d*epoch)
        gradJ=m*yCur*xCur*(sigmoid(yCur*theta@xCur)-1)+2/v * theta #a vector
        loss[epoch]=m*np.log(1+np.exp(-yCur*theta@xCur))+theta.T@theta/v
        theta=theta-learningRate*gradJ      
        sumObj[epoch]=sumLoss(theta,x,y,v)

    return theta,loss,sumObj

# theta,loss,sumLoss=stochasticLR(trainData,100)
#----------------------------MLE-----------------#
def sumLossMLE(theta,x,y,v):
    cols=x.shape[0]
    loss=np.zeros(cols)
    sumLossMLE=0
    for i in range(cols):
        loss[i]=np.log(1+np.exp(-y[i]*theta@x[i,:]))
        sumLossMLE=sumLossMLE+loss[i]

    return sumLossMLE

def stochasticMLE(data,v):  #v=0.01 #0.01, 0.1, 0.5, 1, 3, 5, 10, 100
    r0=0.0001 #0.0002
    d=0.001
    epochs=100
    m=data.shape[0]
    theta=np.zeros(5) #5*1 array
    loss=np.zeros(epochs)
    sumMLEObj=np.zeros(epochs)
    for epoch in range(epochs):
        x,y=data_shuffle(data)
        xCur=x[0,:] #5*1
        yCur=y[0]  #1
        learningRate=r0/(1+r0/d*epoch)
        gradJ=m*yCur*xCur*(sigmoid(yCur*theta@xCur)-1)
        loss[epoch]=m*np.log(1+np.exp(-yCur*theta@xCur))
        theta=theta-learningRate*gradJ      
        sumMLEObj[epoch]=sumLossMLE(theta,x,y,v)

    return theta,loss,sumMLEObj

# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(np.arange(len(sumMLEObj)), sumMLEObj, 'r')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('sumLoss')
# ax.set_title('sumLoss vs. Training Epoch')
# plt.show()

def stochasticLR_accuracy(data,theta):
    rows_test=data.shape[0]
    cols_test=data.shape[1]
    x1 = np.array(data.iloc[:,0:cols_test-1].values)
    y1= np.array(data.iloc[:,-1].values)
    y1[y1==0]=-1
    pre=np.zeros(rows_test)
    counts=0
    for i in range(rows_test):
        pre[i]=theta@x1[i,:]
        if pre[i]*y1[i]<=0:
            counts+=1
    err=counts/rows_test
    return err
#-------------setuo--------------------#
if sys.argv[1] == "MAP":
    theta,loss,sumLoss=stochasticLR(trainData,100)
    errorTrain=stochasticLR_accuracy(trainData,theta)
    errorTest=stochasticLR_accuracy(testData,theta)
    print('stochastic for MAP_LR')
    print(theta,'weights')
    print(errorTrain,'trainning error')
    print(errorTest,'trainning error')
elif sys.argv[1] == "MLE":
    theta,loss,sumMLEObj=stochasticMLE(trainData,100)
    errorTrain=stochasticLR_accuracy(trainData,theta)
    errorTest=stochasticLR_accuracy(testData,theta)
    print(theta,'weights')
    print(errorTrain,'trainning error')
    print(errorTest,'trainning error')