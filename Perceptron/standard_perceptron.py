import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import sys

trainData = pd.read_csv("./data/train.csv", header = None,names=['x1','x2','x3','x4','y'])
testData = pd.read_csv("./data/test.csv", header = None,names=['x1','x2','x3','x4','y'])
trainData.insert(0,'ones',1)
testData.insert(0,'ones',1)

def data_shuffle(data):
    cols=data.shape[1]
    index_list=shuffle(list(range(data.shape[0])))
    dataNew=data.iloc[index_list]
    x = np.matrix(dataNew.iloc[:,0:cols-1].values)
    y = np.matrix(dataNew.iloc[:,-1].values)
    y[y==0]=-1
    return x,y

# print(y)
def perceptron(data):
    r=0.1
    theta=np.matrix(np.zeros(5))
    for epoch in range(10):
        x,y=data_shuffle(data)
        preS=x * theta.T
        for i in range(preS.shape[0]):
            if preS[i,0]*y[0,i]<=0:
                theta=theta+r*y[0,i]*x[i,:]
        epoch+=1
    return theta
def voted_perceptron(data):
    r=0.1
    theta=np.matrix(np.zeros(5))
    cols=data.shape[1]
    x = np.matrix(data.iloc[:,0:cols-1].values)
    y= np.matrix(data.iloc[:,-1].values)
    y[y==0]=-1
    result_list=[]
    for epoch in range(10):
        m=0
        cm=0
        preV=x * theta.T
        for i in range(preV.shape[0]):
            if preV[i,0]*y[0,i]<=0:
                theta=theta+r*y[0,i]*x[i,:]
                m+=1
            else:
                cm+=1
        result_list.append([cm,theta])
        epoch+=1
    return result_list
def average_perceptron(data):
    r=0.1
    theta=np.matrix(np.zeros(5))
    a=np.matrix(np.zeros(5))
    cols=data.shape[1]
    x = np.matrix(data.iloc[:,0:cols-1].values)
    y= np.matrix(data.iloc[:,-1].values)
    y[y==0]=-1
    result_listA=[]
    for epoch in range(10):
        preV=x * theta.T
        for i in range(preV.shape[0]):
            if preV[i,0]*y[0,i]<=0:
                theta=theta+r*y[0,i]*x[i,:]
            a=a+theta
        result_listA.append([a])
        epoch+=1
    return a,result_listA

def average_test_accuracy(data1,data2):#data1:train;data2:test
    a,result_listA=average_perceptron(data1)
    rows_test=data2.shape[0]
    cols_test=data2.shape[1]
    x1 = np.matrix(data2.iloc[:,0:cols_test-1].values)
    y1= np.matrix(data2.iloc[:,-1].values)
    y1[y1==0]=-1
    countsA=0
    preA1=x1 * a.T
    for i in range(preA1.shape[0]):
        if preA1[i,0]*y1[0,i]<=0:
            countsA+=1
    errA=countsA/rows_test
    return errA,result_listA

def stan_test_accuracy(data,theta):
    rows_test=data.shape[0]
    cols_test=data.shape[1]
    x1 = np.matrix(data.iloc[:,0:cols_test-1].values)
    y1= np.matrix(data.iloc[:,-1].values)
    y1[y1==0]=-1

    preS1=x1 * theta.T
    counts=0
    for i in range(preS1.shape[0]):
        if preS1[i,0]*y1[0,i]<=0:
            counts+=1
    err=counts/rows_test
    return err

def voted_gather(data,list):
    rows=data.shape[0]
    cols=data.shape[1]
    x = np.matrix(data.iloc[:,0:cols-1].values)
    y= np.matrix(data.iloc[:,-1].values)
    y[y==0]=-1
    sgn_final=np.matrix(np.zeros(rows)) #(1,872)
    for i in range(10):
        theta=list[i][1]
        preV1=x*theta.T #(872,1)
        sgn=np.matrix(np.zeros(rows)) 
        for index in range(preV1.shape[0]):
            if preV1[index,0] > 0:
                sgn[0,index]=1
            else:
                sgn[0,index]=-1
            cm=list[i][0]
            sgn_final[0,index]+=cm*sgn[0,index]
        # sgn_final+=cm*sgn
    for index in range(sgn_final.shape[1]):
        if sgn_final[0,index]>=0:
            sgn_final[0,index]=1
        else:
            sgn_final[0,index]=-1
    
    return sgn_final

def standard_perceptron(trainData,testData):
    theta=perceptron(trainData)
    err=stan_test_accuracy(testData,theta)
    return theta,err

def voted_test_accuracy(data1,data2): #data1:train;data2:test
    result_list=voted_perceptron(data1)
    rows_test=data2.shape[0]
    y1= np.matrix(data2.iloc[:,-1].values)
    y1[y1==0]=-1
    countsV=0
    sgn_final=voted_gather(data2,result_list)
    for i in range(sgn_final.shape[1]):
        if sgn_final[0,i]!=y1[0,i]:
            countsV+=1

    errV=countsV/rows_test
    return result_list,errV

if sys.argv[1] == "sp":
    [theta,err] = standard_perceptron(trainData,testData)
    print("thetaStandard: ", theta)
    print("errorStrandard: ", err)
elif sys.argv[1] == "vp":
    [result_list,errV] = voted_test_accuracy(trainData,testData)
    for i in range(10):
         print("cm,thetaVoted: ", (result_list[i][0],result_list[i][1]))
    print("errorVoted: ", errV)
elif sys.argv[1] == "ap":
    errA,result_listA= average_test_accuracy(trainData,testData)
    for i in range(10):
        print("thetaAverage: ", (result_listA[i][0]))
    print('errorAverage:',errA)
