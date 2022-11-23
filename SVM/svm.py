import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
    y = np.matrix(dataNew.iloc[:,-1].values)  #1x872
    y[y==0]=-1
    return x,y

def calObjective(theta,x,y,C,N):
    weights=np.delete(theta,0,axis=1)
    pre=x * theta.T 
    if 1-y*pre >= 0:
        max = 1-y*pre
    else:
        max=0
    obj=1/2*weights*weights.T+C*N*max
    return obj
def calObjectFull(theta,x,y,C):
    cols=x.shape[0]
    weights=np.delete(theta,0,axis=1)
    pre=x * theta.T #872x1
    max=np.zeros(cols)
    sumMax=0
    for i in range(cols):
        if 1-y[0,i]*pre[i,0] >= 0:
            max[i] = 1-y[0,i]*pre[i,0]
        else:
            max[i]=0
        sumMax=sumMax+max[i]
    objFull=1/2*weights*weights.T+C*sumMax
    return objFull

#------------gamma 1--------------#
def ssdPrimal(data,C):
    r0=0.05
    a=0.1
    N=data.shape[0]
    theta=np.matrix(np.zeros(5))  #1x5,b=theta[0,0]
    obj=np.zeros(100)
    objFull=np.zeros(100)
    for epoch in range(100):
        x,y=data_shuffle(data)
        xCur=x[0,:] #x:872x5,simply choose the first one item after shuffle 1x5
        yCur=y[0,0] #y:1x872
        learningRate=r0/(1+r0/a*epoch)

        if yCur * xCur * theta.T <= 1:
            gradJ=np.delete(theta,0,axis=1)
            gradJ=np.insert(gradJ,0,values=0,axis=1)  #the fist item in grad is 0 for bias.1x5
            theta=theta-learningRate*gradJ +learningRate*C*N*yCur*xCur
        else:
            for i in range(3):
                theta[0,i+1]=(1-learningRate)*theta[0,i+1] #update all weights except bias.
        obj[epoch]=calObjective(theta,xCur,yCur,C,N)
        objFull[epoch]=calObjectFull(theta,x,y,C)
    return theta,obj


#------------gamma 2--------------#
# def ssdPrimal(data,C):
#     r0=0.01
#     N=data.shape[0]
#     theta=np.matrix(np.zeros(5))  #1x5,b=theta[0,0]
#     obj=np.zeros(100)
#     objFull=np.zeros(100)
#     for epoch in range(100):
#         x,y=data_shuffle(data)
#         xCur=x[0,:] #x:872x5,simply choose the first one item after shuffle 1x5
#         yCur=y[0,0] #y:1x872
#         learningRate=r0/(1+epoch)

#         if yCur * xCur * theta.T <= 1:
#             gradJ=np.delete(theta,0,axis=1)
#             gradJ=np.insert(gradJ,0,values=0,axis=1)  #the fist item in grad is 0 for bias.1x5
#             theta=theta-learningRate*gradJ +learningRate*C*N*yCur*xCur
#         else:
#             for i in range(3):
#                 theta[0,i+1]=(1-learningRate)*theta[0,i+1] #update all weights except bias.
#         obj[epoch]=calObjective(theta,xCur,yCur,C,N)
#         objFull[epoch]=calObjectFull(theta,x,y,C)
#     return theta,obj,objFull

def ssdPrimal_test_accuracy(data,theta):
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


#-------------pic drawing-----------#
# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(np.arange(len(obj)), obj, 'r')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Objective')
# ax.set_title('Objective vs. Training Epoch')
# plt.show()

# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(np.arange(len(objFull)), objFull, 'b')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Objective in full dataset')
# ax.set_title('ObjectiveFull vs. Training Epoch')
# plt.show()
# print(errTest)
#-------------pic drawing-----------#





#--------------dual SVM using Scipy------------#
def dualSVM(data):
    cols=data.shape[1]
    dataX = np.array(data.iloc[:,1:cols-1].values) #872x4
    y = np.array(data.iloc[:,-1].values)  #872,
    y[y==0]=-1
    M=np.dot(dataX,dataX.T)
    return dataX,y,M


def opt(dataX,y,M,C):
    N=dataX.shape[0]
    fun=lambda x: 1/2*(x*y).T@M@(x*y)-np.sum(x)
    x0=np.zeros(N)
    cons=[{'type':'ineq','fun':lambda x:x},{'type':'ineq','fun':lambda x:-x+C},
    {'type':'eq','fun': lambda x:np.sum(x*y)}]
    res = minimize(fun,x0,method='SLSQP', constraints=cons,options={'ftol':1e-4})
    return res.fun,res.x,res.success

def wbCalculate(alpha,y,dataX):
    pre=alpha*y
    theta=np.sum(pre[:,None]*dataX,axis=0)
    b=y-dataX @ theta.T
    b[b<=1e-5]=0
    counts=(b!=0).sum()
    b=np.sum(b)/counts
    return theta,b

def dualSVM_test_accuracy(data,theta,b):
    # w=np.insert(theta,0,values=b,axis=0)
    rows_test=data.shape[0]
    cols_test=data.shape[1]
    x1 = np.array(data.iloc[:,1:cols_test-1].values)
    y1= np.array(data.iloc[:,-1].values)
    y1[y1==0]=-1

    preS1=x1 @ theta + b
    counts=(preS1*y1<=0).sum()
    err=counts/rows_test
    return err

def dualSVM_test_accuracy(data,theta,b):
    rows_test=data.shape[0]
    cols_test=data.shape[1]
    x1 = np.array(data.iloc[:,1:cols_test-1].values)
    y1= np.array(data.iloc[:,-1].values)
    y1[y1==0]=-1

    preS1=x1 @ theta + b
    counts=(preS1*y1<=0).sum()
    err=counts/rows_test
    return err

#-------------Gaussian kernel for NL SVM------------#
def dualKernelSVM(data,gamma):
    cols=data.shape[1]
    dataX = np.array(data.iloc[:,1:cols-1].values) #872x4
    y = np.array(data.iloc[:,-1].values)  #872,
    y[y==0]=-1
    A1=np.sum(dataX*dataX,axis=1,keepdims=True)
    A2=A1.T
    A3=-2*dataX@dataX.T
    A=np.exp(-(A1+A2+A3)/gamma)
    # A4=dataX[:,None,:]-dataX[None,:,:]
    # A5=np.square(A4).sum(axis=-1)
    # A6=np.exp(-(A5)/gamma)
    return dataX,y,A

def opt(dataX,y,A,C):
    N=dataX.shape[0]
    fun=lambda x: 1/2*(x*y).T@A@(x*y)-np.sum(x)
    x0=np.zeros(N)
    cons=[{'type':'ineq','fun':lambda x:x},{'type':'ineq','fun':lambda x:-x+C},
    {'type':'eq','fun': lambda x:np.sum(x*y)}]
    res = minimize(fun,x0,method='SLSQP', constraints=cons,options={'ftol':1e-3})
    return res.fun,res.x,res.success

def bCalculate(alpha,y,A):
    alphay=alpha*y
    preG=alphay.T@A   #should be a vector
    b=y-preG
    b[alpha<=1e-10]=0
    counts=(b!=0).sum()
    b=np.sum(b)/counts
    return b,counts

def errCalculate(alpha,y,b,dataX,dataTest,gamma):
    nTest=dataTest.shape[1]
    rTest=dataTest.shape[0]
    dataXTest = np.array(dataTest.iloc[:,1:nTest-1].values) #872x4
    yTest= np.array(dataTest.iloc[:,-1].values)
    yTest[yTest==0]=-1
    alphay=alpha*y
    counts1=(alpha<=1e-10).sum()
    newA1=np.sum(dataX*dataX,axis=1,keepdims=True)
    newA2=np.sum(dataXTest*dataXTest,axis=1,keepdims=True) 
    newA2=newA2.T
    newA3=-2*dataX@dataXTest.T
    newA=np.exp(-(newA1+newA2+newA3)/gamma)
    preT=alphay.T@newA +b
    
    counts=(preT*yTest<=0).sum()
    err=counts/rTest
    return err,counts1


if sys.argv[1] == "spsvm":
    theta,obj=ssdPrimal(trainData,100/873)
    errTest=ssdPrimal_test_accuracy(testData,theta)
    errTrain=ssdPrimal_test_accuracy(trainData,theta)
    print(theta,'weights')
    print(errTrain,'trainning error')
    print(errTest,'trainning error')
elif sys.argv[1] == "dsvm":
    dataX,y,M=dualSVM(trainData)
    objDual,alpha,sign=opt(dataX,y,M,100/873)
    thetaDual,b=wbCalculate(alpha,y,dataX)
    errorTestDual=dualSVM_test_accuracy(testData,thetaDual,b)
    print(thetaDual,b,'thetaDual','b')
    print(errorTestDual,'Testing error')
elif sys.argv[1] == "ksvm":
    normalized_trainData=(trainData-trainData.min())/(trainData.max()-trainData.min())
    normalized_testData=(testData-trainData.min())/(trainData.max()-trainData.min())
    dataX,y,A=dualKernelSVM(normalized_trainData,0.1)   #gamma
    objDual,alpha,sign=opt(dataX,y,A,500/873)  #C
    b,counts=bCalculate(alpha,y,A)
    errorK=errCalculate(alpha,y,b,dataX,normalized_testData,0.1)
    print(b,'b')
    print(counts,'support vectors')
    print(errorK,'Testing errorKernel')
