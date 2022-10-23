import numpy as np
import pandas as pd
# from numpy import linalg as la
import matplotlib.pyplot as plt

import sys





data = pd.read_csv("./data/train.csv", header = None,names=['x1','x2','x3','x4','x5','x6','x7','y'])
data.insert(0,'ones',1)
cols=data.shape[1]
x = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
x = np.matrix(x.values)
y = np.matrix(y.values)
# theta=np.matrix(np.array([0,0,0,0]))
# print(x.shape,y.shape,len(x))

#define cost 
def computeCost(x,y,theta):
    innerCost=np.power((y-x * theta.T),2)
    sumCost=(np.sum(innerCost)/(2*len(x)))
    return sumCost

#compute gradient
def batGradient(x,y,theta):
    parameters = 8
    grad=np.zeros(parameters)
    error = y-x * theta.T
    for j in range(parameters):
            grad[j] = -np.sum(np.multiply(error,x[:,j]))
    return grad

# batch gradient descent
def batGradientDescent(x,y):
    iterFinal = 1
    alpha = 1
    para=8

    while alpha>1e-3:
        # theta =np.matrix(np.zeros((10001,para)))
        theta = np.matrix(np.zeros(para))
        temp=np.matrix(np.zeros(para))
        # print(theta,'theta')
        cost=[]
        for iters in range(10000):
            # temp = theta
            gradient=batGradient(x,y,theta) #gradient is a array[j].
            # print(gradient,'gradient')
            for j in range(para):
                temp[0,j]=theta[0,j]
                theta[0,j]=theta[0,j]-alpha*gradient[j] #theta[0,j] store w_t+1

            # print(alpha*gradient,'delta')  
            # print(temp,'temp')
            # print(theta,'theta')
            cost.append(computeCost(x,y,theta))
            if np.linalg.norm(theta-temp) < 0.000001:
                iterFinal = iters+1
                costFinal=computeCost(x,y,theta)
                # print(costFinal)
                # print(theta-temp,'theta-temp')
                # print(iterFinal,theta,alpha)
                return [iterFinal,theta,alpha,costFinal] 
        # costCur=computeCost(x,y,theta)
        # print(costCur,"costCur")
        # print(computeCost(x,y,theta),"split")
        # iters=iters+1
        alpha *= 0.5



#stochastic gradient
# NumSamples=x.shape[0]
# SampleX=np.random.choice(NumSamples)
# print(SampleX)
import random
NumSamples=x.shape[0]
def stoGradientDescentBatch(x,y):
    iterFinal = 1
    alpha = 1
    para=8
    costMin=10000
    bestWeight=None
    while alpha>1e-3:
        # theta =np.matrix(np.zeros((10001,para)))
        theta = np.matrix(np.zeros(para))
        temp=np.matrix(np.zeros(para))
        # print(theta,'theta')
        cost=[]

        for iters in range(30000):
            # temp = theta
            SampleIndex=np.random.choice(NumSamples)
            gradient=batGradient(x[SampleIndex,:],y[SampleIndex,:],theta) #gradient is a array[j].
            # print(gradient,'gradient')
            for j in range(para):
                temp[0,j]=theta[0,j]
                theta[0,j]=theta[0,j]-alpha*gradient[j] #theta[0,j] store w_t+1

            # print(alpha*gradient,'delta')  
            # print(temp,'temp')
            # print(theta,'theta')
            CostCur=computeCost(x,y,theta)
            cost.append(CostCur)
            if CostCur<costMin:
                bestWeight=theta
            if np.linalg.norm(theta-temp) < 0.000001:
                iterFinal = iters+1
                # costFinal=computeCost(x,y,theta)
                # print(costFinal)
                # print(theta-temp,'theta-temp')
                # print(iterFinal,theta,alpha)
                return [iterFinal,theta,alpha,cost[-1],bestWeight] 
        # costCur=computeCost(x,y,theta)
        # print(costCur,"costCur")
        # print(computeCost(x,y,theta),"split")
        # iters=iters+1
        alpha *= 0.5

if sys.argv[1] == "bgd":
        [iters,theta, alpha,costFinal] = batGradientDescent(x,y) 
        print('iterations:',iters)
        print("theta: ", theta)
        print("alpha: ", alpha)
        print("TEST COST: ", costFinal)
elif sys.argv[1] == "sgd":
        iterFinal,theta,alpha,costFinal,bestWeight = stoGradientDescentBatch(x,y)
        print("theta: ", theta)
        print("alpha: ", alpha)
        print("TEST COST: ", computeCost(x,y,bestWeight))

# iterFinal,theta,alpha,cost,bestWeight=stoGradientDescentBatch(x,y)
# print(iterFinal,theta,alpha,cost,bestWeight)


# #import test data
# data1= pd.read_csv("./data/test.csv", header = None,names=['x1','x2','x3','x4','x5','x6','x7','y'])
# data1.insert(0,'ones',1)
# #preprocessing the test data
# cols=data1.shape[1]
# x1 = data1.iloc[:,0:cols-1]
# y1 = data1.iloc[:,cols-1:cols]
# x1 = np.matrix(x1.values)
# y1 = np.matrix(y1.values)
# # theta=np.matrix(np.array([0,0,0,0]))
# print(x1.shape,y1.shape,len(x))

# #calculate the cost
# testCost=computeCost(x1,y1,theta)
# print(theta,testCost)

# import sys


# if sys.argv[1] == "bgd":
#         [itersFinal,theta, alpha] = batGradientDescent(x,y)
#         print('iterations:',iters)
#         print("W: ", theta)
#         print("R: ", alpha)
#         print("TEST COST: ", computeCost(x,y,theta))
# elif sys.argv[1] == "sgd":
#         [w, r] = stoGradientDescent()
#         print("W: ", w)
#         print("R: ", r)
#         print("TEST COST: ", computeCost(x,y,theta))



#the analytical solution
# X=np.linalg.pinv(x.T*x)
# optTheta=X*x.T*y
# print(optTheta[:,0])
# print(np.squeeze(optTheta))

# iterFinal,theta,alpha,cost=batGradientDescent(x,y)
# print(iterFinal,theta,alpha,cost[-1])

# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(np.arange(len(cost)), cost, 'r')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Cost')
# ax.set_title('Error vs. Training Epoch')
# plt.show()


