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
    x = np.array(dataNew.iloc[:,0:cols-1].values)
    y = np.array(dataNew.iloc[:,-1].values)
    y[y==0]=-1
    return x,y

def sigmoid(z):
    y=1/(1+np.exp(-z))
    return y

#-----------a three-layered NN---------#
def forward_propagate(data, theta1, theta2,theta3): 
    cols=data.shape[1]  
    x = np.array(data.iloc[:,0:cols-1].values)
    m = x.shape[0]    
    a1=np.insert(x, 0, values=np.ones(m), axis=1) #m*N #input for L1
    z1 = sigmoid(a1 @ theta1.T) #theta1[h1*N],(h1+1) neurons in layer1.
    a2 = np.insert(z1, 0, values=np.ones(m), axis=1) #output of L1/input for L2
    z2 = sigmoid(a2 @ theta2.T)   #(h2+1) neurons in L2. theta2[h2*h1]
    a3 = np.insert(z2, 0, values=np.ones(m), axis=1) 
    yPre = a3 @ theta3.T   
    return a1, z1, a2, z2, a3, yPre


def update_weights(theta, gradJ):
    r0=0.001 #0.0002
    d=0.001
    epochs=100
    for epoch in range(epochs):
        learningRate=r0/(1+r0/d*epoch)
        theta=theta-learningRate * gradJ

    return theta


def loss(yPre, data):
    y = np.array(data.iloc[:,-1].values)
    y[y==0]=-1
    return np.square(yPre - y) / 2

def run_backpropagation(weights, layers, y, prediction, activations):
    loss_deriv = prediction - y
    for i in range(layers):
        if i==layers-1:
            partial_z_w=partial_z_w()
            partial_z_lowerZ=partial_z_lowerZ()
        else:
            partial_z_w=partial_z_w()
            partial_z_lowerZ=partial_z_lowerZ()

    weight_derivs=0
    return weight_derivs

def partial_z_w():

    return partial_z_w

def partial_z_lowerZ():

    return partial_z_lowerZ