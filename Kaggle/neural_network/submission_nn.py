#To deal with such variables in SVM classification, we typically do a “one-hot” encoding
import numpy as np
import pandas as pd
import torch
from torch import nn as nn
from torch import optim as optim #sgd,...
from torch.nn import functional as F  #ReLu, tanh....
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable 
# from dataset import *
from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets as datasets
# from torchvision import transforms as transforms
from tqdm.auto import tqdm, trange
from sklearn.model_selection import train_test_split

trData = pd.read_csv("./data/train_final.csv", header = None,names=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','y'],skiprows=1)
teData = pd.read_csv("./data/removeID_test_final.csv", header = None,names=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14'],skiprows=1)

def oneHot(data):
    data=data.drop(columns=['x14'])
    dataOneHot=pd.get_dummies(data=data,columns=['x2','x4','x5','x6','x7','x8','x9','x10'])#,'x14'
    return dataOneHot
#---------------------NN-------------------#
class BankNote(Dataset):
    def __init__(self, X,y):
        super(BankNote, self).__init__()
        self.X=X
        self.y=y
        
    def __getitem__(self, idx):
        
        return self.X[idx,:], self.y[idx,:]
    
    def __len__(self,):
        # Return total number of samples.
        return self.X.shape[0]

class BankNoteTe(Dataset):
    def __init__(self, X):
        super(BankNoteTe, self).__init__()
        self.X=X
        
    def __getitem__(self, idx):
        
        return self.X[idx,:]
    
    def __len__(self,):
        # Return total number of samples.
        return self.X.shape[0]

class Net(nn.Module):
    def __init__(self, config, act=nn.Tanh(),init=torch.nn.init.xavier_uniform):
                # hidden_dim=100, outputDim=1, weight_scale=1e-3, reg=0.0):     #weight_scale is the variation of normal dis. reg is for regularizer. input_dim is the dim before augmentation.
        super(Net, self).__init__()  #super can implicitly connect method of subclass to method of superclass(父类)
        layers_list = []
        self.init=init

        for l in range(len(config)-2):
          in_dim =  config[l]
          out_dim = config[l+1]

          layers_list.append(nn.Linear(in_features=in_dim, out_features=out_dim))
          layers_list.append(act)
        #last layer
        layers_list.append(nn.Linear(in_features=config[-2], out_features=config[-1]))
        self.net = nn.Sequential(*layers_list)
        # self.net = nn.ModuleList(layers_list)
        self.net.apply(self.init_weights)
    def init_weights(self,m):
      if isinstance(m, nn.Linear):
          self.init(m.weight)
          # if self.init=="xavier":
          #   torch.nn.init.xavier_uniform(m.weight)
          # elif self.init=="he":
          #   torch.nn.init.kaiming_normal(m.weight)
          m.bias.data.fill_(0.01)
    def forward(self,X):
    #   h=X
    #   for layer in self.net:
    #     h=layer(h)
    #   return h
        return self.net(X)

class Solver(object):
    def __init__(self, model, data,lossfcn,optimizer, **kwargs):
        self.model = model
        self.trainData = data['train']
        self.valData = data['val']

        # if the key exists, pop out the value; if not, use the second value.
        # self.num_epochs = kwargs.pop('num_epochs', 10)   #numbers of using the whole dataset,if exits, pop it otherwise use 2.

     
        self.print_every = kwargs.pop('print_every', 1)
        self.verbose = kwargs.pop('verbose', True)
        self.lossfcn=lossfcn
        self.optimizer=optimizer
        self.train_loss_history=[]
        self.train_acc_historyEpoch=[]
        self.val_acc_historyEpoch=[]
        self.best_val_acc=0
        self.loss_history=[]
        self.best_params=None
    def epochTrain(self):
        # Make a minibatch of training data
        for idx,(data_x,data_y) in enumerate(self.trainData):
            # print(data_x.shape,data_y.shape)
            # calls hooks like this one
            # on_train_batch_start()

            # train step
            y_pred=self.model.forward(data_x)

            loss = self.lossfcn(y_pred,data_y)

            # clear gradients
            self.optimizer.zero_grad()

            # backward
            loss.backward()

            # update parameters
            self.optimizer.step()
            self.loss_history.append(loss)
    def check_accuracy(self,testDataLoader):
        y_pred=[]
        y_true=[]
        with torch.no_grad():
            for idx,(BatchData_x,BatchData_y) in enumerate(testDataLoader):
                y_predBatch=self.model.forward(BatchData_x).numpy()
                # print(BatchData_y)
                # print((BatchData_y.size()))
                y_pred.append(y_predBatch[:,0])
                y_true.append(BatchData_y[:,0].numpy())
        y_pred = np.hstack(y_pred)[:,None]
        y_pred[y_pred<0]=-1
        y_pred[y_pred>=0]=1
        y_true = np.hstack(y_true)[:,None]
        acc = np.mean(y_pred == y_true)
        # print(acc,'acc')
        return acc
    def prediction(self,testDataLoader):
        y_pred=[]
        with torch.no_grad():
            for idx,(BatchData_x) in enumerate(testDataLoader):
                y_predBatch=self.model.forward(BatchData_x).numpy()
                # print(y_predBatch)
                # print((len(y_predBatch)))
                y_pred.append(y_predBatch[:,0])
        y_pred = np.hstack(y_pred)[:,None]
        y_pred[y_pred<0]=-1
        y_pred[y_pred>=0]=1
        print(y_pred)
        return y_pred

    def train(self,NumEpochs):

        for epoch in range(NumEpochs):
            self.epochTrain()
            
            # if t % self.print_every == 0:
            
            # print(self.optim_configs['learning_rate'])

            train_acc = self.check_accuracy(self.trainData)
            val_acc = self.check_accuracy(self.valData)
            if self.verbose:
                print(f"training acc {train_acc}, validation acc {val_acc}")
                print('(epoch %d / %d) loss: %f' % (
                        epochs, NumEpochs, self.loss_history[-1]))
            self.train_acc_historyEpoch.append(train_acc)
            self.val_acc_historyEpoch.append(val_acc)
            
            # Keep track of the best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = self.model.state_dict()
        self.model.load_state_dict(self.best_params)
        # At the end of training swap the best params into the model     
        # print(self.loss_history)
        # self.model.params = self.best_params
        return self.loss_history

X_train= trData.iloc[:,0:trData.shape[1]-1]
print(X_train.index)
y_train= trData.iloc[:,-1]
y_train= np.array(y_train)[:,None].astype(np.float32)
y_train[y_train==0]=-1

data=pd.concat([X_train,teData],ignore_index=True)
data=oneHot(data)
print(data)
print(trData.shape[0])
X_train=np.array(data.iloc[0:25000,:].values).astype(np.float32)
X_test=np.array(data.iloc[25000:48842,:].values).astype(np.float32)
print(X_test.shape,X_train.shape)

minnTr=X_train.min(axis=0,keepdims=True)
maxxTr=X_train.max(axis=0,keepdims=True)
meannTr=X_train.mean(axis=0,keepdims=True)
minnTe=X_test.min(axis=0,keepdims=True)
maxxTe=X_test.max(axis=0,keepdims=True)
X_train=(X_train-minnTr)/(maxxTr-minnTr)
X_test=(X_test-minnTe)/(maxxTe-minnTe)
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=1)

dataset_train = BankNote(X_train,y_train)
dataset_val = BankNote(X_val,y_val)
dataset_test = BankNoteTe(X_test)

train_loader = DataLoader(dataset_train, batch_size=640, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=640, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=640, shuffle=False)

# widths=[50,100,125]
# depths=[3,5,10,15]
widths=[125]
depths=[5]
actInits={"relu+he":[nn.ReLU(),torch.nn.init.kaiming_normal]}
# actInits={"tanh+xvair":[nn.Tanh(),torch.nn.init.xavier_uniform],"relu+he":[nn.ReLU(),torch.nn.init.kaiming_normal]}

for width in widths:
    trainRes=[]
    for depth in depths:
        for actInit in actInits:
            subconfig=[width]*depth #width,width,width
            config = [81]+subconfig+ [1]
            act=actInits[actInit][0]
            init=actInits[actInit][1]
            epochs=200
            reg=1e-5
            lr=1e-3
            
            model = Net(config,act=act,init=init)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
            data={"train":train_loader,"val":val_loader}
            loss_func=nn.MSELoss()
            solver=Solver(model=model,data=data,lossfcn=loss_func,optimizer=optimizer,verbose=True)
            solver.train(NumEpochs=epochs)
            testPre=solver.prediction(test_loader)
            trainAcc=solver.check_accuracy(train_loader)
            valAcc=solver.check_accuracy(val_loader)
            print(f"width:{width},depth:{depth},actInit:{actInit},trainAcc is {trainAcc},valAcc is {valAcc}")
    # Res=["depth-"+str(depth)+"width-"+str(width)]+trainRes
    # Res="&".join(Res)+"\\"+"\\"
            # Output = pd.DataFrame({"ID":np.arange(len(testPre))+1,'Prediction': testPre[:,0]})
            # Output=pd.DataFrame({'Prediction': testPre})
            # Output.to_csv(str(depth)+str(width)+"NN.csv", index=False) 