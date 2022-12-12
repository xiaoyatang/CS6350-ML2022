import numpy as np
import pandas as pd
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

def affine_forward(x, w, b):
    out=x@w+b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache
    shape=np.array(x.shape) #(m,n)
    db=np.sum(dout,0) #沿第一列求和，dout=m*1
    dw=x.T@dout
    # dx=np.reshape(dout@w.T,shape) 
    dx= dout@w.T   #20*10
    return dx, dw, db
  
def SigmoidFoward(x):
  output=sigmoid(x)
  cache=(output) #return a tuple
  return output,cache
  
def SigmoidBackward(dout, cache):
  sigma=cache[0]
  dsigma=sigma*(1-sigma)*dout
  return dsigma

def affine_sigmoid_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, sig_cache = SigmoidFoward(a)
    cache = (fc_cache, sig_cache)
    return out, cache

def affine_sigmoid_backward(dout, cache):
    fc_cache, sig_cache = cache
    da = SigmoidBackward(dout, sig_cache)     #partial to neuron
    dx, dw, db = affine_backward(da, fc_cache) #partial to wx+b, w ,
    return dx, dw, db
  
def mse_loss(y, yStar):
  loss=np.mean(0.5*(y-yStar)**2)
  dx=y-yStar
  return loss, dx


class ThreeLayerNet(object):

    def __init__(self, input_dim=4, hidden_dim=100, outputDim=1,
                 weight_scale=1e-3, reg=0.0):     #weight_scale is the variation of normal dis. reg is for regularizer. input_dim is the dim before augmentation.
      #-----initialize the parameters for NN---------#
        self.params = {}
        self.reg=reg

        # W1=np.random.normal(0.0,weight_scale,(input_dim,hidden_dim))
        W1=np.random.uniform(-1/input_dim**0.5,1/input_dim**0.5,(input_dim,hidden_dim))
        # W1=np.zeros((input_dim,hidden_dim))
        b1=np.zeros(hidden_dim)
        # W2=np.random.normal(0.0,weight_scale,(hidden_dim,hidden_dim))
        W2=np.random.uniform(-1/hidden_dim**0.5,1/hidden_dim**0.5,(hidden_dim,hidden_dim))
        # W2=np.zeros((hidden_dim,hidden_dim))
        b2=np.zeros(hidden_dim)        
        # W3=np.random.normal(0.0,weight_scale,(hidden_dim,outputDim))
        W3=np.random.uniform(-1/hidden_dim**0.5,1/hidden_dim**0.5,(hidden_dim,outputDim))
        # W3=np.zeros((hidden_dim,outputDim))
        b3=np.zeros(outputDim) 
        self.params["W1"]=W1
        self.params["W2"]=W2
        self.params["W3"]=W3
        self.params["b1"]=b1
        self.params["b2"]=b2
        self.params["b3"]=b3

    #---------this is for question3-------3
    def setWeight(self):
        self.params["b1"]=np.array([-1,1])
        self.params["W1"]=np.array([[-2,2],[-4,3]])

        self.params["b2"]=np.array([-1,1])
        self.params["W2"]=np.array([[-2,2],[-3,3]])

        self.params["W3"]=np.array([[2,-1.5]]).T
        self.params["b3"]=np.array([-1])
    #---------this is for question3-------3
      
    def prediction(self, X, y=None,printGrad=False):

        out, cache1 = affine_sigmoid_forward(X, self.params["W1"], self.params["b1"])
        out2, cache2=affine_sigmoid_forward(out, self.params["W2"], self.params["b2"])
        out3, cache3=affine_forward(out2, self.params["W3"], self.params["b3"])
        # print(out,out2,out3,"o1 o2 o3")
        scores=out3
        
        
        
        if y is None:
            return scores

        loss, grads = 0, {}
        
        loss, d_start=mse_loss(out3,y)
        dx3,dw3,db3=affine_backward(d_start,cache3)
        dx2,dw2,db2=affine_sigmoid_backward(dx3,cache2)
        dx1,dw1,db1=affine_sigmoid_backward(dx2,cache1)
        if printGrad:
          print(d_start,'partL/partY')
          print(dw3,db3,"layer 3:dw3,db3")
          print(dw2,db2,"layer 2:dw2,db2")
          print(dw1,db1,"layer 1:dw1,db1")
        grads["W1"]=dw1+self.params["W1"]*self.reg   #derivative plus regularizer 
        grads["W2"]=dw2+self.params["W2"]*self.reg
        grads["W3"]=dw3+self.params["W3"]*self.reg
        grads["b1"]=db1
        grads["b2"]=db2
        grads["b3"]=db3
        return loss, grads



def sgd(w, dw, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


class Solver(object):
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # if the key exists, pop out the value; if not, use the second value.
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0) #decay efficient of learning rate
        self.batch_size = kwargs.pop('batch_size', 1)  #we choose 100 per time to do SGD.
        self.num_epochs = kwargs.pop('num_epochs', 10)   #numbers of using the whole dataset
        self.num_train_samples = kwargs.pop('num_train_samples', 872)
        self.num_val_samples = kwargs.pop('num_val_samples', None)

     
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        if self.update_rule not in globals():
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = globals()[self.update_rule]

        self._reset()


    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
        self.optim_configs = {}
        self.valloss_history=[]
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}  #item used list the dic as a list (a:b),(c:d),...,use k,v to scan the list.
            self.optim_configs[p] = d


    def _step(self):
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size) #randomly choose batch_size examples from num_train, this is the data to do SGD.
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.prediction(X_batch, y_batch)
        self.loss_history.append(loss)

        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.prediction(X[start:end])
            scores[scores>0]=1
            scores[scores<0]=-1
            y_pred.append(scores[:,0])
        y_pred = np.hstack(y_pred)[:,None]
        acc = np.mean(y_pred == y)
        # print(acc,'acc')
        return acc

    def train(self):
        gamma0=self.optim_config["learning_rate"]
        d=10000
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            if t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                       t + 1, num_iterations, self.loss_history[-1]))

            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
            for k in self.optim_configs:
                self.optim_configs[k]['learning_rate']=gamma0/(1+gamma0/d*t)
            # print(self.optim_configs['learning_rate'])

            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.test_acc_history.append(val_acc)
                valloss=self.model.prediction(self.X_val, self.y_val)
                self.valloss_history.append(valloss[0])
                # self._save_checkpoint()

                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                           self.epoch, self.num_epochs, train_acc, val_acc))

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params
        # print(self.loss_history)
        # self.model.params = self.best_params
        return self.loss_history
        
# -------------setup--------------------#
if sys.argv[1] == "BP_question3":
  ThreeLayer=ThreeLayerNet(input_dim=(2),hidden_dim=2,outputDim=1)
  ThreeLayer.setWeight()
  x=np.array([1,1])[None,:]
  y=np.array([1])[None,:]
  ThreeLayer.prediction(X=x,y=y,printGrad=True)
elif sys.argv[1] == "3-NN": 
  trainData = pd.read_csv("./data/train.csv", header = None,names=['x1','x2','x3','x4','y'])
  testData = pd.read_csv("./data/test.csv", header = None,names=['x1','x2','x3','x4','y'])
  X_train= np.array(trainData.iloc[:,0:trainData.shape[1]-1].values)
  y_train= np.array(trainData.iloc[:,-1].values)[:,None]
  y_train[y_train==0]=-1
  X_test= np.array(testData.iloc[:,0:testData.shape[1]-1].values)
  y_test= np.array(testData.iloc[:,-1].values)[:,None]
  y_test[y_test==0]=-1
  minn=X_train.min(axis=0,keepdims=True)
  maxx=X_train.max(axis=0,keepdims=True)
  meann=X_train.mean(axis=0,keepdims=True)
  X_train=(X_train-minn)/(maxx-minn)
  X_test=(X_test-minn)/(maxx-minn)
  X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=1)
  # print(X_test.shape,X_train.shape,y_test.shape)
  # X_train=np.random.uniform(size=(872,4))
  # X_test=np.random.uniform(size=(500,4))
  # y_train=np.random.uniform(size=(872,1))
  # y_test=np.random.uniform(size=(500,1))
  model=ThreeLayerNet(input_dim=(4),hidden_dim=100,outputDim=1)

  data = {'X_train': X_train, 
          'y_train': y_train,
          'X_val': X_val,
          'y_val': y_val}
  batch_size=1
  solver = Solver(model, data,
                      update_rule='sgd',
                      optim_config={
                        'learning_rate': 0.1,
                      },
                      num_epochs=10, batch_size=batch_size,
                      print_every=872//batch_size)
  loss_history=solver.train()
  valloss_history=solver.valloss_history
  loss_history=valloss_history
  print("*"*100)
  acc_train=solver.check_accuracy(X_train,y_train)
  acc_test=solver.check_accuracy(X_test,y_test)
  print(1-acc_test,'err_test',1-acc_train,'err_train')
  # fig, ax = plt.subplots(figsize=(12,8))
  # ax.plot(np.arange(len(loss_history))+1, loss_history, 'r')
  # ax.set_xlabel('Epochs')
  # # ax.set_xscale("log")
  # ax.set_ylabel('loss')
  # ax.set_title('loss vs. Training Epoch')
  # plt.show()
  # solver.check_accuracy(X_train, y_train, num_samples=None, batch_size=1) 


