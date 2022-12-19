import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.utils import shuffle
from xgboost import XGBClassifier
import warnings
from xgboost import cv
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# %matplotlib inline

trData = pd.read_csv("./data/train_final.csv", header = None,names=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','y'],skiprows=1)
teData = pd.read_csv("./data/removeID_test_final.csv", header = None,names=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14'],skiprows=1)

#----------declare first------#
# teData.isnull().sum() #checking for missing values and no.
#to specify all categorical features
# X_train= np.array(trData.iloc[:,0:trData.shape[1]-1])
# y_train= np.array(trData.iloc[:,-1])
# y_train[y_train==0]=-1
# X_train["x2"].astype("category") 
# X_train["x4"].astype("category") 
# X_train["x6"].astype("category") 
# X_train["x7"].astype("category") 
# X_train["x8"].astype("category") 
# X_train["x9"].astype("category") 
# X_train["x10"].astype("category") 
# X_train["x14"].astype("category") 
# teData["x2"].astype("category") 
# trData["x4"].astype("category") 
# teData["x6"].astype("category") 
# teData["x7"].astype("category") 
# teData["x8"].astype("category") 
# teData["x9"].astype("category") 
# teData["x10"].astype("category") 
# teData["x14"].astype("category") 

'''tree model or linear model'''
# clf=xgb.XGBClassifier(tree_method="gpu_hist",enable_categorical=True)
#supported tree method are "gpu_hist",'approx", and "hist"

# clf.fit(X_train, y_train)
# clf.save_model("categorical-model.json")

'''use numpy array to tell XGB the categorical columns'''
# ft=['q','c','q','c','q','c','c','c','c','c','q','q','q','c']
# X: np.ndarray=X_train
# assert X.shape[1]==14
# Xy=xgb.DMatrix(X_train,y_train,feature_types=ft,enable_categorical=True)

#---------------one-hot--------------#
def oneHot(data):
    data=data.drop(columns=['x14'])
    dataOneHot=pd.get_dummies(data=data,columns=['x2','x4','x5','x6','x7','x8','x9','x10'])#,'x14'
    return dataOneHot

X_train= trData.iloc[:,0:trData.shape[1]-1]
# X_test=teData
y_train= trData.iloc[:,-1]
# y_train= np.array(y_train)[:,None].astype(np.float32)
# y_train[y_train==0]=-1
data=pd.concat([X_train,teData],ignore_index=True)
data=oneHot(data)
# X_train=np.array(data.iloc[0:25000,:].values).astype(np.float32)
# X_test=np.array(data.iloc[25000:48842,:].values).astype(np.float32)
X_train=data.iloc[0:25000,:]
X_train_xy=pd.concat([X_train,y_train],axis=1)
X_test=np.array(data.iloc[25000:48842,:].values).astype(np.float32)

def data_shuffle(data):
    cols=data.shape[1]
    rows=data.shape[0]
    index_list=shuffle(list(range(data.shape[0])))
    dataNew=data.iloc[index_list]
    seed=600
    x_val = dataNew.iloc[0:seed,0:cols-1]
    y_val = dataNew.iloc[0:seed,-1]  
    x_tr = dataNew.iloc[seed:rows,0:cols-1]
    y_tr = dataNew.iloc[seed:rows,-1]
    y_val[y_val==0]=-1
    y_tr[y_tr==0]=-1
    return x_val,y_val,x_tr,y_tr
pos =X_train_xy[X_train_xy['y']==1]
neg =X_train_xy[X_train_xy['y']==0]
X_val_pos,y_val_pos,X_tr_pos,y_tr_pos=data_shuffle(pos)
X_val_neg,y_val_neg,X_tr_neg,y_tr_neg=data_shuffle(neg)
X_train=np.array(pd.concat([X_tr_pos,X_tr_neg],ignore_index=True).values).astype(np.float32)
y_train=np.array(pd.concat([y_tr_pos,y_tr_neg],ignore_index=True).values).astype(np.float32)
X_val=np.array(pd.concat([X_val_pos,X_val_neg],ignore_index=True).values).astype(np.float32)
y_val=np.array(pd.concat([y_val_pos,y_val_neg],ignore_index=True).values).astype(np.float32)

minnTr=X_train.min(axis=0,keepdims=True)
maxxTr=X_train.max(axis=0,keepdims=True)
meannTr=X_train.mean(axis=0,keepdims=True)
minnTe=X_test.min(axis=0,keepdims=True)
maxxTe=X_test.max(axis=0,keepdims=True)
X_train=(X_train-minnTr)/(maxxTr-minnTr)
X_test=(X_test-minnTe)/(maxxTe-minnTe)
print(X_test.shape,X_train.shape)
data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
y_train1=y_train
y_train1[y_train==-1]=0
data_dmatrix1 = xgb.DMatrix(data=X_train,label=y_train1)


# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.33, random_state = 7)
#---------rebalence the data-------#

params = {
            'objective':'binary:logistic',
            #It determines the loss function to be used in the process. For example, reg:linear for regression problems, reg:logistic for classification problems with only decision, binary:logistic for classification problems with probability.
            'max_depth': 10, #It determines how deeply each tree is allowed to grow during any boosting round.
            'alpha': 1, #It gives us the L1 regularization on leaf weights. A large value of it leads to more regularization.
            #lambda - It gives us the L2 regularization on leaf weights and is smoother than L1 regularization
            'learning_rate': 0.1,
            'scale_pos_weight':20,
            'n_estimators':500  #number of trees we wanna build
        }
eval_set = [(X_val, y_val)]
model= XGBClassifier(**params)
model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)
# xgb_clf.fit(X_train, y_train)
# params1 = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
#                 'max_depth': 8, 'alpha': 1}

# xgb_cv = cv(dtrain=data_dmatrix1, params=params1, nfold=3,
                    # num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=1)
y_train_pre = model.predict(X_train)
y_val_pre = model.predict(X_val)
predictions_val = [round(value) for value in y_val_pre]
predictions_train=[round(value) for value in y_train_pre]
accuracy_val = accuracy_score(y_val, predictions_val)
accuracy_train=accuracy_score(y_train,predictions_train)
print("val_Accuracy: %.2f%%" % (accuracy_val * 100.0))
print("train_Accuracy: %.2f%%" % (accuracy_train * 100.0))
# print(xgb_cv)
# y_train_pre=xgb_clf.predict(X_train)


# print({'XGBoost validation accuracy score: {0:0.4f}'. format(accuracy_score(y_val, y_val_pre)),'XGBoost training accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_train_pre))})

'''feature importance'''
# xgb.plot_importance(xgb_clf)
# plt.rcParams['figure.figsize'] = [6, 4]
# plt.show()
# xgb_clf.feature_importances_

y_test = model.predict(X_test)
Output = pd.DataFrame({"ID":np.arange(len(y_test))+1,'Prediction': y_test})
Output.to_csv("XGBcvrebalance_20_1000.csv", index=False) 
