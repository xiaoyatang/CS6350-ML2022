from data import Data
import numpy as np
import pandas as pd
# DATA_DIR = 'data/'
# data = np.loadtxt(DATA_DIR + 'kaggle/train_final.csv', delimiter=',', dtype = str)

# data_train = Data(fpath = DATA_DIR + 'car/train.csv')
# data_test = Data(fpath = DATA_DIR + 'car/test.csv')

LabelDim=-1

def cal_MajorityError(data_obj):
        """ calculate majority error
        """
        values, counts = np.unique(data_obj.raw_data[:,LabelDim], return_counts=True)
        tag_num=data_obj.raw_data.shape[0]
        if (len(counts)-1)*tag_num==0:
            return 0
        NumSum=np.sum(counts)
        majorityError=(NumSum-np.max(counts))/NumSum
        return majorityError

def cal_entropy(data_obj):
        """ calculate entropy
        """
        values, counts = np.unique(data_obj.raw_data[:,LabelDim], return_counts=True)
        tag_num=data_obj.raw_data.shape[0]
        if (len(counts)-1)*tag_num==0:
            return 0
        # pois_num=len(pois[0])/tag_num   
        #print(pois_num)
        # edi_num=len(edi[0])/tag_num
        # prob=np.array([pois_num,edi_num])
        prob=counts/np.sum(counts)
        return np.sum(-prob*np.log2(prob),0)
def cal_Gini(data_obj):
        """ calculate Gini
        """
        #print("using gini")
        values, counts = np.unique(data_obj.raw_data[:,LabelDim], return_counts=True)
        tag_num=data_obj.raw_data.shape[0]
        if len(counts)*tag_num==0:
            return 0
        # pois_num=len(pois[0])/tag_num   
        #print(pois_num)
        # edi_num=len(edi[0])/tag_num
        # prob=np.array([pois_num,edi_num])
        prob=counts/np.sum(counts)
        return 1-np.sum(prob*prob)
def cal_entropy_gain(data_obj,attribute):
    """ calculate information gain based on entropy
    """
    entropy_orign=cal_entropy(data_obj)
    entropy_after=0
    number_total=data_obj.raw_data.shape[0]
    for i in np.nditer(data_obj.attributes[attribute].possible_vals):
        data_subset = data_obj.get_row_subset(attribute, i)
        number_attr=data_subset.raw_data.shape[0]
        entropy=cal_entropy(data_subset)
        #print(entropy,i)
        if number_total==0:
            result=0
        else:
            result=entropy*number_attr/number_total
        entropy_after=entropy_after+result
    #print(entropy_orign,"ori",entropy_after)
    return entropy_orign-entropy_after

def cal_Gini_gain(data_obj,attribute):
    entropy_orign=cal_Gini(data_obj)
    entropy_after=0
    number_total=data_obj.raw_data.shape[0]
    for i in np.nditer(data_obj.attributes[attribute].possible_vals):
        data_subset = data_obj.get_row_subset(attribute, i)
        number_attr=data_subset.raw_data.shape[0]
        entropy=cal_Gini(data_subset)
        #print(entropy,i)
        if number_total==0:
            result=0
        else:
            result=entropy*number_attr/number_total
        entropy_after=entropy_after+result
    return entropy_orign-entropy_after

def cal_MajorityError_gain(data_obj,attribute):
    entropy_orign=cal_MajorityError(data_obj)
    entropy_after=0
    number_total=data_obj.raw_data.shape[0]
    for i in np.nditer(data_obj.attributes[attribute].possible_vals):
        data_subset = data_obj.get_row_subset(attribute, i)
        number_attr=data_subset.raw_data.shape[0]
        entropy=cal_MajorityError(data_subset)
        #print(entropy,i)
        if number_total==0:
            result=0
        else:
            result=entropy*number_attr/number_total
        entropy_after=entropy_after+result
    return entropy_orign-entropy_after

def Entroy_tree(data_obj,path):
    entropy_orign=cal_entropy(data_obj)
    entropy_diff=dict()
    entropy_diffmax=0
    entropy_diffind=0
    tag_num=data_obj.raw_data.shape[0]

    # pois=np.where(data_obj.raw_data[:,LabelDim]==LabelAlias["Pos"])
    # pois_num=len(pois[0])/tag_num
    # if pois_num>0.5:
    #     most_common=LabelAlias["Pos"]
    # else:
    #     most_common=LabelAlias["Neg"]
    values, counts = np.unique(data_obj.raw_data[:,LabelDim], return_counts=True)
    most_common=values[np.argmax(counts)]
    for i in data_obj.attributes:
        if i in path:
            continue
        #print(i,"attribute",cal_entropy_gain(data_obj,i),"entro")
        #print(cal_entropy_gain(data_obj,i),"**",i)
        if cal_entropy_gain(data_obj,i)>=entropy_diffmax:
            entropy_diffmax=cal_entropy_gain(data_obj,i)
            entropy_diffind=i 
    # if entropy_diffmax==0:
    #     entropy_diffind=i
    #if entropy_diffmax==0:

    # print(entropy_diffind,"secltion")
    return entropy_diffind,entropy_diffmax

def Gini_tree(data_obj,path):
    entropy_orign=cal_entropy(data_obj)
    entropy_diff=dict()
    entropy_diffmax=0
    entropy_diffind=0
    tag_num=data_obj.raw_data.shape[0]

    values, counts = np.unique(data_obj.raw_data[:,LabelDim], return_counts=True)
    most_common=values[np.argmax(counts)]
    
    for i in data_obj.attributes:
        if i in path:
            continue
        #print(i,"is the attribute","the information gain is",cal_Gini_gain(data_obj,i),"gini")
        #print(cal_entropy_gain(data_obj,i),"**",i)
        if cal_Gini_gain(data_obj,i)>entropy_diffmax:
            entropy_diffmax=cal_Gini_gain(data_obj,i)
            entropy_diffind=i 
    if entropy_diffmax==0:
        entropy_diffind=i
    return entropy_diffind,entropy_diffmax

def MajorityError_tree(data_obj,path):
    entropy_orign=cal_MajorityError(data_obj)
    entropy_diff=dict()
    entropy_diffmax=0
    entropy_diffind=0
    tag_num=data_obj.raw_data.shape[0]

    values, counts = np.unique(data_obj.raw_data[:,LabelDim], return_counts=True)
    most_common=values[np.argmax(counts)]
    
    for i in data_obj.attributes:
        if i in path:
            continue
        #print(i,"is the attribute","the information gain is",cal_Gini_gain(data_obj,i),"gini")
        #print(cal_entropy_gain(data_obj,i),"**",i)
        if cal_MajorityError_gain(data_obj,i)>entropy_diffmax:
            entropy_diffmax=cal_MajorityError_gain(data_obj,i)
            entropy_diffind=i 
    if entropy_diffmax==0:
        entropy_diffind=i
    return entropy_diffind,entropy_diffmax
def itera_tree(data_obj,string_,num_iter=0,maxi_iter=100,tree_decision=list(),method='entropy'):
    if num_iter==0:
        tree_decision=list()
    values, counts = np.unique(data_obj.raw_data[:,LabelDim], return_counts=True)
    most_common=values[np.argmax(counts)]
    string=string_[:]
    if num_iter >=maxi_iter:
        string=string+["label"]+[most_common]
        tree_decision.append(string)
        #print(string)
        #print("ia called")
        return tree_decision
    num_iter1=num_iter+1
    if method=='entropy':
        ntropy_diffind,entropy_diffmax=Entroy_tree(data_obj,string)
        #print("*****",ntropy_diffind,"is the selected attribute","maximum information gain is",entropy_diffmax,method)
    elif method=='Gini':
        ntropy_diffind,entropy_diffmax=Gini_tree(data_obj,string)
        #print("*****",ntropy_diffind,"is the selected attribute","maximum information gain is",entropy_diffmax,method)
    #print(ntropy_diffind)
    elif method=='MajorityError':
        ntropy_diffind,entropy_diffmax=MajorityError_tree(data_obj,string)
    else:
        raise 


    tag_num=data_obj.raw_data.shape[0]

    # print(tree_decision,ntropy_diffind,num_iter1)
    # print(data_obj.attributes[ntropy_diffind].possible_vals,"for ##",ntropy_diffind,num_iter1,"num")
    for i in data_obj.attributes[ntropy_diffind].possible_vals:
      #      print("$$$$")
            #print(i)
            string_1=string[:] 
            tag_num=data_obj.raw_data.shape[0]
            data_subset=data_obj.get_row_subset(ntropy_diffind, i)
            tag_num1=data_subset.raw_data.shape[0]
            #print(tag_num1)
            if tag_num1==0:
                #string_2=string_1+str(ntropy_diffind)+"->"+str(i)+"->"+str(most_common)
                #string_2=string_1+"->"+str(most_common)
                string_2=string_1+[ntropy_diffind]+[i]+["label"]+[most_common]
                tree_decision.append(string_2)
                #print(string_2)
                continue
            #string_2=string_1+str(ntropy_diffind)+"->"+str(i)+"->"
            string_2=string_1+[ntropy_diffind]+[i]
            #print(string_2,num_iter1)
            itera_tree(data_subset,string_2,num_iter1,maxi_iter,tree_decision=tree_decision)
    return    tree_decision


def conver2numpy(tree_dec):
    
    
    #b = np.array([len(a),len(max(a,key = lambda x: len(x)))])
    g=len(max(tree_dec,key = lambda x: len(x)))
    b =  [[ None for y in range( g ) ] for x in range(len(tree_dec))]
    for i,j in enumerate(tree_dec):

        b[i][0:len(j)] = j

    return np.array(b)
#tree_dec_np=conver2numpy(tree_dec)
def prediction(column_index_dict,tree_dec_np,test,i=0,label=None):
      
    attri=tree_dec_np[0,i]
    #print(i)

    #print(attri)
    if attri=="label":
        #print(tree_dec_np)
        #print(tree_dec_np[0,i+1])
        label=tree_dec_np[0,i+1]
        return label
    ind_test=column_index_dict[attri]
    # print(tree_dec_np,attri)
    ind_match=np.where(tree_dec_np[:,i+1]==test[ind_test])



    tree_dec_np_prun=tree_dec_np[ind_match,:][0,:,:]
    if len(ind_match[0])==0 or len(tree_dec_np_prun)==0:
        return tree_dec_np[0,-1]
    #print(tree_dec_np.shape)
    #print(tree_dec_np_prun.shape)
    #print(tree_dec_np_prun)
    label=prediction(column_index_dict,tree_dec_np_prun,test,i=i+2)
    return label


def batch_predicDetailed(data_obj2,tree_dec_np_entro):
    N=data_obj2.raw_data.shape[0]
    #print(N)
    array_pre=np.zeros(N)
    #print("begin")
    column_index_dict=data_train.column_index_dict
    for i in range(N):
        test=data_obj2.raw_data[i,:]
        #print(test[[0,-3]],"----->",prediction(tree_dec_np,test,i=0))
        #print(prediction(tree_dec_np_entro,test))
        if prediction(column_index_dict,tree_dec_np_entro,test)==test[LabelDim]:
            array_pre[i]=1
        else:
            pass
            #print("wrong for ",test)
    return array_pre

def batch_predictionArrayOutput(data_obj2,tree_dec_np_entro):
    N=data_obj2.raw_data.shape[0]
    #print(N)
    Prediction=np.zeros(N)
    #print("begin")
    column_index_dict=data_train.column_index_dict
    for i in range(N):
        test=data_obj2.raw_data[i,:]
        #print(test[[0,-3]],"----->",prediction(tree_dec_np,test,i=0))
        #print(prediction(tree_dec_np_entro,test))

        Prediction[i]=prediction(column_index_dict,tree_dec_np_entro,test)
    return Prediction

def batch_predicAccu(data_obj2,tree_dec_np_entro):
    array_pre=batch_predicDetailed(data_obj2,tree_dec_np_entro)
    return array_pre.mean()
def train_predict(data_obj,data_obj2,depth_limit=1,method='entropy'):
    tree_dec=itera_tree(data_obj,[],0,depth_limit,method=method)
    if len(tree_dec)==0:
        raise
    tree_dec_np=conver2numpy(tree_dec)
    return batch_predicAccu(data_obj,tree_dec_np),batch_predicAccu(data_obj2,tree_dec_np)



print("*"*50,"kaggle while treat unknown as feature")

DATA_DIR = 'data/'
data_train = Data(fpath = DATA_DIR + 'kaggle/train_final.csv')
data_test = Data(fpath = DATA_DIR + 'kaggle/test_final_removeID.csv')


##binarize feature according to median.
numericDimList=[0,2,4,10,11,12]
for dim in numericDimList:
    col=data_train.raw_data[:,dim].astype(np.int)
    median =np.median(col)
    data_train.raw_data[:,dim]=(col>median).astype(np.string_)
    testcol=data_test.raw_data[:,dim].astype(np.int)
    data_test.raw_data[:,dim]=(testcol>median).astype(np.string_)
data_test.resetPossibleAttr()
data_train.resetPossibleAttr()
LabelDim=-1
depth_option=[1,2,3,4,5,6,7]
# depth_option=[1,2]
methods=["entropy","Gini","MajorityError"]
print(methods)
for depth in depth_option:
    trainRes=[]
    for method in methods:
        tree_dec=itera_tree(data_train,[],0,depth,method=method)
        if len(tree_dec)==0:
            raise
        tree_dec_np=conver2numpy(tree_dec)
        accurary_train=batch_predicAccu(data_train,tree_dec_np)
        trainRes.append(str(round(1-accurary_train,3)))
        # print("method-{}, depth-{}, error {}".format(method,depth,1-TrainPrecision))
        predictionOut=batch_predictionArrayOutput(data_test,tree_dec_np)
        Output = pd.DataFrame({"ID":np.arange(len(predictionOut))+1,'Prediction': predictionOut})
        # Output.set_index("ID",inplace=True)
        # Output.to_csv(method+"DecisionTree.csv", index=False)  #decision tree d=3 entropy
        Output.to_csv(method+str(depth)+"LinearRegression.csv", index=False)  #decision tree d=3 entropy
    Res=["depth-"+str(depth)]+trainRes
    Res="&".join(Res)+"\\"+"\\"
    print(Res)
# methods=["entropy","Gini","MajorityError"]
# for i in range(len(depth_option)):
#     trainRes=[]
#     testRes=[]
#     for method in methods:
#         accurary_train,accurary_test=train_predict(data_train,data_test,depth_limit=depth_option[i],method=method)
#         # print(1-accurary_train,1-accurary_test)
#         trainRes.append(str(round(1-accurary_train,3)))
#         testRes.append(str(round(1-accurary_test,3)))
#     Res=["depth-"+str(depth_option[i])]+trainRes+testRes
#     Res="&".join(Res)+"\\"+"\\"
#     print(Res)

# ##Additionally convert unknown to most common one

# print("*"*50,"Bank while convert unknown to the most common atrribute of that feature")
# numericDimList=[1,3,8,15]
# for dim in numericDimList:
#     col=data_train.raw_data[:,dim]
#     values, counts = np.unique(col, return_counts=True)
#     counts[values=="unknown"]=0
#     most_common=values[np.argmax(counts)]
#     ind=col=="unknown"
#     data_train.raw_data[ind,dim]=most_common

#     testcol=data_test.raw_data[:,dim]
#     ind=testcol=="unknown"
#     data_test.raw_data[ind,dim]=most_common
#     print(most_common)
# data_test.resetPossibleAttr()
# data_train.resetPossibleAttr()
# LabelDim=-1

# methods=["entropy","Gini","MajorityError"]
# for i in range(len(depth_option)):
#     trainRes=[]
#     testRes=[]
#     for method in methods:
#         accurary_train,accurary_test=train_predict(data_train,data_test,depth_limit=depth_option[i],method=method)
#         # print(1-accurary_train,1-accurary_test)
#         trainRes.append(str(round(1-accurary_train,3)))
#         testRes.append(str(round(1-accurary_test,3)))
#     Res=["depth-"+str(depth_option[i])]+trainRes+testRes
#     Res="&".join(Res)+"\\"+"\\"
#     print(Res)