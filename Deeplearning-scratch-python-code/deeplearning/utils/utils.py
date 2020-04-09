import numpy as np
from matplotlib import pyplot as plt

def apply_batch_size(X,y,batch_size,random_suffe=True):
    if random_suffe==True:
        it =np.array(range(X.shape[0]))
        np.random.shuffle(it)
        for f in range(0,X.shape[0],batch_size):
            s=min(f+batch_size,X.shape[0])
            yield(X[it[f:s]],y[it[f:s]])  
    else:
        for f in range(0,X.shape[0],batch_size):
            yield (X[f:s],y[f:s])
            
def initialize_parameters(**kwagrs):
    parameter={}
    for key in kwagrs.keys():
        parameter[key]=np.random.normal(size=(kwagrs[key]))
    return parameter