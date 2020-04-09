import numpy as np


class CrossEntropy():
    def __init__(self):
        pass
    def loss(y_pred,y):
        y_pred=np.clip(y_pred,1e-15,1-1e-15)
        return np.sum(-1*np.multiply(y,np.log(y_pred))-np.multiply(1-y,np.log(1-y_pred)))*1/y.shape[0]
    def backward(y_pred,y):
        p=np.clip(y_pred,1e-15,1-1e-15)
        return ((-y/p)+(1-y)/(1-p))*1/y.shape[0]

class MeanSquare():
    def __init__(self):
        pass

    def loss(y_pred,y):
        return np.linalg.norm(y_pred-y)**2/y.shape[0]
    def backward(y_pred,y):
        return (y_pred-y)/y.shape[0]
