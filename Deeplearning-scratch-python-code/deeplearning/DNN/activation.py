import numpy as np


class Softmax():
    def __init__(self):
        pass
    @classmethod
    def forward(self,inputs):
        e_x=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        e_x=e_x/np.sum(e_x,axis=1,keepdims=True)
        e_x=np.clip(e_x,1e-15,1-1e-15)
        return e_x
    @classmethod
    def backward(self,inputs):
        phi=Softmax.forward(inputs)
        return np.multiply(phi,1-phi)


class Relu():
    def __init__(self):
        pass
    @classmethod
    def forward(self,inputs):
        return np.where(inputs>0,inputs,0)
    @classmethod
    def backward(self,inputs):
        return np.where(inputs>=0,1,0)
