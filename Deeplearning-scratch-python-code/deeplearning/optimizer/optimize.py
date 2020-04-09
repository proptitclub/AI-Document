import numpy as np
from matplotlib import pyplot as plt


class StochasticGradien():

    def __init__(self,lr=0.01,momemn=0): 
        self.lr=lr
        self.momen=momemn
        self.W_up=None
    def __str__(self):
        return "StochasticGradien"
    def update(self,w,w_grad):
        if self.W_up is None:
            self.W_up=np.zeros_like(w)
        self.W_up=self.momen*self.W_up+(1-self.momen)*w_grad
        return w-self.lr*self.W_up
    # 2 he so la lamda va eta nhung em đọc trên deeplearning.ai thì lamda+eta=1


class Adagrad():
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.G = None 
        self.eps = 1e-8
    def update(self, w, grad_wrt_w):
        if self.G is None:
            self.G = np.zeros(np.shape(w))
        self.G += np.power(grad_wrt_w, 2)
        return w - self.learning_rate * grad_wrt_w / np.sqrt(self.G + self.eps)
    
    #