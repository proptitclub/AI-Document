
import os,sys
dir = os.getcwd()
sys.path.append(dir)
from deeplearning.loss.loss import CrossEntropy
from deeplearning.optimizer.optimize import Adagrad,StochasticGradien
from deeplearning.utils.utils import apply_batch_size

from deeplearning.metrics.metrics import accuracy_score

import abc
import timeit
import numpy as np



class Sequential():

    def __init__(self,**kwagrs):
        self.layers=[]
    def add(self,layer,input_shape=None,**kwagrs):
        self.layers.append(layer)

    def forward(self,X:np.ndarray):
        inputs=np.copy(X)
        for f in self.layers:
            inputs=f.forward(inputs)
        return inputs
    
    
    def compile(self,loss=CrossEntropy,optimizer=Adagrad,metric=accuracy_score):
        self.loss=loss
        self.metric=metric
        for f in self.layers:
            f.compile(optimizer)


    def backward(self,gradien_local=None):
        for f in reversed(self.layers):
            gradien_local=f.backward(gradien_local)

    def train_on_batch(self,X:np.ndarray,Y:np.ndarray):
        y_pred=self.forward(X)
        loss_grad=self.loss.backward(y_pred,Y)
        self.backward(gradien_local=loss_grad)

    def predict(self,X,Y):
        y_pred=self.forward(X)
        loss=self.loss.loss(y_pred,Y)
        acc=self.metric.predict(y_pred,Y)
        return (loss,acc)

    def fit(self,X,Y,activation=None,batch_size=64,epoch=15):

        history_model={
            'loss':[],
            'acc':[]
        }

        if activation is not None:
            history_model['val_loss']=[]
            history_model['val_acc']=[]

        if hasattr(self,'loss')==False:
            raise Exception('pls add loss or compile first')
        for i in range(epoch):
            start = timeit.default_timer()
            for (x_,y_) in apply_batch_size(X,Y,batch_size):
                self.train_on_batch(x_,y_)
            (loss,acc)=self.predict(X,Y)
            end=timeit.default_timer()
            history_model['loss'].append(loss)
            history_model['acc'].append(acc)
            if activation is not None:
                (loss2,acc2)=self.predict(activation[0],activation[1])
                history_model['val_loss'].append(loss2)
                history_model['val_acc'].append(acc2)
                print('epoches {}/{} in {:.3f} ss : loss {} - acc {} - val_loss {} - val_acc {}\n'.format(i,epoch,end-start,loss,acc,loss2,acc2))
            else:
                print('epoches {}/{} in {:.3f} ss : loss {} - acc {}\n'.format(i,epoch,end-start,loss,acc))
        return history_model