import numpy as np

class Metric():
    def funcname(self, parameter_list):
        raise NotImplementedError
    def predict(self, y_pred,y,**kwagrs):
        raise NotImplementedError
class accuracy_score():
    '''
        >>> y_pred=[[0,0,1],[0,1,0]]
        >>> y_true=[[1,0,0],[0,1,0]]
        >>> accuracy_score(y_pred,y_true)
        0.5
    '''
    def funcname(self):
        return 'accuracy_score'
    def predict(y_pred:np.ndarray,y_true:np.ndarray):
        ''''''
        if y_pred.shape != y_true.shape:
            raise Exception('y_true = ({}),y_pred={}'.format(y_true.shape,y_pred.shape))
        return np.sum(np.argmax(y_pred,axis=1)==np.argmax(y_true,axis=1))/y_pred.shape[0]
    
