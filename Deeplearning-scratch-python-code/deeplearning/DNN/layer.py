import numpy as np
import abc

from deeplearning.utils.utils import initialize_parameters
from deeplearning.DNN.activation import Softmax,Relu
from deeplearning.utils.imgbetweencol import im2col_indices,col2im_indices,max_pool_forward_reshape,max_pool_backward_reshape
from deeplearning.DNN.bad_layer_cnn import *

class Layer(abc.ABC):

    def __init__(self,**kwagrs):
        if 'activation' not in kwagrs:
            self.activation='liner'
        else:
            self.activation=kwagrs['activation']
        self.parameter={
             'weights':None,
             'bias':None,
         }
        self.__dict__.update(kwagrs)

    @abc.abstractclassmethod
    def forward(self,inputs):
        raise Exception('overwire forward layer')

    @abc.abstractclassmethod
    def backward(self,div_local):
        raise Exception('overwire backward layer')


class Densen(Layer):
    def __init__(self,**kwagrs):
        super().__init__(**kwagrs)
    def initialize_parameters_layer(self,**kwagrs):

        prams={
        }
        if 'weights' in kwagrs:
            prams['weights']=kwagrs['weights']
        if 'bias' in kwagrs:
            prams['bias']=kwagrs['bias']

        self.parameter=initialize_parameters(**prams)
        self.shape=self.parameter['weights'].shape

    def compile(self,compile,**kwagrs):
        self.parameter_optimizer={}
        for name in self.parameter.keys():
            self.parameter_optimizer[name]=compile(**kwagrs)

    def get_shape(self):
        if hasattr(self,'parameter'):
            return self.shape
        return self.units
        
    def forward(self,inputs):
        self.inputs=inputs.copy()
        if self.parameter['weights'] is None:
            inits={
                'weights':(self.inputs.shape[-1],self.units),
                'bias':(1,self.units),
            }
            self.initialize_parameters_layer(**inits)

        
        out=np.dot(self.inputs,self.parameter['weights'])
        if self.parameter['bias'] is not None:
            out=np.add(out,self.parameter['bias'])
        if self.activation=='liner':
            return out
        if self.activation=='softmax':
            out=Softmax.forward(out)
            return out
        if self.activation=='relu':
            out=Relu.forward(out)
            return out
        raise Exception('{} has not implement activation'.format(self.activation))

    def backward(self,div_local):

        output_affter_fully=np.dot(self.inputs,self.parameter['weights'])
        if self.parameter['bias'] is not None:
            output_affter_fully=np.add(output_affter_fully,self.parameter['bias'])
        E=None
        
        if self.activation=='liner':
           E=div_local
        
        if self.activation=='softmax':
            E=np.multiply(div_local,Softmax.backward(output_affter_fully))
        
        if self.activation=='relu':
            E=np.multiply(div_local,Relu.backward(output_affter_fully))
        
        if E is None:
            raise Exception('{} has not implement activation'.format(self.activation))

        W=np.copy(self.parameter['weights'])
        dw=np.dot(self.inputs.T,E)
        db=np.sum(E,axis=0)
        self.parameter['weights']=self.parameter_optimizer['weights'].update(W,dw)
        if self.parameter['bias'] is not None:
            self.parameter['bias']=self.parameter_optimizer['bias'].update(self.parameter['bias'],db)
        return np.dot(E,W.T)








class Conv2d(Layer):

    def __init__(self,**kwagrs):
        super().__init__(**kwagrs)
        if 'strides' in kwagrs:
            self.strides=kwagrs['strides']
        else:
            self.strides=1
        self.kernel_size=kwagrs['kernel_size']
        if 'padding' in kwagrs:
            if kwagrs['padding'] =='same':
                self.pad=(self.strides+1)//2
            else:
                self.pad=kwagrs['padding']
         
    def __str__(self):
        return "Conv2D"
    def initialize_parameters_layer(self,**kwagrs):

        prams={
        }
        if 'weights' in kwagrs:
            prams['weights']=kwagrs['weights']
        if 'bias' in kwagrs:
            prams['bias']=kwagrs['bias']

        self.parameter=initialize_parameters(**prams)
        self.shape=self.parameter['weights'].shape


    def compile(self,compile,**kwagrs):

        self.parameter_optimizer={}
        for name in self.parameter.keys():
            self.parameter_optimizer[name]=compile(**kwagrs)


    def get_shape(self):
        if hasattr(self,'parameter'):
            return self.shape
        else:
            return self.fillter

    def get_output_shape(self,inputs):
        out_height = int((inputs.shape[1] + 2 *self.pad - self.kernel_size) / self.strides + 1)
        out_width = int((inputs.shape[2] + 2 * self.pad - self.kernel_size) / self.strides + 1)
        return self.filter,out_height,out_width

    def get_params(self):
        if hasattr(self,'parameter'):
            return self.parameter.copy()
        else:
            raise Exception('has not init params')
        
    def forward(self,inputs):

        self.inputs=inputs.copy()
        
        if self.parameter['weights'] is None:
            inits={
                'weights':(self.kernel_size,self.kernel_size,self.inputs.shape[-1],self.filter),
                'bias':(self.filter,1)
            }
            self.initialize_parameters_layer(**inits)

        #out=np.einsum("mabc,abcl->mabl", inputs, self.parameter['weights'])+self.parameter['bias']
        # first resize image if chanel is 3 dim

        self.inputs=self.inputs.transpose(0,3,1,2)
        
        self.inputs_faster=im2col_indices(self.inputs,field_height=self.kernel_size,field_width=self.kernel_size,padding=self.pad,stride=self.strides)

        W=self.parameter['weights'].reshape(self.filter,-1)
        bias=self.parameter['bias']

        out=np.dot(W,self.inputs_faster)+bias


        out=out.reshape(self.get_output_shape(inputs) + (inputs.shape[0], ))
        self.out=out.transpose(3,0,1,2)

        if self.activation=='relu':
            out=np.where(out>0,out,0)
        return out.transpose(3,0,1,2)


    def backward(self,div_local):
        

        
       
        W=self.parameter['weights'].reshape(self.filter,-1)

        if self.activation=='relu':
            div_local[self.out<0]=0

        div_local=div_local.reshape(self.filter,-1)
        dw=div_local.dot(self.inputs_faster.T).reshape(self.parameter['weights'].shape)
        db=np.sum(div_local,axis=1,keepdims=True)
        d_a=W.T.dot(div_local)

        d_a=col2im_indices(d_a,self.inputs.shape,field_height=self.kernel_size,field_width=self.kernel_size,padding=self.pad,stride=self.strides)

        self.parameter['weights']=self.parameter_optimizer['weights'].update(self.parameter['weights'],dw)

        if self.parameter['bias'] is not None:
            self.parameter['bias']=self.parameter_optimizer['bias'].update(self.parameter['bias'],db)

        return d_a

class Maxpool2D(Layer):

    def __init__(self,**kwagrs):
        super().__init__(**kwagrs)
        

        if 'kernel_size' in kwagrs:
            self.kernel_size=kwagrs['kernel_size']
        else:
            self.kernel_size=2
        if 'strides' in kwagrs:
            self.strides=kwagrs['strides']
        else:
            self.strides=self.kernel_size
        if 'padding' in kwagrs:
            if kwagrs['padding'] =='same':
                self.pad=0
            else:
                self.pad=kwagrs['padding']
        else:
            self.pad=0
    def __str__(self):
        return "Maxpool2D"
    def initialize_parameters_layer(self,**kwagrs):
        self.shape=kwagrs['weights'].shape
    def compile(self,compile,**kwagrs):
        pass


    def get_shape(self):
        return self.shape

    def get_params(self):
        if hasattr(self,'parameter'):
            return self.parameter.copy()
        else:
            raise Exception('Maxpool2D k co params')
        
    def forward(self,inputs):

        self.inputs=inputs.copy()
        self.inputs=self.inputs.transpose(0,2,3,1)
        pool=int(self.kernel_size)
        strides=int(self.strides)
        H=int(inputs.shape[2])
        W=int(inputs.shape[3])
        if pool==strides and H%pool==0 and W %pool ==0:
            # faster 
            out,need=max_pool_forward_reshape(inputs,pool_height=self.kernel_size,pool_width=self.kernel_size,stride=self.strides)
            self.need=need
            return out

        return cal_pool(self.inputs,kernel_size=self.kernel_size,strides=self.strides)

    def backward(self,div_local):
        if hasattr(self,'need'):
            dx= max_pool_backward_reshape(div_local,self.need)
            return dx
        d_a=pool_backward(div_local,self.inputs,strides=self.strides,kernel_size=self.kernel_size)

        return d_a.transpose(0,3,1,2)











class Flatten(Layer):
    def __init__(self,**kwagrs):
        super().__init__(**kwagrs)
    
        
    def __str__(self):
        return "Flatten"

    def initialize_parameters_layer(self,**kwagrs):
        prev_shape=kwagrs['weights']
        self.shape=(prev_shape[-3]*prev_shape[-2]*prev_shape[-1])

    def compile(self,compile,**kwagrs):
        pass


    def get_shape(self):
        return self.shape

    def get_params(self):
        if hasattr(self,'parameter'):
            return self.parameter.copy()
        else:
            raise Exception('Flatten k co params')
        
    def forward(self,inputs):

        self.inputs=inputs.copy()
        if hasattr(self,'shape')==False:
            inits={
                'weights':self.inputs.shape
            }
            self.initialize_parameters_layer(**inits)

        m,h,w,c=self.inputs.shape

        return self.inputs.reshape(m,h*w*c)



    def backward(self,div_local):
        d_a=div_local
        m,h,w,c=self.inputs.shape
        return d_a.reshape(m,h,w,c)
