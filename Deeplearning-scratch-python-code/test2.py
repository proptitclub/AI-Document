import numpy as np


from deeplearning.DNN.Sequential import Sequential

from deeplearning.DNN.layer import Densen,Relu,Softmax,Conv2d,Maxpool2D,Flatten
import keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)
y_train=keras.utils.np_utils.to_categorical(y_train,num_classes=10)
y_test=keras.utils.np_utils.to_categorical(y_test,num_classes=10)
print('load-data done')

model=Sequential()

model.add(Conv2d(strides=1,padding='same',kernel_size=3,filter=5,activation='relu'),input_shape=(28,28,1))
model.add(Maxpool2D())
model.add(Flatten())

model.add(Densen(activation='relu',units=1024))
model.add(Densen(activation='relu',units=512))
model.add(Densen(activation='softmax',units=10))

model.compile()

model.fit(x_train,y_train,activation=(x_test,y_test),batch_size=100,epoch=10)


