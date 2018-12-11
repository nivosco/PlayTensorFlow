# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:09:32 2018

@author: nivosco
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class cnn_model:
    def __init__(self,input_shape, num_classes, params):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        if params[0] > 0:
            self.model.add(tf.keras.layers.Dropout(params[0]))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        if params[1] > 0:        
            self.model.add(tf.keras.layers.Dropout(params[1]))
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    def compile(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    def train(self,x_train,y_train,batch_size,epochs,x_test,y_test):
        self.history = self.model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(x_test, y_test))
    def getValidationAcc(self):
        return self.history.history['val_acc']

class model_params:
    def __init__(self):
        self.params = [[0.0,0.25,0.5,0.75],[0.0,0.25,0.5,0.75]]
        self.iterations = np.prod(np.array([len(x) for x in self.params]))
    def get_iterations(self):
        return self.iterations
    def get_params(self,iter):        
        param = []
        for i in range(len(self.params)):
            param.append(self.params[i][(int(iter/(2**i)))%2])
        return param
    
def SearchPlot(val_acc, params):
    plt.bar(np.arange(len(val_acc)), np.abs(np.log(val_acc)), align='center', alpha=0.5)
    plt.ylabel('Validation accuracy logarithm')
    plt.title('Search parameters')
    plt.savefig('search.png')
    plt.text(0.5,-0.6,'Best parameters are option {}: {}'.format(np.argmax(val_acc), params.get_params(np.argmax(val_acc))))
    plt.show()  

## Parameters
batch_size = 128
num_classes = 10
epochs = 1
img_rows, img_cols = 28, 28

## Read the training set
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,img_rows,img_cols,1)
x_test = x_test.reshape(-1,img_rows,img_cols,1)

## build, train and evaluate the models
params = model_params()
validation_acc = []
for i in range(params.get_iterations()):
    print('Iteration {}:'.format(i))
    model = cnn_model((img_rows,img_cols,1),num_classes, params.get_params(i))
    model.compile()
    model.train(x_train,y_train,batch_size,epochs,x_test,y_test)
    validation_acc.append(model.getValidationAcc())

## plot the results
SearchPlot(validation_acc, params)