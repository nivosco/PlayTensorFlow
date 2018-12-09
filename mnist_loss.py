# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:09:32 2018

@author: nivosco
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(self.model.evaluate(self.validation_data[0], self.validation_data[1])[0])

class cnn_model:
    def __init__(self,input_shape, num_classes):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    def compile(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    def train(self,x_train,y_train,batch_size,epochs,x_test,y_test):
        self.lossHistory = LossHistory()
        self.model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(x_test, y_test), callbacks=[self.lossHistory])

def lossPlot(cnn_model):
    plt.plot(np.arange(0,len(cnn_model.lossHistory.losses)),cnn_model.lossHistory.losses,'r',label='Train loss')
    plt.plot(np.arange(0,len(cnn_model.lossHistory.val_losses)),cnn_model.lossHistory.val_losses,'b',label='Validation loss')
    plt.title('Loss function convergence graph')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
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
cnn_model = cnn_model((img_rows,img_cols,1),num_classes)
cnn_model.compile()
cnn_model.train(x_train,y_train,batch_size,epochs,x_test,y_test)

## plot the loss function
lossPlot(cnn_model)