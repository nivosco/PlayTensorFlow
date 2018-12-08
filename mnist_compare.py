# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:09:32 2018

@author: nivosco
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import time

class ml_model:
    def compile(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    def train(self,x_train,y_train,batch_size,epochs,x_test,y_test):
        self.history = self.model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(x_test, y_test))

class cnn_model(ml_model):
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

class dnn_model(ml_model):
    def __init__(self,input_shape, num_classes):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

class nn_model(ml_model):
    def __init__(self,input_shape,num_classes):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

class logistic_regression_model(ml_model):
    def __init__(self,input_shape,num_classes):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

def model_evaluate(models,train_time,epochs):
    fig, axs =plt.subplots(2,1)
    cell_text = [[models[0].model.count_params(),len(models[0].model.layers),epochs[0],train_time[0],models[0].history.history['val_acc'][9]],[models[1].model.count_params(),len(models[1].model.layers),epochs[1],train_time[1],models[1].history.history['val_acc'][49]],[models[2].model.count_params(),len(models[2].model.layers),epochs[2],train_time[2],models[2].history.history['val_acc'][199]],[models[3].model.count_params(),len(models[3].model.layers),epochs[3],train_time[3],models[3].history.history['val_acc'][499]]]
    collabel = ('Parameters', 'Layers','Epochs','Training time (sec)','Accuracy')
    rowlabel = ('CNN','DNN','NN','Logistic regression')
    axs[0].axis('tight')
    axs[0].axis('off')
    plt.title('MNIST comparison')
    axs[0].table(cellText=cell_text,colLabels=collabel,loc='center',rowLabels=rowlabel)
    axs[1].plot(range(1,epochs[0]+1), models[0].history.history['val_acc'][0:10],'r',label='cnn accuracy')
    axs[1].plot(range(1,epochs[0]+1), models[1].history.history['val_acc'][40:50],'g',label='dnn accuracy')
    axs[1].plot(range(1,epochs[0]+1), models[2].history.history['val_acc'][190:200],'c',label='nn accuracy')
    axs[1].plot(range(1,epochs[0]+1), models[3].history.history['val_acc'][490:500],'b',label='log regression accuracy')
    plt.title('MNIST Accuracy graph')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('compare.png')
    plt.show()

## Parameters
batch_size = 128
num_classes = 10
epochs = [10,50,200,500]
img_rows, img_cols = 28, 28
train_time = []

## Read the training set
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)

## build, train and evaluate the models
cnn_model = cnn_model((img_rows,img_cols,1),num_classes)
cnn_model.compile()
start_time = time.time()
cnn_model.train(x_train,y_train,batch_size,epochs[0],x_test,y_test)
train_time.append(time.time() - start_time)

dnn_model = dnn_model((img_rows,img_cols,1),num_classes)
dnn_model.compile()
start_time = time.time()
dnn_model.train(x_train,y_train,batch_size,epochs[1],x_test,y_test)
train_time.append(time.time() - start_time)

nn_model = nn_model((img_rows,img_cols,1),num_classes)
nn_model.compile()
start_time = time.time()
nn_model.train(x_train,y_train,x_train.shape[0],epochs[2],x_test,y_test)
train_time.append(time.time() - start_time)

lr_model = logistic_regression_model((img_rows,img_cols,1),num_classes)
lr_model.compile()
start_time = time.time()
lr_model.train(x_train,y_train,x_train.shape[0],epochs[3],x_test,y_test)
train_time.append(time.time() - start_time)

model_evaluate([cnn_model,dnn_model,nn_model,lr_model],train_time,epochs)
