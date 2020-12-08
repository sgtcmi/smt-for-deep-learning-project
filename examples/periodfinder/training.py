# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 20:27:21 2020

@author: sgtbh
"""
import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset=input()
data = open(dataset,'r').read()

X,Y=eval(data)
print(Y[5:15])

X_train,X_test, Y_train,Y_test=train_test_split(X,Y, test_size=0.30, random_state=42)
Y_train=[[i]for i in Y_train ]
Y_test=[[i]for i in Y_test ]
#define the keras model

def dnn(input_dim,num_layer,epoch,batch_size,sparsity):
  model=Sequential()
  model.add(Dense(input_dim, input_dim=input_dim, activation='relu'))
  for i in range(input_dim,0,-(input_dim//num_layer)):
    model.add(Dense(i, activation='relu'))
  model.add(Dense(1, activation='relu'))
  # compile the keras model
  model.compile( loss=keras.losses.MeanSquaredError(),optimizer='adam')
  # fit the keras model on the dataset
  model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size)
  prediction=model.predict(X_test)
  count=0
  for i in range(len(Y_test)):
    if abs(prediction[i][0]-Y_test[i])<(sparsity/2):
      count+=1
    #else:
      #print(prediction[i][0],Y_test[i])  
  accuracy=(count//len(X_test))*100
  print(accuracy)
  return(model)

