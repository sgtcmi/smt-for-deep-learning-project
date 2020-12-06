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
data = open('dataset.txt', 'r').read()

X,Y=eval(data)

X_train,X_test, Y_train,Y_test=train_test_split(X,Y, test_size=0.33, random_state=42)
#define the keras model


model = Sequential()
model.add(Dense(50, input_dim=1000, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='relu'))
# compile the keras model
model.compile(loss=keras.losses.MeanSquaredError(),optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, Y_train, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))



