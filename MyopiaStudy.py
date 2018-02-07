#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:44:13 2018

@author: neck
"""

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
import numpy as np


seed = 9
np.random.seed(seed)

dataset = np.loadtxt('myopia.csv',delimiter = ";",skiprows = 1)
dataset = dataset[:,1:17]

Y = dataset[:,1]
X = np.delete(dataset,1,axis=1)


(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)

# create the model
model = Sequential()
model.add(Dense(15, input_dim=15, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=200, batch_size=10)#, validation_data=(X_test, Y_test)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))





