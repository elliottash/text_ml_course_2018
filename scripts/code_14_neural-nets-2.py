#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:15:33 2017

@author: elliott
"""

# set this to your working directory
WORK_DIR = '/home/elliott/Dropbox/_Ash_Teaching/2018-09 - Bocconi - Text Data and ML/code'
import os
os.chdir(WORK_DIR)
import pandas as pd

X = pd.read_pickle('X.pkl').toarray()
vocab = pd.read_pickle('vocab.pkl')
df1 = pd.read_csv('death-penalty-cases.csv')
Y = df1['citeCount'] > 0
num_features = X.shape[1]

from keras.models import Sequential
from keras.layers import Activation, Dense

model = Sequential()
model.add(Dense(64, input_dim=num_features, activation='relu')) 

# initializers
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Dense(64, kernel_initializer='he_uniform'))

# other activation functions (https://keras.io/activations/)
model.add(Dense(64, activation="elu"))

# batch normalization
from keras.layers.normalization import BatchNormalization
model.add(Dense(64, use_bias=False)) 
model.add(BatchNormalization())
model.add(Activation('relu'))

# regularization
from keras.regularizers import l1, l2, l1_l2
model.add(Dense(64, 
                kernel_regularizer=l2(0.01),
                activity_regularizer=l1(0.01)))
model.add(Dense(64, 
                kernel_regularizer=l1_l2(l1=0.01,l2=.01),
                activity_regularizer=l1_l2(l1=0.01,l2=.01)))

# Dropout
from keras.layers import Dropout
model.add(Dropout(0.5))

model.add(Dense(1))

# Optimizers
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Early stopping
from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_acc', 
                          min_delta=0.0001, 
                          patience=5, 
                          mode='auto')
callbacks_list = [earlystop]
model.fit(X, Y, batch_size=128, 
           epochs=100, 
           callbacks=callbacks_list, 
           validation_split=0.2)

# Batch Training with Large Data
from numpy import memmap
X_mm = memmap('X.pkl',shape=(32567, 472))

model.fit(X_mm, Y, batch_size=128, 
           epochs=100, 
           callbacks=callbacks_list, 
           validation_split=0.2)

# Grid search with KerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# instantiate KerasClassifier with build function
def create_model(hidden_layers=1):  
    model = Sequential()
    model.add(Dense(16, input_dim=num_features, activation='relu')) 
    for i in range(hidden_layers):
        model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics= ['accuracy'])
    return model
clf = KerasClassifier(create_model)

# set of grid search CV to select number of hidden layers
params = {'hidden_layers' : [0,1,2,3]}
grid = GridSearchCV(clf, param_grid=params)
grid.fit(X,Y)
grid.best_params_

