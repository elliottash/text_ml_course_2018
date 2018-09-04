#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:15:33 2017

@author: elliott
"""

# setup
# set this to your working directory
WORKING_DIR = '/home/elliott/Dropbox/_Ash_Teaching/2018-09 - Bocconi - Text Data and ML/code'
import os
os.chdir(WORKING_DIR)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_pickle('X.pkl').toarray()
vocab = pd.read_pickle('vocab.pkl')
df1 = pd.read_csv('death-penalty-cases.csv')
Y = df1['citeCount'] > 0

# Getting started with Keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential() # create a sequential model
model.add(Dense(50, # output neurons in layer       
          input_dim=X.shape[1], # number of inputs
          activation='relu')) # activation function
model.add(Dense(50, activation='relu')) # hidden layer
model.add(Dense(1, activation='sigmoid')) # output layer
model.summary()

# Visualize a model

# Requires graphviz
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
dot = model_to_dot(model,
                   show_shapes=True,
                   show_layer_names=False)
SVG(dot.create(prog='dot', format='svg'))

from keras.utils import plot_model
plot_model(model, to_file='model.png')

# fit the model
model.compile(loss='binary_crossentropy', # cost function
              optimizer='adam', # use adam as the optimizer
              metrics=['accuracy']) # compute accuracy, for scoring

model_info = model.fit(X, Y, 
                      epochs=5,
                      validation_split=.2)

# these are the learned coefficients
model.get_weights()

# Plot performance by epoch
plt.plot(model_info.epoch,model_info.history['acc'])
plt.plot(model_info.epoch,model_info.history['val_acc'])
plt.legend(['train', 'val'], loc='best')

# form probability distribution over classes
Ypred_prob = model.predict(X)
Ypred = Ypred_prob > .5

# Save a model
model.save('keras-clf.pkl')

# load model
from keras.models import load_model
model = load_model('keras-clf.pkl')

# Regression model with R-squared
Yreg = np.log(1+df1['citeCount'])
model = Sequential() # create a sequential model
model.add(Dense(50, # output neurons in layer       
          input_dim=X.shape[1], # number of inputs
          activation='relu')) # activation function
model.add(Dense(50, activation='relu')) # hidden layer
model.add(Dense(1)) # output layer

from keras import backend as K
def r2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

model.compile(loss='mean_squared_error', # cost function
              optimizer='adam', # use adam as the optimizer
              metrics=[r2]) # compute r-squared
model_info = model.fit(X[:15000], Yreg[:15000], 
                      epochs=3)

from sklearn.metrics import r2_score
Ypred = model.predict(X[15000:])
r2_score(Yreg[15000:],Ypred[:,0])