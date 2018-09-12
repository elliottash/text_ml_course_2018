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
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_pickle('X.pkl').toarray()
vocab = pd.read_pickle('vocab.pkl')
df1 = pd.read_csv('death-penalty-cases.csv')
Y = df1['citeCount'] > 0
num_features = X.shape[1]

# Set up the basic model
from keras.models import Sequential
from keras.layers import Activation, Dense, Embedding
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

# output layer
model.add(Dense(1,activation='sigmoid'))

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
model.fit(X, Y, batch_size=128, 
           epochs=100, 
           callbacks=[earlystop], 
           validation_split=0.2)

# Batch Training with Large Data
from numpy import memmap
X_mm = memmap('X.pkl',shape=(32567, 472))

model.fit(X_mm, Y, batch_size=128, 
           epochs=3, 
           validation_split=0.2)

# Grid search with KerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# instantiate KerasClassifier with build function
def create_model(hidden_layers=1):  
    model = Sequential()
    model.add(Dense(16, input_dim=num_features, 
                    activation='relu')) 
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

# CNN example
# https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-visualize-word-embeddings-part-2-ca137a42a97d

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
num_words = 100 # use 20,000 in practice
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(df1['snippet'])
id2word = {v-1:k for k, v in tokenizer.word_index.items() if v <= 100}
sequences = tokenizer.texts_to_sequences(df1['snippet'])

# represent data as numrows x maxlen matrix
from keras.preprocessing.sequence import pad_sequences
maxlen = max([len(sent) for sent in sequences]) 
X = pad_sequences(sequences, maxlen=maxlen)
Xhot = to_categorical(X)
#X = np.expand_dims(X, axis=2) # needed for CNN input

from keras.layers import SeparableConv1D as Conv1D, GlobalMaxPooling1D
model = Sequential()
model.add(Conv1D(input_shape=(69,100),
                 filters=250, 
                 kernel_size=3)) # trigrams
model.add(GlobalMaxPooling1D())
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
raise
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(Xhot, Y, epochs=5)

raise
# to-do: implement Johnson and Zhang
# https://arxiv.org/pdf/1412.1058.pdf
# https://github.com/riejohnson/ConText
# https://github.com/tariqul-islam/Sequential-CNN
# https://www.kaggle.com/danielsafai/cnn-implementation-of-yoon-kim-s-model

# filter viz code
# https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

# weights on first trigram filter
first_filter = model.get_weights()[0][:,:,0]
first_filter.shape

# highest activating trigram
trigram = np.argmax(np.abs(first_filter),axis=1)
' '.join([id2word[i] for i in trigram])

# heatmap for filters
from seaborn import heatmap
for i in range(3):
    heatmap(model.get_weights()[0][:,:,i].T)
    plt.show()
    
# Recurrent Neural Networks
from keras.layers import SimpleRNN
model = Sequential()
model.add(Embedding(100, 32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.fit(X,Y,validation_split=.2)

# LSTM (Long-Short-Term-Memory) 
try:
    from keras.layers import CuDNNLSTM as LSTM
except:
    from keras.layers import LSTM

model = Sequential()
model.add(Embedding(20000, 100, input_length=X.shape[1]))
model.add(LSTM(100, recurrent_regularizer='l1_l2'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y,validation_split=.2)

# GRU
try:
    from keras.layers import CuDNNGRU as GRU
except:
    from keras.layers import GRU
    
model = Sequential()
model.add(Embedding(20000, 100, input_length=X.shape[1]))
model.add(GRU(100, recurrent_regularizer='l1_l2'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y,validation_split=.2)


# Autoencoder
from keras.layers import Input
data_dims = X.shape[1]
input_img = Input(shape=(data_dims,)) # input placeholder

# encoded: the compressed representation
encoded = Dense(32, activation='relu')(input_img)
# decoded: the lossy reconstruction
decoded = Dense(data_dims, activation='sigmoid')(encoded)

from keras.models import Model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Fit and validate
autoencoder.fit(X, X,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_split=.2)

Xpred = autoencoder.predict(X)