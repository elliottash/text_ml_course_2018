#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 21:50:09 2018

@author: elliott
"""

# set up
WORKING_DIR = '/home/elliott/Dropbox/_Ash_Teaching/2018-09 - Bocconi - Text Data and ML/code'
import os
os.chdir(WORKING_DIR)
import pandas as pd
df1 = pd.read_csv('death-penalty-cases.csv')

# our first dataset
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english',
                             max_features=4)
X = vectorizer.fit_transform(df1['snippet'])
words = vectorizer.get_feature_names()
print(words)
X

X = X.todense()
X = X / X.sum(axis=1) # counts to frequencies
for i, word in enumerate(words):
    column = X[:,i]
    df1['x_'+word] = column
df1.head()

# inspecting data
import numpy as np
df1['logcites'] = np.log(1+df1['citeCount'])
features = ['x_'+x for x in words]
cites_features = ['logcites'] + features
df2 = df1[cites_features]
corr_matrix = df2.corr()
corr_matrix['logcites'].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
scatter_matrix(df2)

df2.plot(kind='scatter', x='x_death', y='logcites', alpha = 0.1)

# create a test set 
from sklearn.model_selection import train_test_split
train, test = train_test_split(df2, test_size=0.2)

# our first machine learning model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
Xtrain = train[features]
Ytrain = train['logcites']
lin_reg.fit(Xtrain, Ytrain)
lin_reg.coef_

# in-sample performance
from sklearn.metrics import mean_squared_error
Ytrain_pred = lin_reg.predict(Xtrain)        
train_mse = mean_squared_error(Ytrain,Ytrain_pred)
train_mse

# out-of-sample performance
Xtest = test[features]
Ytest = test['logcites']
Ytest_pred = lin_reg.predict(Xtest)        
test_mse = mean_squared_error(Ytest,Ytest_pred)
test_mse

# missing values
judge = df1['author_id']
judge.fillna(0,inplace=True)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df2)
df2 = pd.DataFrame(X,columns=df2.columns)

# Encoding categorical variables
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
judge_fes = encoder.fit_transform(judge.values.reshape(-1,1))
judge_ids = ['j_'+str(x) for x in range(len(judge.unique()))]
judge_fes = pd.DataFrame(judge_fes.todense(),columns=judge_ids)
df1 = pd.concat([df1,judge_fes],axis=1)
train, test = train_test_split(df1, test_size=0.2)
df1['anycites'] = df1['citeCount'] > 0

# Cross-validation
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg,
                         df1[features],
                         df1['anycites'],
                         cv=3,
                         n_jobs=-1)
scores.mean()

# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [3, 10, 30],
              'max_features': [2, 4],
              'bootstrap': [True, False]}

grid_search = GridSearchCV(forest_reg, 
                           param_grid, 
                           cv=3)              
grid_search.fit(df1[features],df1['logcites'])

grid_search.best_params_
grid_search.best_score_

from sklearn.model_selection import RandomizedSearchCV
rand_search = RandomizedSearchCV(forest_reg, param_grid, cv=3)              
rand_search.fit(df1[features],df1['logcites'])

rand_search.best_params_

# Saving and loading
from sklearn.externals import joblib
joblib.dump(forest_reg,'forest_reg.pkl')
forest_reg = joblib.load('forest_reg.pkl')