#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:14:30 2017

@author: elliott
"""

# set this to your working directory
WORKING_DIR = '/home/elliott/Dropbox/_Ash_Teaching/2018-09 - Bocconi - Text Data and ML/code'
import os
os.chdir(WORKING_DIR)

import pandas as pd
df1 = pd.read_csv('death-penalty-cases.csv')
Xraw = pd.read_pickle('X.pkl')
vocab = pd.read_pickle('vocab.pkl')

###
# OLS Regression
###

# list of words from our vectorizer
vocab = [w.replace(' ', '_') for w in vocab]
         
# convert frequency counts to dataframe
df4 = pd.DataFrame(Xraw.todense(),
                   columns=vocab)

# import statsmodels package for R-like regression formulas
import statsmodels.formula.api as smf

# add metadata
df4['Y'] = df1['citeCount'] # cites to this opinion
df4['courtfe'] = df1['court_id']   # court fixed effect
df4['yearfe'] = df1['year']        # year fixed effect

# empty lists for t-statistics and coefficients
tstats, betas = [], []
for xvar in vocab: # loop through the words in vocab
    if any([c.isdigit() for c in xvar]) or 'hellip' in xvar:
        tstats.append(0)
        betas.append(0)
        continue
    model = smf.ols('Y ~ %s' % xvar,data=df4)                
    result = model.fit() 
    tstats.append(result.tvalues[1])
    betas.append(result.params[1])
            
# save estimates
pd.to_pickle(tstats,'tstats.pkl')    
pd.to_pickle(betas,'betas.pkl')

# zip up words and t-statistics
stats = list(zip(vocab,tstats))
stats.sort(key = lambda x: x[1], reverse=True) # sort by second item (tstats)
stats[:10] + stats[-10:]


# Overfitting
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import LinearRegression

m = 100
X = 6 * np.random.rand(m,1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m,1)
y = y.ravel()

from sklearn.preprocessing import PolynomialFeatures
poly_2 = PolynomialFeatures(degree=2) # also adds interactions
X_poly_2 = poly_2.fit_transform(X)

poly_300 = PolynomialFeatures(degree=300) 
X_poly_300 = poly_300.fit_transform(X)


lin_reg = LinearRegression()
cross_val_score(lin_reg, X, y, cv=3, n_jobs=3).mean()
cross_val_score(lin_reg, X_poly_2, y, cv=3, n_jobs=3).mean()
cross_val_score(lin_reg, X_poly_300, y, cv=3, n_jobs=3).mean()

###
# Regularized Regression
###

# Lasso
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X,y)

# Ridge
from sklearn.linear_model import Ridge, SGDRegressor
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X,y)

sgd_ridge_reg = SGDRegressor(penalty="l2",max_iter=1000)
sgd_ridge_reg.fit(X,y.ravel())

###
# Elastic Net
###
from sklearn.linear_model import ElasticNetCV
enet_reg = ElasticNetCV(alphas=[.01,.1,1], l1_ratio=[.01,.1,.5,.9, .99, 1])
enet_reg.fit(X,y)
enet_reg.alpha_, enet_reg.l1_ratio_

cross_val_score(enet_reg,X,y).mean()

# Scaling with Sparsity
from sklearn.preprocessing import StandardScaler
sparse_scaler = StandardScaler(with_mean=False)
X_sparse = sparse_scaler.fit_transform(Xraw)

# Multinomial Logistic
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(C=1, # default L2 penalty
                              class_weight='balanced')

scores = cross_val_score(logistic,
                         X_sparse,
                         df4['state'],
                         cv=3,
                         n_jobs=3)

scores.mean(), scores.std()