#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:15:33 2017

@author: elliott
"""

# set this to your working directory
WORKING_DIR = '/data/Dropbox/_Ash_Teaching/2018-09 - Bocconi - Text Data and ML/code'
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

X = pd.read_pickle('X.pkl').toarray()
vocab = pd.read_pickle('vocab.pkl')
df1 = pd.read_csv('death-penalty-cases.csv')
Y = df1['citeCount'] > 0

# Bagging classifier

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=50,
        max_samples=100, bootstrap=True, n_jobs=-1
    )

cross_val_score(bag_clf, X, Y).mean()

# random forest
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, 
                                 max_leaf_nodes=16, 
                                 n_jobs=-1)
y_pred_rf = cross_val_predict(rnd_clf, X, Y)                              
confusion_matrix(Y,y_pred_rf)

rnd_clf.fit(X,Y)
feature_importances = rnd_clf.feature_importances_
sorted(zip(feature_importances, vocab),reverse=True)[:20]


# XGBoost
from xgboost import XGBClassifier, XGBRegressor
dfX = pd.DataFrame(X,columns=vocab)
xgb_clf = XGBClassifier()
cross_val_score(xgb_clf, dfX, Y).mean()

xgb_reg = XGBRegressor(feature_names=vocab)
xgb_reg.fit(dfX,Y)

# Feature importance
sorted(zip(xgb_reg.feature_importances_, vocab),reverse=True)[:10]
from xgboost import plot_importance
plot_importance(xgb_reg, max_num_features=20)
