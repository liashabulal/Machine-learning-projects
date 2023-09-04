# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 12:37:59 2023

@author: liash
"""

import os
os.getcwd()
import pandas as pd

cm=pd.read_csv("Computer_Data.csv")
cm.head()
cm.info()
cm.isnull().sum()

import pingouin as pg

cm.anova(dv="price",between=['cd'])
cm.anova(dv="price",between=['multi'])
cm.anova(dv="price",between=['premium'])
cm.anova(dv="price",between=['ram'])

clean_cm=cm.drop(["multi","ram"],axis=1)

import scipy.stats

scipy.stats.pearsonr(cm['price'],cm['speed'])
scipy.stats.pearsonr(cm['price'],cm['hd'])
scipy.stats.pearsonr(cm['price'],cm['screen'])
scipy.stats.pearsonr(cm['price'],cm['ads'])
scipy.stats.pearsonr(cm['price'],cm['trend'])
scipy.stats.pearsonr(cm['price'],cm['ram'])

y=clean_cm[["price"]]
x=clean_cm.drop(["price"],axis=1)
x=pd.get_dummies(x)

pip install xgboost
import xgboost
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)


from xgboost import XGBRegressor
lm=XGBRegressor()
lm.fit(xtrain,ytrain)
prediction_value=lm.predict(xtest)

from sklearn.metrics import r2_score
r2_score(ytest,prediction_value)
