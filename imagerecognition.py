# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 10:30:01 2023

@author: liash
"""

import numpy as np
from sklearn.datasets import load_digits 

data1=load_digits()
print(data1.data)
print(data1.target)

print(data1.data.shape)
print(data1.images.shape)

data1_imagelen=len(data1.images)
print(data1_imagelen)

n=1222

import matplotlib.pyplot as plt 
plt.gray()
plt.matshow(data1.images[n])
plt.show()

data1.images[n]

x=data1.images.reshape((data1imagelen,-1))
y=data1.target

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=0)
print(xtrain.shape)
print(xtest.shape)

from sklearn import svm
model=svm.SVC(kernel='sigmoid')
model.fit(xtrain,ytrain)

n=1222
result=model.predict(data1.images[n].reshape((1,-1)))
plt.imshow(data1.images[n],cmap=plt.cm.gray_r,interpolation='nearest')
print(result)
print("/n")
plt.axis('off')
plt.title('%i' %result)
plt.show()

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,model.predict(xtest)))
