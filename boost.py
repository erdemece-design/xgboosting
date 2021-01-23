# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 11:22:19 2021

@author: ASUS
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data=pd.read_csv("Churn_Modelling.csv")

#dependent and undepentden variable (x and y)
X=data.iloc[:,3:-1].values
y=data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder #from Categoric data to binary data
label_encoder=LabelEncoder()
X[:,1]=label_encoder.fit_transform(X[:,1]) # for geography
X[:,2]=label_encoder.fit_transform(X[:,2]) # for gender

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

one_hot = ColumnTransformer([("one_hot",OneHotEncoder(dtype=float),[1])],remainder="passthrough")
X=one_hot.fit_transform(X)
X=X[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)