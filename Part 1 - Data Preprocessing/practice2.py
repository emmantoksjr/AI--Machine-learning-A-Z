# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:32:58 2018

@author: user
"""

#import essential libraries 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#import dataset
dataset = pd.read_csv('Data.csv')

#separate the data to dependent and independent features
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#fill in the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
x[:,1:3] = imputer.fit_transform(x[:,1:3])

#convert string to numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#split data to training set and  test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
x_train = standardscaler.fit_transform(x_train)
x_test = standardscaler.transform(x_test)
