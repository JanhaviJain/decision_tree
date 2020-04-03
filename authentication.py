# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

#load data
col_names=['variance','skewness','curtosis','entropy','label']
dataset=pd.read_csv("bill_authentication.csv",header=0,names=col_names)

X=dataset.drop('label',axis=1)
y=dataset['label']

#splitting into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#training model
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

#predicting
y_pred = tree.predict(X_test)
print(y_pred)

#plotting a piechart
count0=0
count1=0
for i in y_pred:
    if i==1:
        count1=count1+1
    else:
        count0=count0+1
sizes=[count0,count1]        
labels='authentic' , 'fake'
colors=['gold' , 'lightcoral']
explode = (0.1, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True)        

#creating confusion matrix to evaluate performance of our model
c_matrix = metrics.confusion_matrix(y_test, y_pred)
c_matrix