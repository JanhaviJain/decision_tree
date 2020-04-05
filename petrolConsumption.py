# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('petrol_consumption.csv',header=0)

#preparing dataset
X = dataset.drop('Petrol_Consumption', axis=1)
y = dataset['Petrol_Consumption']

#splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#training algo
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)         

#predicting results
y_pred = regressor.predict(X_test)
print(y_pred)

#visualizing results of one feature and label
Z=dataset.iloc[:,1:2].values #used for visualizing 
reg= DecisionTreeRegressor()
reg.fit(Z,y)
X_grid = np.arange(min(Z), max(Z), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(Z, y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
