# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:15:02 2019

@author: Tung-Yu Lee
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



df_model = pd.read_csv(r'C:\Users\Tung-Yu Lee\Downloads\HackathonInformation\RhineWaterLevelStudentData\MData_Koln2.csv')
#df_verify = pd.read_csv(r'C:\Users\Tung-Yu Lee\Downloads\HackathonInformation\RhineWaterLevelStudentData\VerifyData.csv')

copy = df_model.copy()


#split df for validation and model


Y = df_model.iloc[3:,1].copy()
df_model.drop('water_level_x', axis=1, inplace=True)

df_model = df_model.shift(-3)

df_model = df_model.dropna()

X_train, X_test, y_train1, y_test1 = train_test_split(df_model, Y, test_size=0.8, random_state=0)

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(X_train)

clf = SVR(kernel='rbf', C=100, epsilon=0.1,tol = 0.01)

clf.fit(x_scaled, y_train1)

X_verification_scaled = min_max_scaler.fit_transform(X_test)

Y_prediction = clf.predict(X_verification_scaled)

X_range = range(X_test.shape[0])

plt.figure(figsize=(20,20))
plt.scatter(X_range, y_test1, s=5, color="blue", label="original")
plt.plot(X_range, Y_prediction, lw=0.4, color="red", label="predicted")
plt.legend()
plt.show()


vary = clf.score(X_verification_scaled,y_test1)

score = r2_score(y_test1,Y_prediction)
