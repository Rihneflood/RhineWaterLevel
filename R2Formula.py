# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:44:43 2019

@author: Tung-Yu Lee
"""
# Input should contain Y1,Y2,Y3,Ypred1,Ypred2,Ypred3
import pandas as pd
import numpy as np


df = pd.read_csv(r'C:\Users\Tung-Yu Lee\Downloads\HackathonInformation\RhineWaterLevelStudentData\to_predict.csv')

Y1_Ypred1_squaredsum = np.sum((df['Y1'] - df['Ypred1'])**2)
Y2_Ypred2_squaredsum = np.sum((df['Y2'] - df['Ypred2'])**2)
Y3_Ypred3_squaredsum = np.sum((df['Y3'] - df['Ypred3'])**2)

Y_mean = (np.sum(df['Y1']) + np.sum(df['Y2']) + np.sum(df['Y3']))/7

Y1_Ymean = np.sum((df['Y1'] - Y_mean)**2) #df.rows.shape[0]
Y2_Ymean = np.sum((df['Y2'] - Y_mean)**2) #df.rows.shape[0]
Y3_Ymean = np.sum((df['Y3'] - Y_mean)**2) #df.rows.shape[0]


R_square = 1 - (Y1_Ypred1_squaredsum + Y2_Ypred2_squaredsum + Y3_Ypred3_squaredsum) / (Y1_Ymean + Y2_Ymean + Y3_Ymean)

