# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:55:42 2022

@author: joelc
"""

import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import argparse
def getFeatures(filename):
    df =pd.read_csv(filename)
    print(df.head())
    Y = df['Salary']
    print(Y)
    X=df.loc[:,['MaritalStatusID','GenderID','Absences','RecruitmentSourceid','PositionID','PerfScoreID','DepartmentId','EmpStatusID','Absences','EngagementSurvey','EmpSatisfaction']]

#importing Train Test Split,Splitting Data
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
    #importing Random Forest Regressor
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 3)
     
    regressor.fit(X, Y) 
    X_pred = regressor.predict(X_train)
    print(X_pred)

    Y_pred = regressor.predict(X_test) 
    print(Y_pred)

    rms = sqrt(mean_squared_error(Y_pred,Y_test))
    print('RootMeanSquare',rms)
if __name__=="__main__":    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('filename', help='Enter preprocessed csv file')
    
    args = parser.parse_args()
    filename = args.filename
    
    print(filename)
    selected_features = getFeatures(filename)