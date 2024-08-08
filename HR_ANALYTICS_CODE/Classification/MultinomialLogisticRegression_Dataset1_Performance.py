# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:42:42 2022

@author: joelc
"""

import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

import argparse

from sklearn.metrics import f1_score


def getFeatures(filename):
    df =pd.read_csv(filename)
    print(df.head())

    Y = df['PerfScoreID']
    print(Y)
    X=df.loc[:,['GenderID','EmpStatusID','EngagementSurvey','EmpSatisfaction','RecruitmentSourceid','Salary','Absences']]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    digreg = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    digreg.fit(X_train, Y_train)

    Y_pred = digreg.predict(X_test)
    print(Y_pred)


    print("Accuracy of Logistic Regression model is:",metrics.accuracy_score(Y_test, Y_pred))
    print(f1_score(Y_test, Y_pred, average='micro'))
    
    
if __name__=="__main__":    
    parser = argparse.ArgumentParser(description='Prediction.')
    parser.add_argument('filename', help='Enter preprocessed csv file')
    
    args = parser.parse_args()
    filename = args.filename
    
    print(filename)
    selected_features = getFeatures(filename)


