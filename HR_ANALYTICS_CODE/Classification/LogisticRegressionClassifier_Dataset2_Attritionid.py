# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:11:38 2022

@author: joelc
"""


import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score
import argparse

def getFeatures(filename):
    df =pd.read_csv(filename)
    print(df.head())
    Y = df['Attritionid']
    print(Y)
    X=df.loc[:,['JobLevel', 'MonthlyIncome', 'StockOptionLevel', 'MaritalStatusid', 'OverTimeid','YearsInCurrentRole', 'TotalWorkingYears','Age', 'JobInvolvement', 'YearsSinceLastPromotion','DailyRate','TotalWorkingYears','YearsWithCurrManager']]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    digreg = linear_model.LogisticRegression()
    digreg.fit(X_train, Y_train)

    Y_pred = digreg.predict(X_test)
    print(Y_pred)


    print("Accuracy of Logistic Regression model is:",metrics.accuracy_score(Y_test, Y_pred))
    print(f1_score(Y_test, Y_pred, average='micro'))

    cf_matrix = confusion_matrix(Y_test, Y_pred)


    print(cf_matrix)
    ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.2%', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
if __name__=="__main__":    
    parser = argparse.ArgumentParser(description='Prediction.')
    parser.add_argument('filename', help='Enter preprocessed csv file')
    
    args = parser.parse_args()
    filename = args.filename
    
    print(filename)
    selected_features = getFeatures(filename)
     