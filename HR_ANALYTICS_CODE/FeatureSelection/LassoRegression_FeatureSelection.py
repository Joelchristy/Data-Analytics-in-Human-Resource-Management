# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:18:43 2022

@author: joelc
"""

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse

def getFeatures(filename, col_val):
    df =pd.read_csv(filename)
    print(df.head())
    y = df[col_val]
    print(y)
    X = df.drop(col_val,1)



    Xnorm = MinMaxScaler().fit_transform(X)

    embeded_lr_selector = SelectFromModel(LogisticRegression(C=1, penalty='l1',solver='liblinear'),max_features=5)
    embeded_lr_selector.fit(Xnorm, y)

    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
    print(str(len(embeded_lr_feature)), 'selected features')
    print(embeded_lr_feature)
if __name__=="__main__":    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('filename', help='Enter preprocessed csv file')
    parser.add_argument('attribute',help='Pass any one attribute name from the csv')
    args = parser.parse_args()
    filename = args.filename
    column_value = args.attribute
    print(filename,column_value)
    selected_features = getFeatures(filename,column_value)