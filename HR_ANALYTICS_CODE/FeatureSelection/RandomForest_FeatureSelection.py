# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:55:42 2022

@author: joelc
"""

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import argparse

def getFeatures(filename, col_val):
    df =pd.read_csv(filename)
    print(df.head())
    y = df[col_val]
    print(y)
    X = df.drop(col_val,1)
    
  
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=5)
    embeded_rf_selector.fit(X, y)

    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    print(str(len(embeded_rf_feature)), 'selected features')
    print(embeded_rf_feature)
    
    
if __name__=="__main__":    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('filename', help='Enter preprocessed csv file')
    parser.add_argument('attribute',help='Pass any one attribute name from the csv')
    args = parser.parse_args()
    filename = args.filename
    column_value = args.attribute
    print(filename,column_value)
    selected_features = getFeatures(filename,column_value)
