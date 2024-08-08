# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:35:01 2022

@author: joelc
"""

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import argparse
def getFeatures(filename, col_val):
    df =pd.read_csv(filename)
    print(df.head())
    y = df[col_val]
    print(y)
    X = df.drop(col_val,1)
    
    
    Xnorm = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=5, step=10, verbose=5)
    rfe_selector.fit(Xnorm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    print(str(len(rfe_feature)), 'selected features')
    print(rfe_feature)


if __name__=="__main__":    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('filename', help='Enter preprocessed csv file')
    parser.add_argument('attribute',help='Pass any one attribute name from the csv')
    args = parser.parse_args()
    filename = args.filename
    column_value = args.attribute
    print(filename,column_value)
    selected_features = getFeatures(filename,column_value)