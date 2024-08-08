# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:59:51 2022

@author: joelc
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
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
    chi_selector = SelectKBest(chi2, k=5)
    chi_selector.fit(Xnorm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    print(chi_feature)


if __name__=="__main__":    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('filename', help='Enter preprocessed csv file')
    parser.add_argument('attribute',help='Pass any one attribute name from the csv')
    args = parser.parse_args()
    filename = args.filename
    column_value = args.attribute
    print(filename,column_value)
    selected_features = getFeatures(filename,column_value)