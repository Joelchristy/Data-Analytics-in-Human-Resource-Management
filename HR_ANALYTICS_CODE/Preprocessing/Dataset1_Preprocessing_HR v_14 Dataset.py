# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:33:30 2022

@author: joelc
"""


import pandas as pd
df = pd.read_csv('D:\HRDataset_v14.csv')
x=df
x.drop(x.iloc[:,[0,1,2,8,10,12,13,14,16,17,18,19,20,24,26,27,32,33,34]],axis=1,inplace = True)
print(x['Department'].unique())


x['DepartmentValue'] = pd.Categorical(x['Department'])
print(x['DepartmentValue'].values)
x['DepartmentId'] = x.DepartmentValue.cat.codes
print(x['DepartmentId'])

print(x['RecruitmentSource'].unique())
x['RecruitmentSourceValue'] = pd.Categorical(x['RecruitmentSource'])
print(x['RecruitmentSourceValue'].values)
x['RecruitmentSourceid'] = x.RecruitmentSourceValue.cat.codes
print(x['RecruitmentSourceid'])
x = x.drop('RecruitmentSourceValue',1)
x = x.drop('RecruitmentSource',1)

x = x.drop('DepartmentValue',1)
print(list(x.columns.values.tolist()))
x = x.drop(['Department','TermReason'],1)
x = x.drop(['DOB','PerformanceScore'],1)
x = x.drop('DateofTermination',1)
x = x.drop('DateofHire',1)
x = x.drop('age',1)
x = x.drop('year',1)
x = x.drop('month',1)
x.to_csv("D:\HRdataset.csv",index=False, sep=str(','), header=True)
