# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:13:01 2022

@author: joelc
"""


import pandas as pd
df = pd.read_csv('D:\HR_ANALYTICS_CODE\Input\Dataset2_Employee-Attrition.csv')
x=df

print(x['Attrition'].unique())
x['AttritionValue'] = pd.Categorical(x['Attrition'])
print(x['AttritionValue'].values)
x['Attritionid'] = x.AttritionValue.cat.codes
print(x['Attritionid'])

print(x['BusinessTravel'].unique())
x['BusinessTravelValue'] = pd.Categorical(x['BusinessTravel'])
print(x['BusinessTravelValue'].values)
x['BusinessTravelid'] = x.BusinessTravelValue.cat.codes
print(x['BusinessTravelid'])


print(x['Department'].unique())
x['DepartmentValue'] = pd.Categorical(x['Department'])
print(x['DepartmentValue'].values)
x['Departmentid'] = x.DepartmentValue.cat.codes
print(x['Departmentid'])


print(x['EducationField'].unique())
x['EducationFieldValue'] = pd.Categorical(x['EducationField'])
print(x['EducationFieldValue'].values)
x['EducationFieldid'] = x.EducationFieldValue.cat.codes
print(x['EducationFieldid'])

print(x['Gender'].unique())
x['GenderValue'] = pd.Categorical(x['Gender'])
print(x['GenderValue'].values)
x['Genderid'] = x.GenderValue.cat.codes
print(x['Genderid'])


print(x['MaritalStatus'].unique())
x['MaritalStatusValue'] = pd.Categorical(x['MaritalStatus'])
print(x['MaritalStatusValue'].values)
x['MaritalStatusid'] = x.MaritalStatusValue.cat.codes
print(x['MaritalStatusid'])

print(x['Over18'].unique())
x['Over18Value'] = pd.Categorical(x['Over18'])
print(x['Over18Value'].values)
x['Over18id'] = x.Over18Value.cat.codes
print(x['Over18id'])

print(x['OverTime'].unique())
x['OverTimeValue'] = pd.Categorical(x['OverTime'])
print(x['OverTimeValue'].values)
x['OverTimeid'] = x.OverTimeValue.cat.codes
print(x['OverTimeid'])

x = x.drop('Attrition',1)
x = x.drop('BusinessTravel',1)
x = x.drop('Department',1)
x = x.drop('EducationField',1)
x = x.drop('Gender',1)
x = x.drop('JobRole',1)
x = x.drop('MaritalStatus',1)
x = x.drop('Over18',1)
x = x.drop('OverTime',1)
x = x.drop('AttritionValue',1)
x = x.drop('BusinessTravelValue',1)
x = x.drop('DepartmentValue',1)
x = x.drop('EducationFieldValue',1)
x = x.drop('GenderValue',1)
x = x.drop('MaritalStatusValue',1)
x = x.drop('Over18Value',1)
x = x.drop('OverTimeValue',1)
x = x.drop('EmployeeNumber',1)
x = x.drop('StandardHours',1)
x = x.drop('EmployeeCount',1)
x = x.drop('Over18id',1)



x.to_csv("D:\HR_ANALYTICS_CODE\Preprocessing\Dataset2_Preprocessed_EmployeeAttrition.csv", index=False,sep=str(','), header=True)


