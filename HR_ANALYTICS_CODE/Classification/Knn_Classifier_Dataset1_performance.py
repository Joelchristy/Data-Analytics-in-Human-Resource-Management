"""
Created on Mon Feb 28 10:42:41 2022

@author: joelc
"""
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns
import numpy as np
df=pd.read_csv('D:\Dataset1_Preprocessed_HRdataset.csv')

Y = df['PerfScoreID']
print(Y)
X=df.loc[:,['GenderID', 'EmpStatusID', 'PerfScoreID', 'EngagementSurvey', 'EmpSatisfaction','Salary','PositionID','RecruitmentSourceid','Absences']]
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.2, random_state=9) 
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
X_train= st_x.fit_transform(X_train)    
X_test= st_x.transform(X_test)  

classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(X_train, Y_train)  

Y_pred= classifier.predict(X_test)  
print(Y_pred)
print("Accuracy of KNN model is:",metrics.accuracy_score(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred, average='micro'))
   
  
cm = confusion_matrix(Y_test, Y_pred)


cf_matrix = confusion_matrix(Y_test, Y_pred)


print(cf_matrix)
ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                    fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['pip','need improvement','fully meets','exceeds'])
ax.xaxis.set_ticklabels(['pip','need improvement','fully meets','exceeds'])



