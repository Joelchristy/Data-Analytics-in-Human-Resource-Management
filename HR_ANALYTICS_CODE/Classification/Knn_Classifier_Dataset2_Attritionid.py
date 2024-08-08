"""
Created on Mon Feb 28 10:42:41 2022

@author: joelc
"""
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import f1_score
import argparse

def getFeatures(filename):
    df =pd.read_csv(filename)
    print(df.head())
    Y = df['Attritionid']
    print(Y)
    X=df.loc[:,['JobLevel', 'MonthlyIncome', 'StockOptionLevel', 'MaritalStatusid', 'OverTimeid','YearsInCurrentRole', 'TotalWorkingYears','Age', 'JobInvolvement', 'YearsSinceLastPromotion','DailyRate','TotalWorkingYears','YearsWithCurrManager']]

    X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.2, random_state=2) 
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
if __name__=="__main__":    
    parser = argparse.ArgumentParser(description='Prediction.')
    parser.add_argument('filename', help='Enter preprocessed csv file')
    
    args = parser.parse_args()
    filename = args.filename
    
    print(filename)
    selected_features = getFeatures(filename)    
    


