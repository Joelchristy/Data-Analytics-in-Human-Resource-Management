"""
Created on Mon Feb 28 10:42:41 2022

@author: joelc
"""
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import argparse
from sklearn.metrics import f1_score


def getFeatures(filename):
    df =pd.read_csv(filename)
    print(df.head())
    print(df.head())
    Y = df['PerformanceRating']
    print(Y)
    X=df.loc[:,['DistanceFromHome','EnvironmentSatisfaction','PercentSalaryHike','RelationshipSatisfaction','YearsInCurrentRole','JobInvolvement','MonthlyIncome','TotalWorkingYears','TrainingTimesLastYear','DailyRate','BusinessTravelid','Departmentid']]

    X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.4, random_state=4) 
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