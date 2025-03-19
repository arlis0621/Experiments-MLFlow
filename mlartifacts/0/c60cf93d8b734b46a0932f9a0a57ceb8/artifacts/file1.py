
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
wine=load_wine()
x=wine.data
y=wine.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
max_depth=10
n_estimators=12
mlflow.set_tracking_uri('http://localhost:5000')

with mlflow.start_run():
    clf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)   
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    print("Accuracy:",accuracy)
    cm=confusion_matrix(y_test,y_pred)
    print("Confusion Matrix:\n",cm)
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm,annot=True,fmt='d',cmap='YlGnBu')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    
    
    
    

