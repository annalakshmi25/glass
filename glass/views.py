from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.staticfiles.storage import staticfiles_storage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def home(request):
    return render(request,'index.html',{"predicted":""})



def predict(request):
    exp1 = float(request.GET['exp1'])
    exp2 = float(request.GET['exp2'])
    exp3 = float(request.GET['exp3'])
    exp4 = float(request.GET['exp4'])
    exp5 = float(request.GET['exp5'])
    exp6 = float(request.GET['exp6'])
    exp7 = float(request.GET['exp7'])
    exp8 = float(request.GET['exp8'])
    exp9 = float(request.GET['exp9'])
    rawdata = staticfiles_storage.path('glass.csv')
    dataset = pd.read_csv(rawdata)
    X = dataset[["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"]]
    y = dataset["Type"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    yet_to_predict = np.array([[exp1,exp2,exp3,exp4,exp5,exp6,exp7,exp8,exp9]])
    y_pred = model.predict(yet_to_predict)
    accuracy = model.score(X_test, y_test)
    accuracy = accuracy*100
    accuracy = int(accuracy)
    return render(request,'index.html',{"predicted":y_pred[0],"exp1":exp1,"exp2":exp2,"exp3":exp3,"exp4":exp4,"exp5":exp5,"exp6":exp6,"exp7":exp7,"exp8":exp8,"exp9":exp9})