# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary libraries.
2.Read the dataset and separate the independent and dependent variables.
3.Split the dataset into training and testing.
4.Do preprocessing if needed, in this case vectorization is needed which is done using CountVectorizer()
5.Train the model using SVC() algorithm and .fit()
6.Predict the model on x_test.
7.Measure its accuracy

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SRINITHI.S
RegisterNumber:  212225040428
*/
```
```
    import pandas as pd
    data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")
    data.info()
    
    x=data['v2'].values
    y=data['v1'].values
    x.shape
    y.shape
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer()
    x_train=cv.fit_transform(x_train)
    x_test=cv.transform(x_test)
    x_train
    
    from sklearn.svm import SVC
    svc=SVC()
    svc.fit(x_train,y_train)
    y_pred=svc.predict(x_test)
    y_pred
    
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(y_test,y_pred)
    acc
```

## Output:
<img width="648" height="356" alt="image" src="https://github.com/user-attachments/assets/0164edc4-206e-4989-b93b-33a3e5366803" />
<img width="193" height="51" alt="image" src="https://github.com/user-attachments/assets/93e5ead2-2448-4a07-8b2a-8d67a2d378ac" />
<img width="806" height="105" alt="image" src="https://github.com/user-attachments/assets/bce1a3b4-0a72-4756-8f8d-3ded7dec764d" />
<img width="986" height="73" alt="image" src="https://github.com/user-attachments/assets/4723b815-dbae-47bb-977c-c3a65ae91e13" />
<img width="415" height="73" alt="image" src="https://github.com/user-attachments/assets/f602f180-28e7-4d1d-8faf-3d5d72f3c062" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
