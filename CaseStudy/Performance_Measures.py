Machine Learning
==========================
Machines learns from data
Accuracy -> 
Classification Model = gives yes/no prediction
Regression Model = gives the numerical value/range prediction

Performance Measures
===================
Classification : Predict category
- simple accuracy
- precision
- recall
- Fbeta measures
- ROC (and AUC)

Regression : Predict value
- Sum of squares error
- Mean absolute error
- RMS error
error = difference between actual value and predicted value

Accuracy = No. of samples predocted correctly/Total no of samples
(9990 = Non-Nike & 10 Nike --> Problem here is Non-Nike is predict correctly but Nike is not able to predict)
Precision = Of the shoes classified Nike, how many are actually Nike

Recall = Of the shoes that are actually Nike, how many are classified as Nike

No of shoes classified Nike = TP + FP
No of shoes actually Nike = TP 

Precision = TP/TP+FP

No of shoes actually Nike = TP + FN
No of shoes classified as Nike = TP

Recall = TP/TP+FN

Simple Accuracy = TP+TN/TP+TN+FP+FN

Accuracy = correctly identified predictions for each class/total dataset = diagonal values/Total no count = 9+15+24+15/80 = 0.78
Accuracy works well on balanced data.

F1 score = harmonic mean = 2*(precision* recall/Precision+Recall)

True Positive rate = Sensitivity or Recall
True Negative rate = Specificity

-> A confusion matrix is used to describe the performance of a classification model
True Positives(TP) = when classifier predicted TRUE & the correct value was TRUE
True Negative(TN) = when model predicted FALSE & correct class was FALSE
False Positive(FP) = Type I error = Classifier predicted TRUE & correct class was FALSE
False Negative(FN) = Type II error = Classifier predicted FALSE & correct class was TRUE


classification accuracy = (TP+TN)/(TP+TN+FP+FN)
Misclassification rate (Error rate) = (FP+FN)/(TP+TN+FP+FN)
Precision = TP/Total true predictions = TP/(TP+FP)
Recall = TP/Actual true = TP/(TP+FN)

#********************************************************
#******************************************************
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

print(os.getcwd())

os.chdir('C:\Users\kshaik\Documents\Khasim2020\Khasim2021\DataScience\')
print(os.getcwd())

# import warning filters
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore',category=FutureWarning)

df = pd.read_csv('heart_disease_predictio.csv')
df.head()

data = df
print("(Rows,columns):" +str(data.shape))
data.columns

data.nunique(axis=0) # returns the number of unique values for each variable

# summerizes the count,mean, standard deviation, min and max for numerical variables
data.describe()

print(data.isna().sum()) # verify the null values

# see if there is good proportion between +ve and -ve results, means we have a good balance between the two.
data['target'].value_counts()   # gives the +ve cases and -ve cases based on target column

# calculate correlation matrix
corr = data.corr()
plt.subplots(figsize(15,10))
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns, annot=True,
sns.heatmap(corr,xticklabels=corr.columns,
            yticklabels=corr.columns,
            annot=True,
            cmap=sns.diverging_palette(220,20, as_cmap=True))


subData = data[['age','trestbps','chol',thalach','oldpeak']]
sns.pairplot(subData)

# Filtering data by positive Heart disease patient
pos_data = data[data['target']==1]
pos_data.describe()

#Filter data by negative heart disease patient
neg_data = data[data['target']==0]
neg_data.describe()

print("(Positive patients ST depression): " + str(pos_data['oldpeak'].mean()))
print("(Negative patients ST depression): " + str(neg_data['oldpeak'].mean()))

print("(Positive patients thalach): " + str(pos_data['thalach'].mean()))
print("(Negative patients thalach): " + str(neg_data['thalach'].mean()))

# Modelling
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(random_state=1) # get instancec of model
model1.fit(x_train,y_train) #train fit model

y_pred1 = model1.predict(x_test)    #get y prediction
print(classification_report(y_test,y_pred1))    # output accuracy














