#***************
# Logistic Regression
#***************

# Heart disease prediction
- range of conditions that affect your heart.
Diseases include blood vessel diseases, coronary artery disease, heart rhythm problems, heart defects your born with(congenital heart defects).
- Heart disease is one of the biggest causes of morbidity and mortality among the population of the world. Prediction of the cardiovascular disease is regarded as one of the most important subjects in the section of clinical data analysis. The amount of the data in the healthcare industry is huge.
- It is difficult to identify heart disease bcs of several contributory risk factors such as diabetes,high blood pressure,high cholesterol,abnormal pulse rate,and many other factors. Due to such constraints, scientists have turned towards modern approaches like Data mining and Machine Learning for predicting the disease.

- ML proves to be effective in assisting in making decisions and predictions from the large quantity of the data produced by the health industry.


Predictive Modeling
----------------------
- Building a Statistical model to predict the future behavior
- Predictive Modeling is Predictive analytics
- Popular techniques used for Predictive modelling:
Linear Regression
Logistic Regression
Classification & Regression Trees
Neural Networks
Naive Bayes Classifier

Build a Predictive Modeling
------------------------------
- Need 2 sets of variables
(Target or Response or Dependent variable = y)
(Predictor of Independent variable = x)

- Estimate a mathematical relationship between the Dependent & Independent variable

Eg:- Dependent var = Purchase the product -> yes or no
Independent var = Age, Gender, Salary, Marital status,saving ac balance, Mortgage...

Predictive Modeling process
---------------------------------
1) Understanding the Business problem -> Translate the Buss prob into Statistical prob -> Prepare the data specification & Get data 
i.e. Modify the dependent var then
Modify the statistical model for each predictive model (Algorithm...Logistic Regression)
Then Hypotheysis Predictive var (Independent var)

2) Take the data then split into 2.
Training data (70%)
Validation data (30%)
 & Finally develop a Modeling
 
3) Model deployment = Model Implementation.

#=============================
# Logistic Regression :
#==============================
LR predicts the probability of occurrence of an event
LR analyzes the relationship between a dichotomous(menas it gives yes/no results) dependent var & Independent var
LR will work on textual data

Logistic Regression - Equation
--------------------------------
ln[p/(1-p)] = alpha + BetaX + e

p = probability that the event Y occurs
ln[p/(1-p)] = logs odds ratio = or logit

Estimated probabilities to lie between 0 and 1

Estimated Probability = p = 1/[1+exp(-alpha -(beta *x)]

y = b0 + b1*x
p = 1/1+e power -y  # sigmoid function -> sigmoid curve
ln(p/1-p) = b0 + b1*x

# Dataset consists of 303 individual data. There are 14 columns in the dataset.
Age, Sex, chest-pain type, Resting Blood pressure, Serum cholestrol, Fasting Blood sugar, Resting ECG, Max heartrate achieved, Exercise induced angina, ST depression, Peak exercise ST segment, No of major vessels, Thal, Diagosis of heart disease

#**********

import os
print(os.getcwd())

os.chdir('C:\Users\kshaik\Documents\Khasim2020\Khasim2021\DataScience\')
print(os.getcwd())

import pandas as pd

# import warnings filter 
from warnings import simplefilter
#ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('cleveland_heart_disease.csv', header=None)
df.head()

df.columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
df.head()

df.isnull().sum()

df['target'].unique()

df['target'] = df.target.map({0:0,1:1,2:1,3:1,4:1}) # chaage the values 2 to 1, 3 to 3, 4 to 1.
df['target'].unique()

df['sex'].uinique()

df['sex'] =df.sex.map({0:'female',1:'male'})    # change 0 to female, 1 to male

df['sex'].unique()

df['thal'].isnull().sum()

df['thal'] = df.thal.fillna(df.thal.mean()) # fill the null column values with the mean
df['thal'].isnull().sum()

df['ca'].isnull().sum()

df['ca'] = df.ca.fillna(df.ca.mean())
df['ca'].isnull.sum()

import matplotlib.pyplot as plt 
import seaborn as sns

#distribution of target vs age
sns.set_context("paper",font_scale=2,rc={"font.size":20,"axes.titlesize":25,

sns.catplot(kind='count',data=df,x='age',hue='target',order=df['age'].

plt.title('Variation of Age for each target class')
plt.show()

#bar plot of age vs sex with hue=target
sns.catplot(kind='bar',data=df, y='age', x='sex', hue='target')
plt.title('Distribution of age vs sex with the target class')
plt.show()

df['sex'] = df.sex.map({'female':0,'male':1})
df['sex'].unique()

#######################Data processing
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_stat

from sklearn.preprocessing import StandardScaler as ss
sc=ss()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#***** LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred,y_test)
cm_test

y_pred_train = classifier.predict(x_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Logistic Regression={}'.format((cm_train[0][0])))
print('Accuracy for Test set for Logistic Regression = {}',format((cm_test[0][0] + 


                                                                                                                    

 


































