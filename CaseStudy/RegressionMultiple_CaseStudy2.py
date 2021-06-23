#***************
Regression - Multiple
#***************
# ==> one dependent variable(Profit column) & multiple independent variables (R and D spend, Administration, MarketingSpend, state columns) = all these columns defined as variables.

'''
Profit of the company depends on other independent variables.

R&D spend = R&D spend cost
Administration = Infrastructure cost
Marketing spend = Marketing cost
state = location
Profit = profit of the company
'''

# Venture capitals wants to invest in a company based on the data. We design a model for selecting the best company.

#***** Multi line Regression *******

import pandas as pd # for dataframe
import numpy as np  # for arrays and matrises
import matplotlib.pyplot as plt # for visualization
from sklearn.linear_model import LinearRegression   # for m/c learning

import os
print(os.getcwd())

os.chdir('C:\Users\kshaik\Documents\Khasim2020\Khasim2021\DataScience')
print(os.getcwd())

import math
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

sns.set_style('whitegrid')

# importing the dataset
dataset = pd.read_csv('Startups.csv')
dataset

r_d = dataset.iloc[:,0]
r_d

x = r_d.values  # converting r_d into numpy array, bcs ML algorithms wil take array as input but not dataframes

x
type(x) 

prft = dataset.iloc[:,4]
prft

y = prft.values
y

import matplotlib.pyplot as plt
plt.scatter(x,y,label='',color='k',s=100)
plt.xlabel('R&D')
plt.ylabel('Profit')
plt.title('Profit vs R&D spend')
plt.legend()
plt.show()

admn = dataset.iloc[:,1]
admn

x = admn.values()
x

plt.scatter(x,y,label='',color='k')
plt.xlabel('Admin')
plt.ylabel('Profit')
plt.title('Profit vs Admin Spend')
plt.legend()
plt.show()

mrktng = dataset.iloc[:,2]
x = mrktng.values
plt.scatter(x,y,label='',color='k')
plt.xlabel('Marketing')
plt.ylabel('Profit')
plt.title('Profit vs Marketing Spend')
plt.legend()
plt.show()

x = dataset.iloc[:,3].values
plt.scatter(x,y,label='',color='k')
plt.xlabel('State')
plt.ylabel('Profit')
plt.title('Profit vs State')
plt.legend()
plt.show()

# Scatter Plot = we use scatter plot when x-axis, y-axis has numerical data. It does not fit when we have textual data.
# BoxPlot = when one of the x-axis or y-axis has the textual data then we use BoxPlot.

df = dataset.iloc[:,3:5]
df.boxplot(column='Profit',by='State') 

# boxplot = Middleline is representing Mean/Median
# Visualization = gives u correlation = correlation gives you a direction.
# correlation = is not a causation = To understand causation, we use Regression = Regression is a ML algorithm.

x = dataset.iloc[:,:-1].values
x

y = dataset.iloc[:,4].values
y

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state)

x_train.shape

# Fitting LinearRegression to the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train) # gives error bcs one of the column data is textual


x = dataset.iloc[:,:-2].values # removed the text data state column
x
x.shape

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state)
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# here we are giving the data to the Algorithm & we will get a model

y_pred = regressor.predict(x_test)
y_pred

print(regressor.coef_) # coefficient is nothing but causation = gives magnitude = profit/loss range

print(regressor.intercept_) # interception point

#***************************
#Dummy variables & Encoders
#****************************

from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

dataset2 = ['Pizza','Burger','Bread','Bread','Bread','Burger','Pizza','Burger']
dataset2

values = array(dataset2)    # list is converted into an array
print(values)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)   # label_encoder convert the textual data to numeric data
print(integer_encoded)

integer_encoded = integer_encoded.reshape(len(integer_encoded),1) # reshape convert the array into a Matrix
print(integer_encoded)

onehot = OneHotEncoder(sparse=False)

onehot_encoded = onehot.fit_transform(integer_encoded)
print(onehot_encoded)

x = dataset.iloc[:, :-1].values
x

y = dataset.iloc[:,4].values
y

label_encoder = LabelEncoder()
x
x.shape

x[:,3] = labelencoder.fit_transform(x[:,3])
x

print('----- OneHotEncoder ------')
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('state', OneHotEncoder(), [3])], remainder='passthrough')
x = ct.fit_transform(x) # to array
print(x)

x = x[:,1:]
x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state)
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
y_pred

print(regressor.coef_)


************************************************************************























 