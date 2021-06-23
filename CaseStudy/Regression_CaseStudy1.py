#****************************
Linear Regression
#****************************

# import libraries
import numpy as np  # Arrays, Matrix pkg
import pandas as pd # DataFrame pkg
import matplotlib.pyplot as plt # Data visualization pkg

# load dataset
path = r"https://drive.google.com/uc?export=download&id=13ZTYmL3E8S0nz-UKl4aaTZJaI3DVBGHM"

df = pd.read_csv(path)

df  # returns the data frame

df.head()   # gives top rows

df.tail()   # gives bottom rows

df.info()   # gives metadata

df.describe()   # gives descriptive statistics

df.shape  # gives rows and columns

plt.scatter(x = df.study_hours, y = df.student_marks)   # data visualization
plt.xlabel("Students Study Hours")
plt.ylabel("Students marks")
plt.title("Scatter Plot of Students Study Hours vs Students marks")
plt.show()

#*****
Visualization give you correlation but not causation.
#*******

## Prepare the data for Machine Learning Algorithm

df.isnull().sum()   # data cleaning = checks any columns has null values # gives null value rows

df.mean()   # gives you avg of study hours & avg of students marks. It goes to each column, then sum and gives you avg study hrs.

df2 = df.fillna(df.mean())  #fillna = fill null values with mean() values
df2.isnull().sum() # after fillna, null values are updated. It returned 0 null values

df2.head()

# split dataset

x = df2.drop("student_marks", axis = "columns")
x
y = df2.drop("study_hours", axis = "columns")
y

print("shape of x = ", x.shape)
print("shape of y = ", y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=51)  # unpacking

# Algorithm will do all the mathematics & give the output i.e, called Model

# ==> We are giving 80% of data to Algorithm, Algorithm will create a Model for this 80% of data.

print("shape of x_train = ", x_train.shape)
print("shape of y_train = ", y_train.shape)
print("shape of x_test = ", x_test.shape)
print("shape of y_test = ", y_test.shape)

# select a Model and train it
# y = m*x + c

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train) # fit the data 80% train resulting 80% o/p
lr.coef_    # coefficient -> Tell the data direction

lr.intercept_   # Best fit line meeting the y-axis

lr.predict([[4]])[0][0].round(2)    # 4hrs study will get how many marks

lr.predict([[5]])[0][0].round(2)    # 5hrs study will get how many marks  

y_pred = lr.predict(x_test) # test = 20% of the data
y_pred

pd.DataFrame(np.c_[x_test,y_test,y_pred],columns=["study_hours","student_marks_original","student_marks_predicted"]

lr.score(x_test,y_test) # gives how much % accurate achieved = score will tell you Rsquare value.

plt.scatter(x_train,y_train)

plt.scatter(x_test,y_test)
plt.plot(x_train, lr.predict(x_train), color="r")

# save ML Model
import joblib

joblib.dump(lr,"student_mark_predictor.pkl")    # lr is a Model orject derived in above steps

model = joblib.load("student_mark_predictor.pkl")   # model

model.predict([[5]])[0][0]  # model prediction



































