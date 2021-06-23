#****************************************
# Case Study - Demographic data & Pandas
#****************************************

print(os.getcwd())  # gives os is not defined error bcs os packages are not imported

import os
print(os.getcwd())  # gives current working directory (eg. c:\users\admin)

stats = pd.read_csv('C:\Users\kshaik\Documents\Khasim2020\Khasim2021\DataScience\DemographicData.csv')
# gives pd is not defined error bcs pandas package is not imported

import pandas as pd  # pandas is a package for Dataframe
stats = pd.read_csv('C:\Users\kshaik\Documents\Khasim2020\Khasim2021\DataScience\DemographicData.csv')

stats # gives dataframe, its a table like structure.

stats.head() # gives top 5 rows & all column values

stats.tail() # gives last 5 rows & all column values

os.chdir('C:\Users\kshaik\Documents\Khasim2020\Khasim2021') # change the working directory to new directory

print(os.getcwd()) # gives current working directory 'C:\Users\kshaik\Documents\Khasim2020\Khasim2021'

# If the file present in the current working directory location then you can give directly the file name instead of the complete path
stats = pd.read_csv('DemographicData.csv')

stats   # gives the dataframe

len(stats) # gives no of rows, length will give you the no of rows

stats.columns   # gives all the columns names

len(stats.columns)  # gives no of columns count

stats.head(3)   # gives top 3 rows & all columns

stats.info()    # gives all metadata 
#Dtype object means Textual data
#Dtype float menas Numerical data

stats.describe()    # gives you basic/descriptive statistics (like count, min, mean, std, max...)
# it gives statistics on numerical column, bcs you can't perform statistics on textual data

stats.describe.transpose() # transpose converts rows into columns & columns into rows.
# T is a short form of transpose

lst = stats.columns
lst # gives all column names.

stats['Country Name']   # gives CountryName column values.

stats.columns = (['CountryName','CountryCode','BirthRate','InternetUsers','IncomeGroup'])
# we place all the columns inside a list & removed the spaces in the column names
stats.head()

stats.CountryName

stats.CountryName.head()

stats['CountryName'].head() # gives CountryName column values

stats[['CountryName','BirthRate']].head()   # gives CountryName, BirthRate column values

stats[4:8][['CountryName','BirthRate']] # gives rows pos from 4 to 8 = From 5th row to 8th row 

stats[['CountryName','BirthRate']][4:8]

stats.BirthRate.head()

stats.InternetUsers.head()

result = stats.BirthRate * stats.InternetUsers 

result.head()

stats.head()

# How to add a new column

stats['MyCalc'] = stats.BirthRate * stats.InternetUsers

stats.head() # which will create and add a new column called 'MyCalc'

# How to delete a column

stats.drop('MyCalc',1) 
# 1 is axis=0 & 1 refers to cols

stats.head()

s2 = stats.drop('MyCalc',1)
s2.head()

stats = stats.drop('MyCalc',1)
stats.head()

stats['MyCalc'] = stats.BirthRate * stats.InternetUsers
stats.head()

stats.drop('MyCalc',1,inplace=True) # inplace=True modifying original dataframe, removed the column 'MyCalc'.
stats.head()

#***********
# Accessing rows
# loc = Accessing single or multiple rows by using integer - index based
# iloc = Accessing single or multiple rows by using label - (row or column label)
#**********

df = stats
df.head()

df.iloc[0]  # iloc = Indexed location = iloc[0] = gives the first row with all column values.

df.iloc[[0,1]]  # returns rows 0 & 1

df.iloc[[0,1],2] # returns rows 0 & 1 rows then select the 2nd column values from that.
# [[0,1],2] = [0,1] rows & 2nd column values will be returned

df.loc[0] # returns row 0

df.loc[[0,1]]   # returns 0,1 rows

df.loc[[0,1],['BirthRate','InternetUsers']]
# [[0,1],['BirthRate','InternetUsers']] = [0,1] rows & BirthRate, InternetUsers column values will be returned.

df.iloc[0:2] # gives 0,1 rows with all column values

df.iloc[:] # gives all rows & with all column values

df.iloc[:1] # gives 0th row with all column values

df.iloc[1:] # gives 1st row till the end rows.

df[0:4:2]   # gives 0th row & 2nd row = [0:4:2] = 0th row to 3rd row & 2 represents pickup every 2nd row in the dataframe. 

df[0:2]['CountryName']  # gives 0th, 1st rows & select CountryName column values
df.head()

df1 = df.set_index('CountryCode')   # setting the CountryCode values as index
df1

df1['ABW':'AGO']   # gives CountryCode value ABW rows to AGO rows 

df1['ABW':'AGO']['BirthRate']   # gives all rows froom ABW to AGO  & display only BirthRate column values

df1['ABW':'AGO'][['BirthRate','InternetUsers']]

type(df1['ABW':'AGO'][['BirthRate','InternetUsers']]) # gives pandas.core.frame.DataFrame

df.iloc[0]  # gives 0th row
df.iloc[0,1] # gives 0th, 1st rows

# multiple columns
df.iloc[[0,1],[0,1]]    #[[0,1],[0,1]] = 0th row, 1st row & display matching 0th column,1st column

df.iloc[[0][0,1]]   # gives 0th row & display matching 0th,1st columns

df.iloc[[0,1],[0]]  # gives 0th,1st rows & display matching 0th column values

df.iloc[[0,2]]    # gives 0th row, 2nd row 

df.iloc[::-1]   # :: means multiple of 1 = It gives reverse order = right to left = -1 to till the END

df1.iloc[0:4:2,0:2] # rows 0th to 3rd,pickup every 2nd row & columns 0th to 1st

df.loc[0]   # gives the first row values

df1.head()

df1.loc['ABW']  # gives the rows which matches index 'ABW'

df1.loc[0]  # will not working Bcs loc is index based.
df1.iloc[0] # will working  Bcs iloc is row/column label based.

df1.loc[['ABW','AFG']]  # gives ABW, AFG index matching rows

df1.loc[::2,'CountryName':'InternetUsers']  # gives CountryName to InternetUsers columns & select multiple of 2 index rows only.

































