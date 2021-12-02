Exploring data arrays with NumPy
***************************************
Suppose a college takes a sample of student grades for a data science class.

=============
data = [50,50,47,97,49,3,53,42,26,74,82,62,37,15,70,27,36,35,48,52,63,64]
print(data)

# To load the data into Numpy array

import numpy as np

grades = np.array(data)
print(grades)

=======
print (type(data),'x 2:', data * 2)
print('---')
print (type(grades),'x 2:', grades * 2)

You might have spotted that the class type for the numpy array above is a numpy.ndarray. The nd indicates that this is a structure that can consists of multiple dimensions (it can have n dimensions). Our specific instance has a single dimension of student grades.

grades.shape

grades[0]

grades.mean() # find the simple average grade (in other words, the mean grade value).

# Let's add a second set of data for the same students, this time recording the typical number of hours per week they devoted to studying.

# Define an array of study hours
study_hours = [10.0,11.5,9.0,16.0,9.25,1.0,11.5,9.0,8.5,14.5,15.5,
               13.75,9.0,8.0,15.5,8.0,9.0,6.0,10.0,12.0,12.5,12.0]

# Create a 2D array (an array of arrays)
student_data = np.array([study_hours, grades])

# display the array
student_data

# Show shape of 2D array
student_data.shape

# Show the first element of the first element
student_data[0][0]

# Get the mean value of each sub-array
avg_study = student_data[0].mean()
avg_grade = student_data[1].mean()

print('Average study hours: {:.2f}\nAverage grade: {:.2f}'.format(avg_study, avg_grade))


Exploring tabular data with Pandas
***************************************
******************************************
when you start to deal with two-dimensional tables of data, the **Pandas** package offers a more convenient structure to work with - the **DataFrame**.


import pandas as pd

df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie','Rhonda', 'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny',
'Jakeem','Helena','Ismat','Anila','Skye','Daniel','Aisha'],
                            'StudyHours':student_data[0],
                            'Grade':student_data[1]})

df_students 


Finding and filtering data in a DataFrame
***************************************
You can use the DataFrame's loc method to retrieve data for a specific index value

# Get the data for index value 5
df_students.loc[5]

# Get the rows with index values from 0 to 5
df_students.loc[0:5]


In addition to being able to use the loc method to find rows based on the index, you can use the iloc method to find rows based on their ordinal position in the DataFrame (regardless of the index):

# Get data in the first five rows
df_students.iloc[0:5]

# Get the values for the columns in positions 1 and 2 in row 0
df_students.iloc[0,[1,2]]

df_students.loc[0,'Grade']

# Here's another useful trick. You can use the loc method to find indexed rows based on a filtering expression that references named columns other than the index

df_students.loc[df_students['Name']=='Aisha']

df_students[df_students['Name']=='Aisha']

df_students.query('Name=="Aisha"')

df_students[df_students.Name == 'Aisha']

Loading a DataFrame from a file
===================================
We constructed the DataFrame from some existing arrays. However, in many real-world scenarios, data is loaded from sources such as files. Let's replace the student grades DataFrame with the contents of a text file.

df_students = pd.read_csv('data/grades.csv',delimiter=',',header='infer')
df_students.head()

# The DataFrame's read_csv method is used to load data from text files.

Handling missing values
========================
One of the most common issues data scientists need to deal with is incomplete or missing data. So how would we know that the DataFrame contains missing values? You can use the isnull method to identify which individual values are null

df_students.isnull()

# Of course, with a larger DataFrame, it would be inefficient to review all of the rows and columns individually; so we can get the sum of missing values for each column

df_students.isnull().sum()

# So now we know that there's one missing StudyHours value, and two missing Grade values.
# To see them in context, we can filter the dataframe to include only rows where any of the columns (axis 1 of the DataFrame) are null.

df_students[df_students.isnull().any(axis=1)]

# When the DataFrame is retrieved, the missing numeric values show up as NaN (not a number).

# One common approach is to impute replacement values. For example, if the number of study hours is missing, we could just assume that the student studied for an average amount of time and replace the missing value with the mean study hours. To do this, we can use the fillna method

df_students.StudyHours = df_students.StudyHours.fillna(df_students.StudyHours.mean())
df_students

# Alternatively, it might be important to ensure that you only use data you know to be absolutely correct; so you can drop rows or columns that contains null values by using the dropna method. In this case, we'll remove rows (axis 0 of the DataFrame) where any of the columns contain null values.

df_students = df_students.dropna(axis=0, how='any')
df_students

Explore data in the DataFrame
====================================
# Now that we've cleaned up the missing values, we're ready to explore the data in the DataFrame. Let's start by comparing the mean study hours and grades.

# Get the mean study hours using to column name as an index
mean_study = df_students['StudyHours'].mean()

# Get the mean grade using the column name as a property (just to make the point!)
mean_grade = df_students.Grade.mean()

# Print the mean study hours and mean grade
print('Average weekly study hours: {:.2f}\nAverage grade: {:.2f}'.format(mean_study, mean_grade))

# Filter the DataFrame to find only the students who studied for more than the average amount of time.
# Get students who studied for the mean or more hours
df_students[df_students.StudyHours > mean_study]

# find the average grade for students who undertook more than the average amount of study time.
# What was their mean grade?
df_students[df_students.StudyHours > mean_study].Grade.mean()

# We can use that information to add a new column to the DataFrame, indicating whether or not each student passed.
# First, we'll create a Pandas Series containing the pass/fail indicator (True or False), and then we'll concatenate that series as a new column (axis 1) in the DataFrame.

passes  = pd.Series(df_students['Grade'] >= 60)
df_students = pd.concat([df_students, passes.rename("Pass")], axis=1)
df_students

# You can use the groupby method to group the student data into groups based on the Pass column you added previously, and count the number of names in each group - in other words, you can determine how many students passed and failed.

print(df_students.groupby(df_students.Pass).Name.count())

# You can aggregate multiple fields in a group using any available aggregation function. For example, you can find the mean study time and grade for the groups of students who passed and failed the course.

print(df_students.groupby(df_students.Pass)['StudyHours', 'Grade'].mean())

# Many DataFrame operations return a new copy of the DataFrame; so if you want to modify a DataFrame but keep the existing variable, you need to assign the result of the operation to the existing variable. For example, the following code sorts the student data into descending order of Grade, and assigns the resulting sorted DataFrame to the original df_students variable.

# Create a DataFrame with the data sorted by Grade (descending)
df_students = df_students.sort_values('Grade', ascending=False)

# Show the DataFrame
df_students

*************
Visualizing data with Matplotlib
**********************************
DataFrames provide a great way to explore and analyze tabular data, but sometimes a picture is worth a thousand rows and columns. The Matplotlib library provides the foundation for plotting data visualizations that can greatly enhance your ability the analyze the data.

# A simple bar chart that shows the grade of each student.

# Ensure plots are displayed inline in the notebook
%matplotlib inline

from matplotlib import pyplot as plt

# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade)

# Display the plot
plt.show()
--------------
# Note that you used the pyplot class from Matplotlib to plot the chart. This class provides a whole bunch of ways to improve the visual elements of the plot. For example, the following code:

- Specifies the color of the bar chart.
- Adds a title to the chart (so we know what it represents)
- Adds labels to the X and Y (so we know which axis shows which data)
- Adds a grid (to make it easier to determine the values for the bars)
- Rotates the X markers (so we can read them)

# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade, color='orange')

# Customize the chart
plt.title('Student Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)

# Display the plot
plt.show()

-------------
# Create a Figure
fig = plt.figure(figsize=(8,3))

# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade, color='orange')

# Customize the chart
plt.title('Student Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)

# Show the figure
plt.show()
------------
# A figure can contain multiple subplots, each on its own axis.

# The following code creates a figure with two subplots - one is a bar chart showing student grades, and the other is a pie chart comparing the number of passing grades to non-passing grades.

# Create a figure for 2 subplots (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize = (10,4))

# Create a bar plot of name vs grade on the first axis
ax[0].bar(x=df_students.Name, height=df_students.Grade, color='orange')
ax[0].set_title('Grades')
ax[0].set_xticklabels(df_students.Name, rotation=90)

# Create a pie chart of pass counts on the second axis
pass_counts = df_students['Pass'].value_counts()
ax[1].pie(pass_counts, labels=pass_counts)
ax[1].set_title('Passing Grades')
ax[1].legend(pass_counts.keys().tolist())

# Add a title to the Figure
fig.suptitle('Student Data')

# Show the figure
fig.show()
---------
# The DataFrame provides its own methods for plotting data, as shown in the following example to plot a bar chart of study hours.

df_students.plot.bar(x='Name', y='StudyHours', color='teal', figsize=(6,4))

Getting started with statistical analysis
==========================================
A lot of data science is rooted in statistics, so we'll explore some basic statistical techniques.

Descriptive statistics and data distribution
------------------
When examining a variable (for example a sample of student grades), data scientists are particularly interested in its distribution (in other words, how are all the different grade values spread across the sample). The starting point for this exploration is often to visualize the data as a histogram, and see how frequently each value for the variable occurs.

# Get the variable to examine
var_data = df_students['Grade']

# Create a Figure
fig = plt.figure(figsize=(10,4))

# Plot a histogram
plt.hist(var_data)

# Add titles and labels
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the figure
fig.show()
-------------
Measures of central tendency
============================
To understand the distribution better, we can examine so-called measures of central tendency; which is a fancy way of describing statistics that represent the "middle" of the data. The goal of this is to try to find a "typical" value. Common ways to define the middle of the data include:

- The mean: A simple average based on adding together all of the values in the sample set, and then dividing the total by the number of samples.
- The median: The value in the middle of the range of all of the sample values.
- The mode: The most commonly occuring value in the sample set*.

# Let's calculate these values, along with the minimum and maximum values for comparison, and show them on the histogram.
Of course, in some sample sets , there may be a tie for the most common value - in which case the dataset is described as bimodal or even multimodal.
-----------
# Get the variable to examine
var = df_students['Grade']

# Get statistics
min_val = var.min()
max_val = var.max()
mean_val = var.mean()
med_val = var.median()
mod_val = var.mode()[0]

print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                        mean_val,
                                                                                        med_val,
                                                                                        mod_val,
                                                                                        max_val))

# Create a Figure
fig = plt.figure(figsize=(10,4))

# Plot a histogram
plt.hist(var)

# Add lines for the statistics
plt.axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
plt.axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
plt.axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
plt.axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
plt.axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

# Add titles and labels
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the figure
fig.show()
-----------------
# Another way to visualize the distribution of a variable is to use a box plot (sometimes called a box-and-whiskers plot). Let's create one for the grade data.

# Get the variable to examine
var = df_students['Grade']

# Create a Figure
fig = plt.figure(figsize=(10,4))

# Plot a histogram
plt.boxplot(var)

# Add titles and labels
plt.title('Data Distribution')

# Show the figure
fig.show()
--------------------------
The box plot shows the distribution of the grade values in a different format to the histogram. The box part of the plot shows where the inner two quartiles of the data reside - so in this case, half of the grades are between approximately 36 and 63. The whiskers extending from the box show the outer two quartiles; so the other half of the grades in this case are between 0 and 36 or 63 and 100. The line in the box indicates the median value.

It's often useful to combine histograms and box plots, with the box plot's orientation changed to align it with the histogram (in some ways, it can be helpful to think of the histogram as a "front elevation" view of the distribution, and the box plot as a "plan" view of the distribution from above.)
-------------
# Create a function that we can re-use
def show_distribution(var_data):
    from matplotlib import pyplot as plt

    # Get statistics
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                            mean_val,
                                                                                            med_val,
                                                                                            mod_val,
                                                                                            max_val))

    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize = (10,4))

    # Plot the histogram   
    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    # Add lines for the mean, median, and mode
    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    # Plot the boxplot   
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    # Add a title to the Figure
    fig.suptitle('Data Distribution')

    # Show the figure
    fig.show()

# Get the variable to examine
col = df_students['Grade']
# Call the function
show_distribution(col)
------------------
For example, the student data consists of 22 samples, and for each sample there is a grade value. You can think of each sample grade as a variable that's been randomly selected from the set of all grades awarded for this course. With enough of these random variables, you can calculate something called a probability density function, which estimates the distribution of grades for the full population.

The Pandas DataFrame class provides a helpful plot function to show this density.

def show_density(var_data):
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(10,4))

    # Plot density
    var_data.plot.density()

    # Add titles and labels
    plt.title('Data Density')

    # Show the mean, median, and mode
    plt.axvline(x=var_data.mean(), color = 'cyan', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.median(), color = 'red', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.mode()[0], color = 'yellow', linestyle='dashed', linewidth = 2)

    # Show the figure
    plt.show()

# Get the density of Grade
col = df_students['Grade']
show_density(col)
-------------------
As expected from the histogram of the sample, the density shows the characteristic 'bell curve" of what statisticians call a normal distribution with the mean and mode at the center and symmetric tails.

Now let's take a look at the distribution of the study hours data.

# Get the variable to examine
col = df_students['StudyHours']
# Call the function
show_distribution(col)
--------------
# Get the variable to examine
col = df_students[df_students.StudyHours>1]['StudyHours']
# Call the function
show_distribution(col)
-----------
 For example, the following code uses the Pandas quantile function to exclude observations below the 0.01th percentile (the value above which 99% of the data reside).

q01 = df_students.StudyHours.quantile(0.01)
# Get the variable to examine
col = df_students[df_students.StudyHours>q01]['StudyHours']
# Call the function
show_distribution(col)

Tip: You can also eliminate outliers at the upper end of the distribution by defining a threshold at a high percentile value - for example, you could use the quantile function to find the 0.99 percentile below which 99% of the data reside.
-----------------
Let's look at the density for this distribution.

# Get the density of StudyHours
show_density(col)

Typical statistics that measure variability in the data include:

- Range: The difference between the maximum and minimum. There's no built-in function for this, but it's easy to calculate using the min and max functions.
- Variance: The average of the squared difference from the mean. You can use the built-in var function to find this.
- Standard Deviation: The square root of the variance. You can use the built-in std function to find this.

for col_name in ['Grade','StudyHours']:
    col = df_students[col_name]
    rng = col.max() - col.min()
    var = col.var()
    std = col.std()
    print('\n{}:\n - Range: {:.2f}\n - Variance: {:.2f}\n - Std.Dev: {:.2f}'.format(col_name, rng, var, std))
-------------------
When working with a normal distribution, the standard deviation works with the particular characteristics of a normal distribution to provide even greater insight. Run the cell below to see the relationship between standard deviations and the data in the normal distribution.

import scipy.stats as stats

# Get the Grade column
col = df_students['Grade']

# get the density
density = stats.gaussian_kde(col)

# Plot the density
col.plot.density()

# Get the mean and standard deviation
s = col.std()
m = col.mean()

# Annotate 1 stdev
x1 = [m-s, m+s]
y1 = density(x1)
plt.plot(x1,y1, color='magenta')
plt.annotate('1 std (68.26%)', (x1[1],y1[1]))

# Annotate 2 stdevs
x2 = [m-(s*2), m+(s*2)]
y2 = density(x2)
plt.plot(x2,y2, color='green')
plt.annotate('2 std (95.45%)', (x2[1],y2[1]))

# Annotate 3 stdevs
x3 = [m-(s*3), m+(s*3)]
y3 = density(x3)
plt.plot(x3,y3, color='orange')
plt.annotate('3 std (99.73%)', (x3[1],y3[1]))

# Show the location of the mean
plt.axvline(col.mean(), color='cyan', linestyle='dashed', linewidth=1)

plt.axis('off')

plt.show()
--------------------------
# There's a built-in Describe method of the DataFrame object that returns the main descriptive statistics for all numeric columns.

df_students.describe()
-------------------------
******************
Comparing data
*******************
To examine your data to identify any apparent relationships between variables.

First of all, let's get rid of any rows that contain outliers so that we have a sample that is representative of a typical class of students. We identified that the StudyHours column contains some outliers with extremely low values, so we'll remove those rows.

df_sample = df_students[df_students['StudyHours']>1]
df_sample

Comparing numeric and categorical variables
=============================================
The data includes two numeric variables (StudyHours and Grade) and two categorical variables (Name and Pass). Let's start by comparing the numeric StudyHours column to the categorical Pass column to see if there's an apparent relationship between the number of hours studied and a passing grade.

To make this comparison, let's create box plots showing the distribution of StudyHours for each possible Pass value (true and false).

df_sample.boxplot(column='StudyHours', by='Pass', figsize=(8,5))
------------
Now let's compare two numeric variables. We'll start by creating a bar chart that shows both grade and study hours.

# Create a bar plot of name vs grade and study hours
df_sample.plot(x='Name', y=['Grade','StudyHours'], kind='bar', figsize=(8,5))
------------
from sklearn.preprocessing import MinMaxScaler

# Get a scaler object
scaler = MinMaxScaler()

# Create a new dataframe for the scaled values
df_normalized = df_sample[['Name', 'Grade', 'StudyHours']].copy()

# Normalize the numeric columns
df_normalized[['Grade','StudyHours']] = scaler.fit_transform(df_normalized[['Grade','StudyHours']])

# Plot the normalized values
df_normalized.plot(x='Name', y=['Grade','StudyHours'], kind='bar', figsize=(8,5))
-----------------
With the data normalized, it's easier to see an apparent relationship between grade and study time. It's not an exact match, but it definitely seems like students with higher grades tend to have studied more.

So there seems to be a correlation between study time and grade; and in fact, there's a statistical correlation measurement we can use to quantify the relationship between these columns.

df_normalized.Grade.corr(df_normalized.StudyHours)

The correlation statistic is a value between -1 and 1 that indicates the strength of a relationship. Values above 0 indicate a positive correlation (high values of one variable tend to coincide with high values of the other), while values below 0 indicate a negative correlation (high values of one variable tend to coincide with low values of the other). In this case, the correlation value is close to 1; showing a strongly positive correlation between study time and grade.

Note: Data scientists often quote the maxim "correlation is not causation". In other words, as tempting as it might be, you shouldn't interpret the statistical correlation as explaining why one of the values is high. In the case of the student data, the statistics demonstrates that students with high grades tend to also have high amounts of study time; but this is not the same as proving that they achieved high grades because they studied a lot. The statistic could equally be used as evidence to support the nonsensical conclusion that the students studied a lot because their grades were going to be high.
*****************
Another way to visualise the apparent correlation between two numeric columns is to use a scatter plot.
******************************
# Create a scatter plot
df_sample.plot.scatter(title='Study Time vs Grade', x='StudyHours', y='Grade')

We can see this more clearly by adding a regression line (or a line of best fit) to the plot that shows the general trend in the data. To do this, we'll use a statistical technique called least squares regression.

𝑦=𝑚𝑥+𝑏
 
In this equation, y and x are the coordinate variables, m is the slope of the line, and b is the y-intercept (where the line goes through the Y-axis).

In the case of our scatter plot for our student data, we already have our values for x (StudyHours) and y (Grade), so we just need to calculate the intercept and slope of the straight line that lies closest to those points. Then we can form a linear equation that calculates a new y value on that line for each of our x (StudyHours) values - to avoid confusion, we'll call this new y value f(x) (because it's the output from a linear equation function based on x). The difference between the original y (Grade) value and the f(x) value is the error between our regression line and the actual Grade achieved by the student. Our goal is to calculate the slope and intercept for a line with the lowest overall error.

Specifically, we define the overall error by taking the error for each point, squaring it, and adding all the squared errors together. The line of best fit is the line that gives us the lowest value for the sum of the squared errors - hence the name least squares regression.

Fortunately, you don't need to code the regression calculation yourself - the SciPy package includes a stats class that provides a linregress method to do the hard work for you. This returns (among other things) the coefficients you need for the slope equation - slope (m) and intercept (b) based on a given pair of variable samples you want to compare.
-------------
from scipy import stats

#
df_regression = df_sample[['Grade', 'StudyHours']].copy()

# Get the regression slope and intercept
m, b, r, p, se = stats.linregress(df_regression['StudyHours'], df_regression['Grade'])
print('slope: {:.4f}\ny-intercept: {:.4f}'.format(m,b))
print('so...\n f(x) = {:.4f}x + {:.4f}'.format(m,b))

# Use the function (mx + b) to calculate f(x) for each x (StudyHours) value
df_regression['fx'] = (m * df_regression['StudyHours']) + b

# Calculate the error between f(x) and the actual y (Grade) value
df_regression['error'] = df_regression['fx'] - df_regression['Grade']

# Create a scatter plot of Grade vs StudyHours
df_regression.plot.scatter(x='StudyHours', y='Grade')

# Plot the regression line
plt.plot(df_regression['StudyHours'],df_regression['fx'], color='cyan')

# Display the plot
plt.show()
----------------------
Note that this time, the code plotted two distinct things - the scatter plot of the sample study hours and grades is plotted as before, and then a line of best fit based on the least squares regression coefficients is plotted.

The slope and intercept coefficients calculated for the regression line are shown above the plot.

The line is based on the f(x) values calculated for each StudyHours value. Run the following cell to see a table that includes the following values:

The StudyHours for each student.
The Grade achieved by each student.
The f(x) value calculated using the regression line coefficients.
The error between the calculated f(x) value and the actual Grade value.
Some of the errors, particularly at the extreme ends, are quite large (up to over 17.5 grade points); but in general, the line is pretty close to the actual grades.
------------
# Show the original x,y values, the f(x) value, and the error
df_regression[['StudyHours', 'Grade', 'fx', 'error']]

Using the regression coefficients for prediction
===============================================
Now that you have the regression coefficients for the study time and grade relationship, you can use them in a function to estimate the expected grade for a given amount of study.

# Define a function based on our regression coefficients
def f(x):
    m = 6.3134
    b = -17.9164
    return m*x + b

study_time = 14

# Get f(x) for study time
prediction = f(study_time)

# Grade can't be less than 0 or more than 100
expected_grade = max(0,min(100,prediction))

#Print the estimated grade
print ('Studying for {} hours per week may result in a grade of {:.0f}'.format(study_time, expected_grade))
--------------------
So by applying statistics to sample data, you've determined a relationship between study time and grade; and encapsulated that relationship in a general function that can be used to predict a grade for a given amount of study time.

This technique is in fact the basic premise of machine learning. You can take a set of sample data that includes one or more features (in this case, the number of hours studied) and a known label value (in this case, the grade achieved) and use the sample data to derive a function that calculates predicted label values for any given set of features.









