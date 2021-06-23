#***************
# Clustering
#***************

''' Organizing the data into clusters for :
High Intra-cluster similarities
Low Intra-cluster similarities
Natural groups among objects
'''
#****Clusting is For :
# Organizing data into clusters shows internal structure of the data -->(Eg: Medical domain)
# Sometimes partitioning is the goal -->(Eg: Marketting Segmentation - we grp the set of people as cluster & And find out the hidden patterns abt business)
# Prepare for other AI techniques
# Techniques for Clustering is useful in knowledge
# Desicovery in data (Eg: Underlining rules, hidden patterns...)

Eg:- Height vs Weight --> There are multiple clusters based on Height and Weight. There may be cluster of animals(like grp of cats, dogs, tallest animal cluster, heavy animal cluster...)

Runs vs Wikets  --> Good batsman, Good Bowlers, Good All rounder clusters based on Runs and Wikets data.

Vector
==========
A vector has 2 independent properties.
1) Magnitude
2) Direction

x-axis, y-axis,z-axiz --> All are vectors.

Similarity Measurement
============
Similarity by Correlation
Similarity by Distance

Similarity by Distance
==============
1) Euclidean distance = distance between 2 n-dimensional vectors 
d = sqrt((a1-b1)square + (a2-b2)square + .....+ (an-bn)square)

--> If the distance is less then they are similar. distance is more then they are dissimilar.

2) Manhatten distance = distance between 2 n-dimensional vectors 
d = |a1-b1| + |a2-b2| + ..... + |an-bn|

3) Cosine distance measure = distance between n-dimensional vectors 
d = 1 - (a1b1+a2b2+...+anbn)/sqrt(a1 square+a2 square+...+an square)* sqrt(b1 square+b2 square+...+bn square)

4) Tanimoto distance measure = distance between n-dimensional vectors 
d = 1 - (a1b1+a2b2+...+anbn)/sqrt(a1 square+a2 square+...+an square)+ sqrt(b1 square+b2 square+...+bn square)-(a1b1+a2b2+...+anbn)

K-Means Clustering
=======================
The process by which objects are classified into a number of groups so that they are as much dissimilar as possible from one group to another group but as much similar as possible within each group.
Cluster analysis means deviding the whole population into groups which are distinct between themselves but internally similar.

--> The attributes of the objects are allowed to determine which objects should be grouped together.

K-Means steps
==============
1) Define the number of groups - randomly generated within data domain
2) Then from every group centroid, we calculate the mean for all other groups centroid. The measurement is based on distance (techniques used are Euclidean/Manhatten/Cosine....)
3) Centroid of the each of the k clusters becomes the new mean.
4) Repeat steps 2,3 untill convergence has been reached.

#***********************************************************
#***************************************************************

#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   # advanced package for visualization
import os

print(os.getcwd())
os.chdir('C:\Users\kshaik\Documents\Khasim2020\Khasim2021\DataScience')

#import random
#import math

from sklearn.cluster import KMeans  # sklearn is the package for ML

dataset = pd.read_csv('Mall_Customers.csv')

df = dataset

df.head()   # gives first 5 records

df.tail() # gives last 5 records

df.info()   # gives info about no missing data

df.describe()   # gives descriptive statistic info like standard deviation, mean,median,....data

df.drop(["CustomerID"], axis=1, inplace=True)
df.head()

plt.figure(figsize=(10,6))
plt.title("Ages Frequency")
#sns.axes_style("dark")
sns.violinplot(y=df["Age"])
plt.show()

genders = df.Genre.value_counts()
genders

sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
sns.barplot(x=genders.index, y=genders.values)
plt.show()

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.boxplot(y=df["Spending Score (1-100)"],color="red")
plt.subplot(1,2,2)
sns.boxplot(y=df["Annual Income (k$)"])
plt.show()

age18_25 = df.Age[(df.age<=25) & (df.age>=18)]
age18_25
age26_35 = df.Age[(df.age<=35) & (df.age>=26)]
age36_45 = df.Age[(df.age<=45) & (df.age>=36)]
age46_55 = df.Age[(df.age<=55) & (df.age>=46)]
age55above = df.Age[(df.age>=56)]

x = ["18-25","26-35","36-45","46-55","55+"]

y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]
y

plt.figure(figsize(15,6))
sns.barplot(x=x,y=y,palette="rocket")
plt.title("Number of Customer and Age")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()

dataset = pd.read_csv('Mall_Customers.csv')

x = dataset.iloc[:,[3,4]].values
x

tmpDF = pd.DataFrame(x)
tmpDF

#Fitting K-Means to the dataset

kmeans = KMeans(n_clusters=5,init='k-means++',random_state=42)

y_kmeans = kmeans.fit_predict(x)

#Visualising the clusters   
plt.scatter(x[y_kmeans==0, 0],x[y_kmeans==0,1], s=100, c='red', label='cluster1')
plt.scatter(x[y_kmeans==1, 0],x[y_kmeans==1,1], s=100, c='blue', label='cluster2')
plt.scatter(x[y_kmeans==2, 0],x[y_kmeans==2,1], s=100, c='green', label='cluster3')
plt.scatter(x[y_kmeans==3, 0],x[y_kmeans==3,1], s=100, c='cyan', label='cluster4')
plt.scatter(x[y_kmeans==4, 0],x[y_kmeans==4,1], s=100, c='magenta', label='cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


plt.scatter(x[y_kmeans==0, 0],x[y_kmeans==0,1], s=100, c='red', label='Careless')
plt.scatter(x[y_kmeans==1, 0],x[y_kmeans==1,1], s=100, c='blue', label='Standard')
plt.scatter(x[y_kmeans==2, 0],x[y_kmeans==2,1], s=100, c='green', label='Target')
plt.scatter(x[y_kmeans==3, 0],x[y_kmeans==3,1], s=100, c='cyan', label='Sensible')
plt.scatter(x[y_kmeans==4, 0],x[y_kmeans==4,1], s=100, c='magenta', label='Careful')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

dataset["clusters"] = kmeans.labels_
dataset.head()
dataset.sample(5)

centers = pd.DataFrame(kmeans.cluster_centers_)
centers

#*********************************************************
