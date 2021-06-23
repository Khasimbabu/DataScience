
Lifecycle of DataScience
==========================
1) Data Acquisition/Data Collection
2) Data Preparation
3) Hypothysis and Modeling
4) Evaluation and Interpretation
5) Deployment
6) Operation and Optimization

ExploratoryData Analysis(EDA) = Pre-Processing step to understand the data = Data Preparation, cleaning, wrangling...

#***********
CardioVascular diseases or heart diseases are the number one causes of death globally. Cardiovascular diseases are concertedly contributed by hypertension, diabetes, overweight and unhealthy lifestyles.
#************

#EDA
# Libraries for Exploratory Data Analysis
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import seaborn as sns 
sns.set_style('darkgrid')

import os
print(os.getcwd())
os.chdir('C:\Users\kshaik\Documents\Khasim2020\Khasim2021\DataScience\')

print(os.getcwd())

df = pd.read_csv('heart_disease_predictio.csv')
df.head()
df.shape
df.columns
df.info()

# To know the type of the variables
df.nunique()

df.dtypes

# change the categorical type to categorical variables
def['sex'] = df['sex'].astype('object')
def['cp'] = df['cp'].astype('object')
def['fbs'] = df['fbs'].astype('object')
def['restecg'] = df['restecg'].astype('object')
def['exang'] = df['exang'].astype('object')
def['slope'] = df['slope'].astype('object')
def['ca'] = df['ca'].astype('object')
def['thal'] = df['thal'].astype('object')

df.dtypes

df['ca'].unique()
df.ca.value_counts()

df[df['ca']==4]

df.loc[df['ca']==4,'ca'] = np.NaN
df[df['ca']==4]

df['ca'].unique()
df.ca.value_counts()

df.isnull.sum()

df.thal.unique()
df.thal.value_counts()
df.loc[df['thal']==0]
df.loc[df['thal']==0,'thal']=np.NaN
df.loc[df['thal']==0]

df.thal.unique()

# To check the missing values
df.isnull().sum()

df=df.fillna(df.median())
df.isnull().sum()

df.head()

# To check the duplicate values
duplicated = df.duplicated.sum()
duplicated

if duplicated:
	print("Duplicate rows in dataset are {}'.format(duplicated))
else:
	print('No duplicates')

duplicated = df[df.duplicated(keep=false)]
duplicated

df.describe()

# chage the labelling for better interpretation and visualization understanding

df['target']=df.target.replace({1:"Disease", 0:"No_Disease"})

df.head()

df['sex'] = df.sex.replace({1:"Male", 0:"Female"}) 

df['cp'] = df.cp.replace({1:"typical_angina",
2:"atyplical_angina",
3:"non-anginal pain",
4:"asymtomatic"})

df['exang'] = df.exang.replace({1:"Yes", 0:"No"})

df['slope'] = df.cp.replace({1:"upsloping",2:"flat",3:"Downsloping"})

df['thal'] = df.thal.replace({1:"fixed_defext",2:"reversable_defect",3:"normal"})

# To know the basic stats
df.describe()

df.describe(include='object')	#gives categorical data stat info

df.plot(kind='box',subplots=True, layout=(2,4),sharex=False,sharey=False,figsize=

sns.boxplot(x='target',y='chol',data=df)
sns.boxplot(x='target',y='oldpeak', data=df)	

# Define continuous variable & plot

continuous_features = ['age','trestbps','chol','thalach','oldpeak']
def outliers(df_out, drop=False):
	for each_feature in df_out.columns:
		feature_data = df_out[each_feature]
		Q1 = np.percentile(feature_data, 25.)	#25 percentile of the data
		Q3 = np.percentile(feature_data, 75.)	#75 percentile of the data
		IQR = Q3-Q1 #Interquartile range
		outlier_step = IQR *1.5
		outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (	feature_data
		
		if not drop:
			print('For the feature {}, No of Outliers is {}'.format(each_feature
		if drop:
			df.drop(outliers,inplace=True, errors='ignore')
			print('Outliers from {} feature removed'.format(each_feature))
outliers(df[continuous_features])

# Drop the outliers
outliers(df[continuous_features], drop=True)	# droping the outliers

outliers(df[continuous_features])	# checking the outliers

sns.boxplot(x='target',y='chol',data=df)
sns.boxplot(x='target',y='oldpeak', data=df)

# Distribution and relationship

print(df.target.value_counts())
df['target'].value_counts().plot(kind='bar').set_title('Heart disease Classes')

print(df.sex.value_counts())
df['sex'].value_counts().plot(kind='bar').set_title('sex destribution')

print(df.cp.value_counts())
df['cp'].value_counts().plot(kind='bar').set_title('Chest pain distribution')

print(df.restecg.value_counts())
df['restecg'].value_counts().plot(kind='bar').set_title('Resting ecg distribution')

print(df.exang.value_counts())
df['exang'].value_counts().plot(kind='bar').set_title('Excercise induced angina distribution')

print(df.ca.value_counts())
df['ca'].value_counts().plot(kind='bar').set_title('Number of major vessel distribution')

print(df.thal.value_counts())
df['thal'].value_counts().plot(kind='bar').set_title('thal distribution')

for column in df.select_dtypes(include='object'):
	if(df[column].nunique() < 10:
		sns.countplot(y=column, data=df)
		plt.show()

# visualize categorical data distribution
sns.countplot(x='sex', hue='target', data=df, palette='Set2').set_title('Disease classes according to sex')

sns.countplot(x='cp',hue='target', data=df, palette='Set2').set_title('Disease classes according to cp')

sns.countplot(x='thal',hue='target', data=df, palette='Set2').set_title('Disease classes according to thal')

sns.countplot(x='exang',hue='target', data=df, palette='Set2').set_title('Disease classes according to exang')

sns.countplot(x='fbs',hue='target', data=df, palette='Set2').set_title('Disease classes according to fbs')

sns.countplot(x='ca',hue='target', data=df, palette='Set2').set_title('Disease classes according to ca')

for column in df.select_dtypes(include='object'):
	if(df[column].nunique() < 10:
		sns.countplot(y=column, data=df)
		plt.show()
		
# visualize all together
# for ploting, group categorical features in cat_feat
# to create dist in 8 feature, 9th is the target
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(17,10))
cat_feat = ['sex','cp','fbs','restecg','exang','slope','ca','thal','target']

for idx,feature in enumerate(cat_feat):
	ax=axes[int(idx/3), idx%3]
	if feature!= 'target':
		sns.countplot(x=feature,hue='target',data=df,ax=ax,palette='Set2')
		
# Anotherway of visualizing: Pie charts for thalassemia having heart disease
labels='Normal','Fixed defect','Reversible defect'
sizes=[6, 130, 28]
colors=['pink','orange','purple']
plt.pie(sizes,labels=labels,colors=colors,autopct='%.1f%%', startangle=140)
plt.axis('equal')
plt.title('Thalassemia with heart disease')
plt.show()

# Not having heart disease
labels='Normal','Fixed defect','Reversible defect'
sizes=[12, 36, 89]
colors=['pink','orange','purple']
plt.pie(sizes,labels=labels,colors=colors,autopct='%.1f%%', startangle=140)
plt.axis('equal')
plt.title('Thalassemia with NO heart disease')
plt.show()

# visualize the distribution of continuous variable across target variable 
# define continuous variable & plot
continuous_features = ['age','chol','thalach','oldpeak','trestbps']
sns.pairplot(df[continuous_features + ['target']], hue='target')

# To understand the relationship between age and chol in each of the target based on sex
sns.implot(x='age',y='chol',hue='sex',col='target')
	markers=["o", "x"],
	palette="Set1",
	data=df)
plt.show()

# To understand the relationship between age and chol in each of the target based on target
sns.implot(x='age',y='chol',hue='target',col='sex')
	markers=["o", "x"],
	palette="Set2",
	data=df)
plt.show()

# Relation plot relplot 
sns.relplot(x='thalach',y='age', data=df)
sns.relplot(x='thalach',y='age',hue='sex',data=df)

# the correlations
sns.set(styles="white")
plt.rcParams['figure.figsize']=(15,10)
sns.heatmap(df.corr(), annot=True, linewidth=.5, cmap="Blues")
plt.title('Corelation between variables', fontsize=30)
plt.show()

# cp, thalach,slope - shows good positive corelation
# oldpeak,exang,ca,thal,sex,age - shows a good negative correlation
# fbs, chol,trestbps,restecg - Has low correlation with out target

# print(df.age.value_counts())
df['Age'].value_counts().plot(kind='bar').set_title('Age distribution')

# Analyse the distribution in age in range 10
print(df.age.value_counts()[:10])
sns.barplot(x=df.age.value_counts()[:10].index,
			y=df.age.value_counts()[:10].values,
			palette='Set2')
plt.xlabel('Age')
plt.ylabel('Age distribution')

# Most of the patients are in the age between 50s to 60s

# To know the youngest or oldest in age
print(min(df.age))
print(max(df.age))
print(df.age.mean())

# Gender distribution vs target

fig,ax = plt.subplots(figsize=(8,5))
name = df['sex']
ax = sns.countplot(x='sex',hue='target',data=df,palette='Set2')
ax.set_title('Sex distribution according to target', fontsize=13, weight='bold')
ax.set_xticklabels(name,rotation=0)

totals=[]
for i in ax.patches:
	totals.append(i.get_height())
total = sum(totals)
for i in ax.patches:
	ax.text(i.get_x()+.05, i.get_height()-15,
		str(round((i.get_height()/total)*100,2))+'%', fontsize=14, color='white',weight='bold')
plt.tight_layout()

#*************************************************************








                                                                                                                     














