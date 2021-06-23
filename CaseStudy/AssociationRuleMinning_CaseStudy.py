#***************
# Association Rule Minning
#***************

Is a popular and well researched method for discovering interesting relations between variables in large databases.

The rule found in the sales data of a supermarket would indicate that if a customer buys onions and potatoes together, he/she likely to buy hamburger meat.
- Such info is used as the basis for decisions about Marketing activities.(Eg: sush as Promotional pricing or Product placements).

#******************
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import matplotlib.pyplot as plt
from mlxtend.frequency_patterns import association_rules

import os
print(os.getcwd())
os.chdir('F:\\folder location")
print(os.getcwd())

data = pd.read_excel('Online Retail.xlsx')
data.head()

alst = data.columns
alst

alst = alst.str.strip()	# take each column name and convert into a string and strip any spaces.
alst

alst = alst.str.lower()
alst

alst = alst.str.replace(' ',,'-')
alst

alst = alst.str.replace('(','').str.replace(')','')
alst

data.head()

data.columns = data.columns.str.strip().str.lower().str.replace(' ','-').str.replace('(','').str.replace(')','')

data.head()

data.info() # gives meta data

print('Data dimension(row count, col count): {dim}'.format(dim=data.shape))

print('Count of unique invoice numbers: {cnt}'.format(cnt=data.invoiceno.nunique()))

print('Count of unique customer ids: {cnt}'.format(cnt=data.customerid.nunique()))
 
data['invoiceno'].value_counts() 

len(data)

data['invoiceno'] = data['invoiceno'].astype['str']
data.info()

print(len(data))

data.head()

data = data[~data[ 'invoiceno'].str.contains('C')]
print(len(data))

data['country'].unique()

len(data['country'].unique())

basket = (data[data['country'] == 'Australia'].groupby(['invoiceno','description'])['quantity'])

basket.head()

basket = (data[data['country'] == 'Australia'].groupby(['invoiceno','description'])['quantity']).sum()
basket.head(20)

basket = (data[data['country'] == 'Australia'].groupby(['invoiceno','description'])['quantity']).sum().unstack()

basket.head()

basket.shape()

basket.head()

basket = (data[data['country'] == 'Australia'].groupby(['invoiceno','description'])['quantity']).sum().unstack().reset_index()

basket = (data[data['country'] == 'Australia'].groupby(['invoiceno','description'])['quantity']).sum().unstack().reset_index().fillna(0).set_index('invoiceno'))

basket.head()

print(basket.shape)

def encode_units(x):
		if x <= 0:
			return 0
		if x >= 1:
			return 1

basket.head(2)

basket_sets = basket.applymap(encode_units)	# apply on each cell

basket_sets.head(2)

basket_sets[['POSTAGE']].head()

print(len(basket_sets.columns))

basket_sets.drop('POSTAGE', inplace=True, axis=1)	#droping the specific column
print(len(basket_sets.columns))

basket_sets.head(2)

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# generate the rules with their corresponding support,confidence and lift
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

df1 = rules[(rules[lift]>=6) & (rules[confidence]>=0.8)]
df1.head()

print(basket['RED RETROSPOT CAKE STAND'].sum())

print(basket['36 PENCILS TUBE RED RETROSPOT'].sum())








																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				