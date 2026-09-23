import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("PelicanStores.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])

df= df.drop(columns=['Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12'])
print(df.columns)

'''First question'''
plt.bar(df['Type of Customer'],df['Net Sales'])
plt.xlabel('Type of Customer')
plt.ylabel('Net Sales')
plt.title('Distribution b/w Type of Customer and Net Sales')
plt.show()

'''Second question'''
plt.bar(df['Method of Payment'],df['Items'])
plt.xlabel('Method of payment')
plt.ylabel('Items')
plt.xlabel('Items')
plt.ylabel('Method of Payment')
plt.title('Items purchased by method of payment')
plt.show()

'''Third question'''
print(df[['Age','Marital Status']].head())
print(df[['Age','Marital Status']].dtypes)

print(df.shape)
print(df['Marital Status'].value_counts())

plt.figure(figsize = (8,6))
sns.violinplot(data=df,x='Marital Status',y='Age',inner='quartile')
plt.title('Age Distribution by Marital Status')
plt.show()

''''Fourth question'''

plt.figure(figsize = (8,6))
sns.scatterplot(data=df,x='Age',y='Net Sales',hue='Type of Customer')
plt.title('Relationship between Age and Net Sales by Customer type')
plt.xlabel('Age')
plt.ylabel('Net Sales')
plt.legend(title='Type of Customer')
plt.show()

'''Fifth question'''

sales_by_method = df.groupby('Method of Payment')['Net Sales'].sum()

plt.figure(figsize=(6,6))
plt.pie(sales_by_method.values,labels=sales_by_method.index,autopct='%1.1f%%',startangle=90)
plt.title('Sales by Method of Payment')
plt.axis('equal')
plt.show()