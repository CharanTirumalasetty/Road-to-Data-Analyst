import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("train.csv")
print(df.head())

print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])

num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print(df.isnull().sum())

# Output 1
categorical_cols = ['GarageCond','GarageQual','KitchenQual','HeatingQC','BsmtQual','ExterCond','ExterQual']
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,8),sharey=True)
axes = axes.flatten()
for i, col in enumerate(categorical_cols):
    sns.barplot(x=col, y='SalePrice', data=df, ax=axes[i], hue=col,palette='tab10',legend=False, edgecolor='black')
    axes[i].set_xlabel(col)
    if i in [0,4]:
        axes[i].set_ylabel('SalePrice')
    else:
        axes[i].set_ylabel('')

axes[-1].set_visible(False)
plt.tight_layout()
plt.show()

#Output 2

categorical_cols2 =['Street','PavedDrive','CentralAir','SaleCondition','Electrical','Utilities','LotShape','LotConfig','LandSlope','MasVnrType','GarageFinish',
                    'BldgType','Foundation','Heating','BsmtExposure','LandContour','RoofStyle','MSZoning','BsmtCond']
fig, axes2 = plt.subplots(nrows=7, ncols=3, figsize=(16,22),sharey=False)
axes2 = axes2.flatten()
for i, col in enumerate(categorical_cols2):
    sns.barplot(
        x=col, y='SalePrice', data=df, ax=axes2[i], hue=col,palette='tab10',legend=False, edgecolor='black'
    )
axes2[i].set_xlabel(col, fontsize=10)
axes2[i].set_ylabel('SalePrice', fontsize=10)
axes2[i].tick_params(axis='x',labelrotation=15,labelsize=10)

for j in range(len(categorical_cols2), len(axes2)):
    axes2[j].set_visible(False)

plt.tight_layout()
plt.show()