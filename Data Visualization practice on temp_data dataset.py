import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("temp_data.csv")
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])

df['AvgTempC'] = (df['Average_Fahrenheit_Temperature'] - 32) * 5/9

max_temp = df['AvgTempC'].idxmax()
min_temp = df['AvgTempC'].idxmin()
print(max_temp)
print(min_temp)

print(df.head())
print(df.dtypes)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(df['Year'],
         df['Average_Fahrenheit_Temperature'],
         marker='o')

plt.xlabel('Year')
plt.ylabel('Temperature (°F)')
plt.title('Temperature Trend')
plt.grid(True)

plt.show()
colors = ['gray'] * len(df)
colors[max_temp] = 'red'
colors[min_temp] = 'blue'

plt.figure(figsize=(12, 6))
bars = plt.bar(df['Year'], df['AvgTempC'], color=colors)

# Add temperature labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height,
             f'{height:.1f}',
             ha='center',
             va='bottom')

plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.title('Average Temperature by Year')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
