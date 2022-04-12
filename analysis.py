
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns


colnames=['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
index=('Iris Setosa', 'Iris Versicolour', 'Iris Virginica')
df = pd.read_csv("iris.data",names=colnames)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(df)
#df.describe()
meanValues = df.groupby('class').mean()
print('=================MEAN VALUES=================')
print(meanValues)

maxValues = df.groupby('class').max()
print('=================MAX VALUES=================')
print(maxValues)

#medianValues = df.groupby('class').median()
#print('=================MEDIAN VALUES=================')
#print(medianValues)

#modeValues = df.groupby('class').mode()
#print('=================MODE VALUES=================')
#print(modeValues)

stdValues = df.groupby('class').std()
print('=================STANDARD DEVIATION VALUES=================')
print(stdValues)


#df =df.pivot(index='name',columns='subject,values='grade')
#df.plot.bar()
#plt.show()
#print(df.describe())

#df.mean(axis=0)
#df.mean(axis='colnames')


#df2 = df["class"]
#print(df2)

print("========================== Describe =================================")
print(df.groupby("class").describe())


print('============================Correlation==============================')
print(df.corr())

print('#################################################################################')
print(df.groupby("class").corr())

#seaborn.set(style="ticks")
#seaborn.pairplot(df, hue="Class")
#plt.suptitle('iris plot')
#plt.show()
#sns.set(style="ticks", color_codes=True)
#df.drop('Id',axis=1,inplace=True)

#sns.pairplot(df, hue="Species", palette="husl", markers=["o", "s", "D"])

df = sns.load_dataset('iris')
df.head()

sns.boxplot( y=df["sepal_length"] );
#plt.show()

sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")

g = sns.pairplot(iris,hue="species", palette="husl", markers=["o", "s", "D"])

plt.show()





