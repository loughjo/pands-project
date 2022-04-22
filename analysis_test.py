

from msilib.schema import Class
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
iris_describe = df.groupby('class').describe()
print(iris_describe)
#iris_all = df.describe('class')
#print(iris_all)

print("#####################")
print(df.shape)

print(df.columns)

print(df["class"].value_counts())


iris = df.groupby("class").describe()
print(iris)
#print(df.get_group["Iris-setosa"])

##### Group the data to find out summary stats #######
iris_class = df.groupby('class')
setosa = iris_class.get_group('Iris-setosa')
print(setosa.describe())

versicolor = iris_class.get_group('Iris-versicolor')
print(versicolor.describe())

virginica = iris_class.get_group('Iris-virginica')
print(virginica.describe())

print("### all the data ######")
print(df.describe())


print (df.head(10))
x = df["sepal length in cm"]
  
plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")
plt.show()

y = df["sepal width in cm"]
  
plt.hist(x, bins = 20, color = "blue")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")
plt.show()

x = df["petal length in cm"]
  
plt.hist(x, bins = 20, color = "red")
plt.title("petal length in cm")
plt.xlabel("Petal_Length_cm")
plt.ylabel("Count")
plt.show()

x = df["petal width in cm"]
  
plt.hist(x, bins = 20, color = "magenta")
plt.title("Petal Width in cm")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")
plt.show()

sns.pairplot(df,hue="class", palette="husl", markers=["o", "s", "D"])
plt.show()
