#  Iris Data Analysis Project data Analysis code

# Import all the functions 

from msilib.schema import Class
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns



# The code below is for mapping column names to each column in the data and more importantly read in the iris.data csv 
# file and assign it to dataframe which we call df.

colnames=['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
index=('Iris Setosa', 'Iris Versicolour', 'Iris Virginica')
df = pd.read_csv("iris.data",names=colnames)


# We now check to see the column names and make sure the data is read in ok from the csv file to the dataframe.                   
print(df.columns)


# We check the first 10 rows of data using the head() function to see how the data is displayed and have a brief look at the dataset. 
iris_describe = df.head(10)
print(iris_describe)


# Next using the shape method we can check the dimensions of the df array. We can see the df array has 150 rows and 5 columns. 
print(df.shape)
print(df["class"].value_counts())


# ## Clean the Iris Data
# We use the dropna() method to remove any rows with missing values.
df.dropna(inplace=True)


# We can check the data gain to see if there were in deed any rows with missing values.
print(df.shape)
print(df["class"].value_counts())

# We now check for duplicate data using the drop_duplicates() method and set *inplace=True* so it removes in any duplicates 
# in the current dataframe - df.
df.drop_duplicates(inplace=True)


# We check the data again and we can see that there were 3 duplicates in total. 
print(df.shape)
print(df["class"].value_counts())

# we will get the summary data using the describe() function. 
summary_iris_all = df.describe()
print(summary_iris_all)


# Histograms for each numeric variable
# We assign the 'sepal length in cm' data to sepal_length. We create a histogram using plt.hist() method 


sepal_length = df["sepal length in cm"]
  
plt.hist(sepal_length, bins = 20, color = "green")
plt.title("Sepal Length in cm - All Species")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")
plt.savefig('Sepal_Length_All_Species_Histogram.png')
#plt.show()
plt.close()

# The histogram for sepal width which 


sepal_width = df["sepal width in cm"]
  
plt.hist(sepal_width, bins = 20, color = "blue")
plt.title("Sepal Width in cm - All Species")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")
plt.savefig('Sepal_Width_All_Species_Histogram.png')
#plt.show()
plt.close()

# The histogram for petal length is interesting, 

petal_length = df["petal length in cm"]
  
plt.hist(petal_length, bins = 20, color = "red")
plt.title("petal length in cm - All Species")
plt.xlabel("Petal_Length_cm")
plt.ylabel("Count")
plt.savefig('Petal_Length_All_Species_Histogram.png')
#plt.show()
plt.close()

# Below is a histogram of petal width 

petal_width = df["petal width in cm"]
  
plt.hist(petal_width, bins = 20, color = "magenta")
plt.title("Petal Width in cm - All Species")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")
plt.savefig('Sepal_Width_All_Species_Histogram.png')
#plt.show()
plt.close()

# We will now carry out Univariate analysis on each Iris plant Class to see what we can fine.
# First of all we need to split the data by class or species and we use the groupby() method to carry this out. 
# We now have the iris_class object.
iris_class = df.groupby('class')


# We use the get_group method on the iris_class object to pull out the data rows for the 'Iris-setosa' class or species.

setosa = iris_class.get_group('Iris-setosa')
print(setosa.describe())


# We do the same for the versicolor class.
versicolor = iris_class.get_group('Iris-versicolor')
print(versicolor.describe())


# We do the same for the verginica class.

virginica = iris_class.get_group('Iris-virginica')
print(virginica.describe())



iris_class = df.groupby('class')
setosa = iris_class.get_group('Iris-setosa')
print(setosa.describe())


versicolor = iris_class.get_group('Iris-versicolor')
print(versicolor.describe())



virginica = iris_class.get_group('Iris-virginica')
print(virginica.describe())



setosa_petal_width = setosa["petal width in cm"]
  
plt.hist(setosa_petal_width, bins = 20, color = "magenta")
plt.title("Petal Width in cm - Setosa Species")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")
plt.savefig('Setosa_Petal_Width_Histogram.png')
#plt.show()
plt.close()



setosa_petal_length = setosa["petal length in cm"]
  
plt.hist(setosa_petal_length, bins = 20, color = "yellow")
plt.title("Petal length in cm - Setosa Species")
plt.xlabel("Petal_length_cm")
plt.ylabel("Count")
plt.savefig('Setosa_Petal_Length_Histogram.png')
#plt.show()
plt.close()



setosa_sepal_length = setosa["sepal length in cm"]
  
plt.hist(setosa_sepal_length, bins = 20, color = "blue")
plt.title("Sepal length in cm - Setosa Species")
plt.xlabel("Sepal_length_cm")
plt.ylabel("Count")
plt.savefig('Setosa_Sepal_Length_Histogram.png')
#plt.show()
plt.close()



setosa_sepal_width = setosa["sepal width in cm"]
  
plt.hist(setosa_sepal_width, bins = 20, color = "pink")
plt.title("Sepal Width in cm - Setosa Species")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")
plt.savefig('Setosa_Sepal_Width_Histogram.png')
#plt.show()
plt.close()

# Versicolor


versicolor_petal_width = versicolor["petal width in cm"]
  
plt.hist(versicolor_petal_width, bins = 20, color = "magenta")
plt.title("Petal Width in cm - Versicolor Species")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")
plt.savefig('Versicolor_Petal_Width_Histogram.png')
#plt.show()
plt.close()



versicolor_petal_length = versicolor["petal length in cm"]
  
plt.hist(versicolor_petal_length, bins = 20, color = "yellow")
plt.title("Petal length in cm - Versicolor Species")
plt.xlabel("Petal_length_cm")
plt.ylabel("Count")
plt.savefig('Versicolor_Petal_Length_Histogram.png')
#plt.show()
plt.close()


versicolor_sepal_length = versicolor["sepal length in cm"]
  
plt.hist(versicolor_sepal_length, bins = 20, color = "blue")
plt.title("Sepal length in cm - Versicolor Species")
plt.xlabel("Sepal_length_cm")
plt.ylabel("Count")
plt.savefig('Versicolor_Sepal_Length_Histogram.png')
#plt.show()
plt.close()



versicolor_sepal_width = versicolor["sepal width in cm"]
  
plt.hist(versicolor_sepal_width, bins = 20, color = "pink")
plt.title("Sepal Width in cm - Versicolor Species")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")
plt.savefig('Versicolor_Sepal_Width_Histogram.png')
#plt.show()
plt.close()



virginica_petal_width = virginica["petal width in cm"]
  
plt.hist(virginica_petal_width, bins = 20, color = "magenta")
plt.title("Petal Width in cm - Virginica Species")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")
plt.savefig('Virginica_Petal_Width_Histogram.png')
#plt.show()
plt.close()



virginica_petal_length = virginica["petal length in cm"]
  
plt.hist(virginica_petal_length, bins = 20, color = "yellow")
plt.title("Petal length in cm - Virginica Species")
plt.xlabel("Petal_length_cm")
plt.ylabel("Count")
plt.savefig('Virginica_Petal_Length_Histogram.png')
#plt.show()
plt.close()



virginica_sepal_length = virginica["sepal length in cm"]
  
plt.hist(virginica_sepal_length, bins = 20, color = "blue")
plt.title("Sepal length in cm - Virginica Species")
plt.xlabel("Sepal_length_cm")
plt.ylabel("Count")
plt.savefig('Virginica_Sepal_Length_Histogram.png')
#plt.show()
plt.close()



virginica_sepal_width = virginica["sepal width in cm"]
  
plt.hist(virginica_sepal_width, bins = 20, color = "pink")
plt.title("Sepal Width in cm - Virginica Species")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")
plt.savefig('Virginica_Sepal_Width_Histogram.png')
#plt.show()
plt.close()



#plt.hist(virginica_sepal_width, bins = 20, color = "pink")
#plt.hist(setosa_sepal_width, bins = 20, color = "red")
#plt.hist(setosa_sepal_width, bins = 20, color = "blue")



fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(setosa_sepal_width, color = "red")
axs[0].set_xlabel("Setosa Sepal Width")
axs[1].hist(virginica_sepal_width, color = "magenta")
axs[1].set_xlabel("Virginica Sepal Width")
axs[2].hist(versicolor_sepal_width, color = "yellow")
axs[2].set_xlabel("Versicolor Sepal Width")
plt.savefig('All_Species_separate_Histograms_Sepal_Width.png')
#plt.show()
plt.close()



fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(setosa_sepal_length, color = "red")
axs[0].set_xlabel("Setosa Sepal Length")
axs[1].hist(virginica_sepal_length, color = "magenta")
axs[1].set_xlabel("Virginica Sepal Length")
axs[2].hist(versicolor_sepal_length, color = "yellow")
axs[2].set_xlabel("Versicolor Sepal Length")
plt.savefig('All_Species_separate_Histograms_Sepal_Length.png')
#plt.show()
plt.close()



fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(setosa_petal_width, color = "red")
axs[0].set_xlabel("Setosa Petal Width")
axs[1].hist(virginica_petal_width, color = "magenta")
axs[1].set_xlabel("Virginica Petal Width")
axs[2].hist(versicolor_petal_width, color = "yellow")
axs[2].set_xlabel("Versicolor Petal Width")
plt.savefig('All_Species_separate_Histograms_Petal_Width.png')
#plt.show()
plt.close()



fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(setosa_petal_length, color = "red")
axs[0].set_xlabel("Setosa petal Length")
axs[1].hist(virginica_petal_length, color = "magenta")
axs[1].set_xlabel("Virginica petal Length")
axs[2].hist(versicolor_petal_length, color = "yellow")
axs[2].set_xlabel("Versicolor Petal Length")
plt.savefig('All_Species_separate_Histograms_Petal_Length.png')
#plt.show()
plt.close()

# ## Boxplots

# Boxplot for Petal Lenghth - Setosa, Versicolor and Virginica. 



box_plot_data=[setosa_petal_length,versicolor_petal_length,virginica_petal_length]
#plt.boxplot(box_plot_data)

box=plt.boxplot(box_plot_data,vert=0,patch_artist=True,labels=['Setosa petal length','versicolor petal length','virginica petal length'],
            )
 
colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.savefig('All_Species_separate_Boxplots_Petal_Length.png')    
#plt.show()
plt.close()

# Boxplots for petal Width - Setosa, Versicolor and Virginica. 


box_plot_data=[setosa_petal_width,versicolor_petal_width,virginica_petal_width]
box=plt.boxplot(box_plot_data,vert=0,patch_artist=True,labels=['Setosa petal width','versicolor petal width','virginica petal width'],
            )
 
colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.savefig('All_Species_separate_Boxplots_Petal_Width.png')
#plt.show()
plt.close()

# Boxplot for Sepal Lenghth for Setosa, Versicolor and Virginica. 


box_plot_data=[setosa_sepal_length,versicolor_sepal_length,virginica_sepal_length]
box=plt.boxplot(box_plot_data,vert=0,patch_artist=True,labels=['Setosa sepal length','versicolor sepal length','virginica sepal length'],
            )
 
colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.savefig('All_Species_separate_Boxplots_Sepal_Length.png')    
#plt.show()
plt.close()

# Boxplot for Sepal Width for the 3 Classes



box_plot_data=[setosa_sepal_width,versicolor_sepal_width,virginica_sepal_width]
box=plt.boxplot(box_plot_data,vert=0,patch_artist=True,labels=['Setosa sepal width','versicolor sepal width','virginica sepal width'],
            )
 
colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.savefig('All_Species_separate_Boxplots_Sepal_Width.png')
#plt.show()
plt.close()

print(df.corr())




sns.pairplot(df,hue="class", palette="husl", markers=["o", "s", "D"])
plt.savefig('All_Species_scatter_plot.png')
#plt.show()
plt.close()

print(setosa.corr(method='pearson'))


# If we apply the correlation efficient to the setosa class by using the corr() method we can see that the strongest relationship



print(versicolor.corr(method='pearson'))




print(virginica.corr(method='pearson'))

