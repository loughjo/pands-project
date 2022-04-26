
from msilib.schema import Class
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns




# The code below is for mapping column names to each column in the data and also read in the iris.data csv file and assign it to dataframe df

colnames=['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
index=('Iris Setosa', 'Iris Versicolour', 'Iris Virginica')
df = pd.read_csv("iris.data",names=colnames)


print(df.columns)


# We check the first 10 rows of data using the head() function to see how the data is displayed and have a brief look at the dataset. 

iris_describe = df.head(10)
print(iris_describe)


# Next using the shape method we can check the dimensions of the df array. 
print(df.shape)
print(df["class"].value_counts())



# We use the dropna() method to remove any rows with missing values.
df.dropna(inplace=True)


# We can check the data gain to see if there were in deed any rows with missing values.
print(df.shape)
print(df["class"].value_counts())


df.drop_duplicates(inplace=True)


# We check the data again and we can see that there were 3 duplicates in total. One duplicate within the Iris-virginica class and two duplicates in the Iris-setosa class.


print(df.shape)
print(df["class"].value_counts())


# ## Univariate Analysis

# The first piece of real analysis is to carry out analysis on each variable in the dataset. 
# This is what is known as univariate analysis where We will get the summary data using the describe() function. 
#file1 = open("analysis.txt","a")

summary_iris_all = df.describe()
print(summary_iris_all)


#numpy_array = a_dataframe.to_numpy()
#np.savetxt("analysis.txt", summary_iris_all, fmt = "%d")
np.savetxt("analysis.txt", summary_iris_all, delimiter=' ', newline='n', header='', footer='', comments='# ', encoding=None)


# Histogram


sepal_length = df["sepal length in cm"]
  
plt.hist(sepal_length, bins = 20, color = "green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")
plt.show()




sepal_width = df["sepal width in cm"]
  
plt.hist(sepal_width, bins = 20, color = "blue")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")
plt.show()


# In[67]:


petal_length = df["petal length in cm"]
  
plt.hist(petal_length, bins = 20, color = "red")
plt.title("petal length in cm")
plt.xlabel("Petal_Length_cm")
plt.ylabel("Count")
plt.show()


# In[68]:


petal_width = df["petal width in cm"]
  
plt.hist(petal_width, bins = 20, color = "magenta")
plt.title("Petal Width in cm")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")
plt.show()


# In[69]:


sns.pairplot(df,hue="class", palette="husl", markers=["o", "s", "D"])
plt.show()


# In[70]:


iris_class = df.groupby('class')
setosa = iris_class.get_group('Iris-setosa')
print(setosa.describe())


# In[16]:


sns.pairplot(df,palette="husl", markers=["o", "s", "D"])
plt.show()


# In[71]:


versicolor = iris_class.get_group('Iris-versicolor')
print(versicolor.describe())


# In[72]:


virginica = iris_class.get_group('Iris-virginica')
print(virginica.describe())


# In[73]:


setosa_petal_width = setosa["petal width in cm"]
  
plt.hist(setosa_petal_width, bins = 20, color = "magenta")
plt.title("Petal Width in cm")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")
plt.show()


# In[74]:


setosa_petal_length = setosa["petal length in cm"]
  
plt.hist(setosa_petal_length, bins = 20, color = "yellow")
plt.title("Petal length in cm")
plt.xlabel("Petal_length_cm")
plt.ylabel("Count")
plt.show()


# In[75]:


setosa_sepal_length = setosa["sepal length in cm"]
  
plt.hist(setosa_sepal_length, bins = 20, color = "blue")
plt.title("Sepal length in cm")
plt.xlabel("Sepal_length_cm")
plt.ylabel("Count")
plt.show()


# In[76]:


setosa_sepal_width = setosa["sepal width in cm"]
  
plt.hist(setosa_sepal_width, bins = 20, color = "pink")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")
plt.show()


# Versicolor

# In[77]:


versicolor_petal_width = versicolor["petal width in cm"]
  
plt.hist(versicolor_petal_width, bins = 20, color = "magenta")
plt.title("Petal Width in cm")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")
plt.show()


# In[78]:


versicolor_petal_length = versicolor["petal length in cm"]
  
plt.hist(versicolor_petal_length, bins = 20, color = "yellow")
plt.title("Petal length in cm")
plt.xlabel("Petal_length_cm")
plt.ylabel("Count")
plt.show()


# In[79]:


versicolor_sepal_length = versicolor["sepal length in cm"]
  
plt.hist(versicolor_sepal_length, bins = 20, color = "blue")
plt.title("Sepal length in cm")
plt.xlabel("Sepal_length_cm")
plt.ylabel("Count")
plt.show()


# In[80]:


versicolor_sepal_width = versicolor["sepal width in cm"]
  
plt.hist(versicolor_sepal_width, bins = 20, color = "pink")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")
plt.show()


# In[81]:


virginica_petal_width = virginica["petal width in cm"]
  
plt.hist(virginica_petal_width, bins = 20, color = "magenta")
plt.title("Petal Width in cm")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")
plt.show()


# In[82]:


virginica_petal_length = virginica["petal length in cm"]
  
plt.hist(virginica_petal_length, bins = 20, color = "yellow")
plt.title("Petal length in cm")
plt.xlabel("Petal_length_cm")
plt.ylabel("Count")
plt.show()


# In[83]:


virginica_sepal_length = virginica["sepal length in cm"]
  
plt.hist(virginica_sepal_length, bins = 20, color = "blue")
plt.title("Sepal length in cm")
plt.xlabel("Sepal_length_cm")
plt.ylabel("Count")
plt.show()


# In[84]:


virginica_sepal_width = virginica["sepal width in cm"]
  
plt.hist(virginica_sepal_width, bins = 20, color = "pink")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")
plt.show()


# In[85]:


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

plt.show()


# In[86]:


fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(setosa_sepal_length, color = "red")
axs[0].set_xlabel("Setosa Sepal Length")
axs[1].hist(virginica_sepal_length, color = "magenta")
axs[1].set_xlabel("Virginica Sepal Length")
axs[2].hist(versicolor_sepal_length, color = "yellow")
axs[2].set_xlabel("Versicolor Sepal Length")

plt.show()


# In[87]:


fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(setosa_petal_width, color = "red")
axs[0].set_xlabel("Setosa Petal Width")
axs[1].hist(virginica_petal_width, color = "magenta")
axs[1].set_xlabel("Virginica Petal Width")
axs[2].hist(versicolor_petal_width, color = "yellow")
axs[2].set_xlabel("Versicolor Petal Width")

plt.show()


# In[88]:


fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(setosa_petal_length, color = "red")
axs[0].set_xlabel("Setosa petal Length")
axs[1].hist(virginica_petal_length, color = "magenta")
axs[1].set_xlabel("Virginica petal Length")
axs[2].hist(versicolor_petal_length, color = "yellow")
axs[2].set_xlabel("Versicolor Petal Length")

plt.show()


# Boxplot for Petal Lenghth - Setosa, Versicolor and Virginica. You can clearly see from the 3 boxplots that the median for Setosa Petal length is 1.5cms compared to versicolor petal length which is just below 4.5cms and virginica petal length which is approimately 5.75cm.  

# In[89]:


box_plot_data=[setosa_petal_length,versicolor_petal_length,virginica_petal_length]
#plt.boxplot(box_plot_data)

box=plt.boxplot(box_plot_data,vert=0,patch_artist=True,labels=['Setosa petal length','versicolor petal length','virginica petal length'],
            )
 
colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.show()


# Boxplots for petal Width - Setosa, Versicolor and Virginica. Again Setosa 

# In[96]:


box_plot_data=[setosa_petal_width,versicolor_petal_width,virginica_petal_width]
box=plt.boxplot(box_plot_data,vert=0,patch_artist=True,labels=['Setosa petal width','versicolor petal width','virginica petal width'],
            )
 
colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.show()


# Boxplot for Sepal Lenghth for Setosa, Versicolor and Virginica

# In[95]:


box_plot_data=[setosa_sepal_length,versicolor_sepal_length,virginica_sepal_length]
box=plt.boxplot(box_plot_data,vert=0,patch_artist=True,labels=['Setosa sepal length','versicolor sepal length','virginica sepal length'],
            )
 
colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.show()


# Boxplot for Sepal Width for the 3 Classes

# In[94]:


box_plot_data=[setosa_sepal_width,versicolor_sepal_width,virginica_sepal_width]
box=plt.boxplot(box_plot_data,vert=0,patch_artist=True,labels=['Setosa sepal width','versicolor sepal width','virginica sepal width'],
            )
 
colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.show()


# In[ ]:




