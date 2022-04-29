# Data Analysis Project - Iris Data set

The Project assignment involves researching the Fishers Iris dataset. 
There are a number of tasks to be carried out during the research and analysis of the dataset. 

I have included an analysis.py python script and a Juypter notebook that steps through the code and goes through the analysis of the dataset.

Looking at the readme file that comes with the dataset we need to carry out the following tasks :

* Read in the Iris Data set
* Check the Iris Data set
* Clean the Iris data
* What questions can we ask of the Iris data set
* Univariate Analysis
  * Histograms for each numeric variable
  * Univariate per Class
  * Analyzing the Summary statistics for the 3 Species
  * Histograms per Species/ Class
  * Boxplots
* Bivariate Analysis
  * Scatter Plots
  * Pearson Correlation Coeficient 
  * What further analysis would I like to carry out
* Summary of Analysis

These tasks are outlined in the Juypter notebook.

## How to run the python Code
I hava included an analysis.py which when run will run the code and carry out the following :

* Outputs a summary of each variable to a single text file called analysis.txt
* Saves a histogram of each variable to png files
* Outputs a scatter plot of each pair of variables
* Prints out checks on the Iris data set and also Correlation Coefficients

I have also included a Juypter notebook which steps through the code and explains the analysis carried out

## Investigations into the data set 
I have oulined the investigations I have carried out in the Juypter notebook.
This included both univariate and bivariate analysis.

## Summary of dataset
We can see from carrying out both Univariate and Bivariate analysis that we can distinguish between each of the three Iris plant species. 

Univariate analysis shows that Petal length and Petal Width are good variables to distinguish between Setosa and the other two Iris plant species. By using univariate analysis the variables don't seem to be good and distinguishing between Virginica and versicolor species.
With Bivariate analysis and analyzing the scatter plots or using the correlation coefficient method we can see there are positive high coefficients where variables have strong relationships. For Setosa there is a strong relation between sepal width and sepal length with a correlation coefficient of 0.74. The highest correlation coefficient for versicolor is 0.78 which shows there is a strong relationship between petal length and petal width. Lastly for virginica there is a correlation coefficient of 0.86 which is very high and is for variables sepal length and petal length. 

## Bibliography
Stackoverflow. (2011). How do you append to a file ? Available at: https://stackoverflow.com/questions/4706499/how-do-you-append-to-a-file (Accessed: 28 April 2022). 

Stackoverflow. (2011). Writing string to a new line every time. Available at: https://stackoverflow.com/questions/2918362/writing-string-to-a-file-on-a-new-line-every-time
(Accessed: 02 April 2022).

Real Python. (2020). The Pandas DataFrame: Make Working With Data Delightful. Available at: https://realpython.com/pandas-dataframe/
(Accessed: 23 March 2022).

w3schools. (n.d). Pandas Tutorial. Available at: https://www.w3schools.com/python/pandas/default.asp
(Accessed: 25 March 2022).

Chartio. (n.d). How to save a plot to a file using Matplotlib. Available at: https://chartio.com/resources/tutorials/how-to-save-a-plot-to-a-file-using-matplotlib/
(Accessed: 27 April 2022).

geeksforgeeks. (2021). Python – Basics of Pandas using Iris Dataset. Available at: https://www.geeksforgeeks.org/python-basics-of-pandas-using-iris-dataset/
(Accessed: 03 May 2022)