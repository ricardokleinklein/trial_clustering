# trial_clustering repository

This repository is but a proof-of-concept for some clustering ideas with enphasis in typical data analytics purposes.

> The dataset used consists of 100000 biological events which are characterized by 15 numerical parameters respectively. Hence if any two events have the exact same number for all the parameters, you can define those two events are the same. And we know that this dataset includes a lot of groups or clusters are hidden in this whole data. Each group/cluster have a wide variety of the number of events. ( ex. One group/cluster can have 10000 events. Or another group/cluster might have 10 events.) Another condition is that these group/cluster should not be completely independent. In other words, they are a sort of continuous. We expect you can find a sort of map where you can see as many groups/clusters as you can and those groups/clusters are connected each other just like train map or something.
> Therefore, we would like you to try to find out how many groups/clusters you can find or how much those clusters are similar each other or what kind of correlation you can find in this dataset.

## Requirements

- python 2.7
- numpy >= 1.14
- matplotlib >= 2.2.0
- docopt >= 0.6.0
- tqdm >= 4.19.0
- pytorch >= 0.3.1
- pandas >= 0.22.0
- scikit-learn >= 0.19.1

## Steps followed

There are five main phases followed for this project. Rather than actually meaningful results, which would be hard to get due to the limited domain expertise available, the pipeline will be more process-oriented, i.e., the steps will be those that would normally try at a first attempt to any data mining challenge. Thus, those phases are: 
* Loading and cleansing of the data
* General features/properties of data
* Preprocessing of the data - Dimensionality reduction
* Clustering
* Analysis of results

### Load and clean data

Packages used to load data and convert them into an easy dealt-with array are Pandas and Numpy. 
Simple process, it checks whether the dataset exists, if it contains any missing values and whether or not there are duplicate samples (two rows with the exact same attribute values).

The outcome of this step is an array of $100000 \times 15$ elements. 

- No missing values
- Not duplicated samples found

For these checkings I have relied on Pandas' implemented routines as trustworthy.

### Preprocessing

In order to better understand the data, and how the different attributes can be related between themselves, I run a series of experiments that try to check what's the distribution of each attribute.
- Presence of outliers: No outliers found in any sample. The method used is the interquartile method, considering the 25% and 75% of the data. Thus, all the sample shall be considered.
- Given that the attributes have no outliers, but their relative sizes are quite different in orders of magnitud, I proceed to normalize all attributes' values to have mean 0 and standard deviation 1 so any learning algorithm will not be skewed because of this.
- I proceed to obtain a correlation table for every pair of attributes. The index used to establish the correlation between attributes is the *Pearson coefficient*.
![correlation table of attributes](/images/correlation_table.png)
- The table shows some very strongly correlated pairs of attributes: *Par1-2-3, Par3-4, Par3-5, Par4-5.*
- It also shows that there are a couple of attributes apparently unrelated to any other: *Parameter10* and *Parameter11*.
- **It would be interesting to explore these relationships in further detail**, but due to limitations in time we leave it as future work.
- I plot the data distribution for every attribute, prior to normalization. This step allows to see possible skewed data distributions. Particular features in these distributions will give us clues on how to explore the dataset.
![distribution table of attributes](/images/distribution_table.png)
- All the parameters present something not far from Gaussian distributions, although some of them are extremely skewed. There's an interesting peak on *Parameter9*.
- Looking at the correlation table, we can say that because most of the values in *Parameter10* and *Parameter14* are close to each other, and these attributes are not strongly correlated to any attribute, I shall not take them into consideration in further processing, because at first sight it doesn't look like they'll provide much information to any clustering algorithm.






