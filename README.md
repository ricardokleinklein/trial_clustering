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

1. Loading and cleansing of the data
2. General features/properties of data
3. Preprocessing of the data - Dimensionality reduction
4. Clustering
5. Analysis of results

### 1. Load and clean data

Packages used to load data and convert them into an easy dealt-with array are Pandas and Numpy. 
Simple process, it checks whether the dataset exists, if it contains any missing values and whether or not there are duplicate samples (two rows with the exact same attribute values).

The outcome of this step is an array of $100000 \times 15$ elements. 

- No missing values
- Not duplicated samples found

For these checkings I have relied on Pandas' implemented routines as trustworthy.

### 2. Preprocessing

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

### 3. Feature extraction

The next logical step is to try to find a feature representation that allows us to gain a deeper understanding of the underlying information hidden in the data.

To that aim, we perform two different algorithms over data: PCA analysis and ICA analysis, both in 2D and 3D.
- *PCA* allow us to identify those axis along which data retain a greater variance of the original multi-dimensional data. *ICA* performs a somehow opposite analysis. This way we can check whether there's any attribute/projection axis by which data are linearly separable.
![PCA in 2d and 3d](/images/pca.png)

- A shape that resembles that of a heart appears in both the 2 and 3 main components. That indicates a continuity in the data, so most likely no independent clustering will be possible, at least in visualizable spaces.

- *ICA* yields a dimensionality space in which particularly dominant attributes would dominate. It is applied here with some caution, cause it's not an algorithm suited for gaussian distributed data, but the attributes, with some exceptions, does not appear to be highly correlated, so in order to get some insight I run it nonetheless.

![ICA in 2d and 3d](/images/ica.png)

- A hearted-shape distribution is observed. It is a bit different from the one found with PCA, though, and poses some interesting features: it can be seen a couple of tails, a bulk, and some minor bulks. 
- Therefore, we shall use the features in 3D extracted from the ICA analysis for clustering.

### 4. Clustering

The final task is to cluster the data available. Because we have no defined label, we perform unsupervised clustering, leaving to the algorithms the task of finding the right grouping centers and criteria.

Because of the surveying aim of this project, I am running two simple but powerful unsupervised clustering algorithms: K-Means and a Gaussian Mixture Model.

- K-Means: Based on euclidean distance between points of the dataset, it is expected that it'll distribute the data into clusters depending on the position of the points in the 3D space obtained after the ICA analysis.
- This clustering doesn't seem to provide any relevant clustering, though it identifies some regions like the main bulk and the tails in different clusters, with just 8 labels.
- GMM: based on probabilistic distributions, assuming the data follow a gaussian distribution, which I have assumed is accurate enough.
- The clustering is similar to that obtained with K-Means. However, GMM tend to assign the same label to a larger amount of points. This yields a distribution in which some labels group the most of the samples in the dataset, leaving some other groups with almost no elements, or at least far fewer elements.
![different clusterings](/images/clusters.png)








