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