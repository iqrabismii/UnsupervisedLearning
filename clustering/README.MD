 # :credit_card: :receipt: Credit Card Customer Segmentation

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org) 
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
[![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org)
[![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) 
[![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) 
[![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Introduction 
Credit card __customer segmentation__ is an important aspect of managing a credit card business. 
It allows financial institutions to divide their customer base into groups based on specific 
characteristics, such as spending habits, income levels, and demographic information. This information 
can then be used to develop targeted marketing campaigns, create custom products and services, and improve 
risk management. With better understanding of their customers, financial institutions can make more informed decisions, 
leading to improved customer satisfaction, higher conversion rates, and increased revenue. Customer segmentation is crucial 
for the success of any credit card business,
as it helps to personalize the customer experience, build customer loyalty, and stay ahead of the competition.

## Challenges
Despite its importance, credit card customer segmentation faces several challenges. 
Firstly, accurately identifying customer segments can be difficult as customers' needs 
and behaviors can change over time. Financial institutions need to ensure that their segmentation 
techniques are constantly updated to reflect these changes. Secondly, obtaining complete and accurate 
customer data can also be a challenge, as customers may be hesitant to provide sensitive information or 
may not accurately report their spending habits. Thirdly, developing and implementing effective marketing strategies 
that are tailored to each customer segment can be time-consuming and resource-intensive. Finally, maintaining customer 
privacy and security is also a significant challenge, especially in light of increasing concerns over data privacy and security.
Financial institutions need to implement robust security measures to ensure that customer data is protected and that marketing 
efforts do not infringe on privacy rights. Despite these challenges, credit card customer segmentation remains a valuable tool for financial institutions, 
as it provides valuable insights into customer behavior and preferences and helps to drive business growth and success.

## Data Science and Unsupervised Machine Learning
__Unsupervised machine learning__  can help in credit card customer segmentation by automating the process of dividing customers into groups based on their behaviors and characteristics. Unsupervised ML algorithms, such as __clustering and dimensionality reduction__, can analyze large amounts of customer data and identify patterns and relationships that might not be immediately apparent to human analysts. This can help financial institutions to segment customers more accurately and quickly, and to gain deeper insights into customer behavior and preferences.

With unsupervised ML, financial institutions can identify customer segments that might not have been previously recognized. This can lead to the development of new products and services that are better suited to the needs and wants of specific customer segments. Additionally, unsupervised ML can help to identify customer segments that are at high risk of default or other forms of financial difficulty, allowing financial institutions to take proactive measures to mitigate these risks.

Unsupervised ML can also help to reduce the time and resources required for credit card customer segmentation, as it eliminates the need for manual analysis and interpretation of customer data. This allows financial institutions to focus on other areas of the business, such as marketing and product development.

In conclusion, unsupervised ML can play a significant role in improving credit card customer segmentation, by providing accurate, timely, and actionable insights into customer behavior and preferences. This can help financial institutions to better understand their customers, develop targeted marketing strategies, and improve risk management, leading to increased revenue and business success.

## EDA
The dataset was taken from Kaggle. Link to the dataset is provided [here.](https://github.com/iqrabismii/UnsupervisedLearning/blob/main/clustering/creditcard.csv)
__Pandas Profiler__ to analyse data first. This helps to analysis at a wider angle. A few missing values were identified. There was also high collinearity in the features. For this,PCA was implemented. Also the data was scaled and log tranformed so that data is less skewed and all values are in same maginitude. 
More details about profiler and data cleaning is available [here.](https://github.com/iqrabismii/UnsupervisedLearning/blob/main/clustering/CreditCardCustomersClustering.ipynb)


## Modelling

First, PCA was implemented to convert highly correlated variables into linear uncorrelated variables. Screeplot was used to see the variance explained by components.
As the reduced data should explain 80% of the variance, 4 components was used. Then, different unsupervised methods were used to segment customers. 
* __Hierarchical clustering__ is a type of unsupervised machine learning technique used to group similar objects into clusters. There are two main types of hierarchical clustering:
1. Agglomerative Hierarchical Clustering: This type of clustering starts with each object being treated as its own cluster, and then iteratively merges the closest clusters into larger groups until all objects belong to a single cluster or a specified number of clusters have been formed.
2. Divisive Hierarchical Clustering: This type of clustering starts with all objects in a single cluster, and then iteratively splits the cluster into smaller groups until each object belongs to its own cluster.

Both types of hierarchical clustering can be visualized as a dendrogram, which shows the relationships between the clusters and the objects within them.

* __Model Based Clustering__: It is a type of unsupervised machine learning technique that uses statistical models to group similar objects into clusters. Gaussian Mixture Model (GMM) is one such model-based clustering algorithm.

In GMM, each cluster is represented by a Gaussian distribution, and the algorithm fits a Gaussian mixture model to the data by estimating the mean, covariance, and weight of each cluster. The algorithm uses an Expectation-Maximization (EM) algorithm to find the optimal parameters that maximize the likelihood of the data given the Gaussian mixture model.

One advantage of GMM is that it can model clusters with different shapes and sizes, which makes it well-suited for data that has complex structures. Additionally, GMM can also model clusters that are overlapping or have different densities.

* __DBSCAN__ (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are close to each other in the feature space and separates points that are far away from each other. It does this by defining a neighborhood around each data point, and grouping together points that are close enough to each other to be considered a cluster.

The two key parameters in DBSCAN are the radius (eps) and the minimum number of points (minPts) required to form a cluster. The radius defines the neighborhood around each data point, and the minimum number of points determines the minimum number of points that must be within the radius to form a cluster.

One advantage of DBSCAN is that it can identify clusters of arbitrary shapes and sizes, and can handle noisy data. Additionally, DBSCAN does not require the number of clusters to be specified in advance, unlike other clustering algorithms such as k-means.

DBSCAN is often used in a variety of applications, including market segmentation, customer profiling, and image segmentation. It is a flexible technique that can handle a large number of features and can be useful for uncovering hidden patterns in data.

In conclusion, DBSCAN is a density-based clustering algorithm that groups together points that are close to each other in the feature space and separates points that are far away from each other. With its ability to handle complex data structures and uncover hidden patterns, DBSCAN can provide valuable insights into the relationships between objects in the dataset.

* __KMeans Clustering__ : K-Means is a popular centroid-based clustering algorithm. It partitions a dataset into K clusters, where K is a user-defined number, by assigning each data point to the cluster with the nearest mean. The algorithm iteratively updates the cluster means and reassigns data points until convergence, producing K clusters that minimize the within-cluster sum of squared distances.


Customers were segmented in two clusters using Agglomerative method. Ward(min variance) was used for Linkage.
For DBSCAN, customers were segmented in four clusters. 
Details about each algorithm is available [here.](https://github.com/iqrabismii/UnsupervisedLearning/blob/main/clustering/CreditCardCustomersClustering.ipynb)


