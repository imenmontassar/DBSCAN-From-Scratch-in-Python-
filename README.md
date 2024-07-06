# DBSCAN-From-Scratch-in-Python-

In this repository, we will implement the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm from Scratch using Python. 
DBSCAN is a powerful clustering algorithm used in machine learning and data mining. Unlike other clustering algorithms like K-means, DBSCAN does not require the number of clusters to be specified beforehand and can discover clusters of arbitrary shape.

## Fundamental Concepts of DBSCAN:

DBSCAN relies on two key parameters:  
  1. Epsilon (ε): The maximum distance between two samples for them to be considered as part of the same neighborhood.
  2. MinPts: The minimum number of samples in a neighborhood to define a cluster.
Based on these parameters, DBSCAN classifies points into three categories:
  1• Core Point: A point with at least MinPts neighbors within ε.
  2• Border Point: A point that is not a core point but is in the neighborhood of a core point.
  3• Noise Point: A point that is neither a core point nor a border point.

