#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 14:32:54 2024

@author: montassarimen
"""

import numpy as np 
import matplotlib.pyplot as plt 
from collections import deque 
from sklearn.datasets import make_moons

def euclidean_distance(point1, point2):     
    return np.sqrt(np.sum((point1 - point2) ** 2)) 

def get_neighbors(point, data, epsilon):     
    neighbors = []     
    for i in range(len(data)):         
        if euclidean_distance(point, data[i]) < epsilon:             
            neighbors.append(i)     
    return neighbors 

def dbscan(data, epsilon, min_points):     
    labels = [-1] * len(data)  # Initialize labels as -1 (unclassified)     
    cluster_id = 0  
        
    for i in range(len(data)):         
        if labels[i] != -1:             
            continue 
                
        neighbors = get_neighbors(data[i], data, epsilon)  
                
        if len(neighbors) < min_points:             
            labels[i] = -1  # Mark as noise         
        else:             
            cluster_id += 1             
            labels = expand_cluster(data, labels, i, neighbors, cluster_id, epsilon, min_points)          
    return labels 

def expand_cluster(data, labels, point_index, neighbors, cluster_id, epsilon, min_points):     
    labels[point_index] = cluster_id     
    queue = deque(neighbors)          
    
    while queue:         
        neighbor_index = queue.popleft()                  
        
        if labels[neighbor_index] == -1:             
            labels[neighbor_index] = cluster_id                  
        
        if labels[neighbor_index] != -1:             
            continue                  
        
        labels[neighbor_index] = cluster_id         
        new_neighbors = get_neighbors(data[neighbor_index], data, epsilon)                  
        
        if len(new_neighbors) >= min_points:             
            queue.extend(new_neighbors)          
    
    return labels 

def plot_clusters(data, labels):     
    unique_labels = set(labels)     
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]          
    
    for k, col in zip(unique_labels, colors):         
        if k == -1:             
            col = [0, 0, 0, 1]  # Black used for noise.                  
        
        class_member_mask = (np.array(labels) == k)                  
        xy = data[class_member_mask]        
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)          
    plt.title('DBSCAN Clustering')     
    plt.show() 
    
data, _ = make_moons(n_samples=500, noise=0.1)  
#plt.scatter(data[:, 0], data[:, 1])
#plt.title('Original Data')
#plt.show()

# Run DBSCAN 
epsilon = 0.2  # Adjusted epsilon value
min_points = 3
labels = dbscan(data, epsilon, min_points)  

#print("Labels:", labels)  # Debugging statement to check labels

# Plot the results 
plot_clusters(data, labels)
