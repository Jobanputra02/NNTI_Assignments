#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 23:11:46 2024

@author: shaique, nikolai , chaitanya


Name: Shaique Solanki
Student ID (matriculation number): 7062750, 
Email: moso00002@stud.uni-saarland.de

Name: Nikolai Egorov
Student ID (matriculation number): 7062750, 
Email: nieg00001@stud.uni-saarland.de

Name: Chaitanya Jobanputra
Student ID (matriculation number): 7062300,
Email: chjo00006@uni-saarland.de


"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

# Exercise 1

def linearly_separable_data_creator():
    np.random.seed(42)
    num_clusters = 2
    num_points = 100
    dimension = 2
    mean_blob_1 = [-2, -2]
    mean_blob_2 = [2, 2]
    blob_1 = np.random.rand(num_points, dimension) + mean_blob_1
    blob_2 = np.random.rand(num_points, dimension) + mean_blob_2
    # label_blob_1 = np.full((num_points, 1), 0)
    # label_blob_2 = np.ones((num_points, 1))
    label_blob_1 = np.ones((num_points, 1))
    label_blob_2 = np.full((num_points, 1), -1)
    # plt.scatter(blob_1 , num_points)
    # plt.scatter(blob_2 , num_points)
    # plt.show()
    data_points = np.vstack([blob_1, blob_2])
    labels = np.vstack([label_blob_1, label_blob_2])
    shuffled_indices = np.random.permutation(num_points * 2)
    data_points_shuffled = data_points[shuffled_indices]
    labels_shuffled = labels[shuffled_indices]
    return (data_points_shuffled, labels_shuffled)

def xor_dataset():
    # 4 datapoints
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    # Extract x and y value from each data point
    points_x = X[:,0]
    points_y = X[:,1]
    # Classify datapoints using XOR operation
    labels = np.logical_xor(points_x, points_y)
    # When True, yield 1, otherwise -1
    labels = np.where(labels, 1, -1)
    return X, labels

def plot_dataset(X,y):
    # Plot data points
    # X[:, 0]: blob 1
    # X[:, 1]: blob 2
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr")
    plt.show()

# Exercise 2

def plot_decision_boundary(X, y, model):
    # Calculate weight vector and intercept (bias)
    w = model.coef_[0]
    b = model.intercept_

    # Plot data points
    # X[:, 0]: blob 1
    # X[:, 1]: blob 2
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr")

    # Decision boundary of the form: y(x) = <w,x> + b = 0,
    # since for each data point we have two features, our decision boundary is defined as:
    # x1*w1 + x2*w2 + b = 0    # => x2 = -x1*w1/w2 - b/w2
    # Generate values for feature 1
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
    # Compute corresponding feature 2 values
    x2 = - (w[0] / w[1]) * x1 - (b / w[1])

    # Plot the decision boundary
    plt.plot(x1, x2, color='black')
    plt.title('Linearly separable training data')
    plt.show()

# Exercise 3:
# 1. Difference between datasets is in volume of available information.
# First dataset has enough information (datapoints) for pattern recognition
# and is able to train efficiently with the data.
# Second dataset however has a limited amount of datapoints (only 4).
# This makes it impossible to recognise any pattern in the data or make any prediction.
# => optimization is not possible and therefore, derivation of decision boundary.

# 2. Since we are dealing with SVM, our model tries to maximize the margin between
# the decision line and the nearest datapoint.
# In this case we may assume, that our decision boundary is unique.

# 3. Decision boundary got shifted due to the outliers.

def create_outliers(dataset, labels, shape_of_the_dataset):
    num_of_points = np.arange(shape_of_the_dataset[0])
    np.random.shuffle(num_of_points)
    idx = num_of_points[:8]
    classes = labels[idx]
    flipped_classes = np.where(classes == -1, 1, -1)
    labels[idx] = flipped_classes
    return dataset, labels

if __name__=="__main__":

    X, y = linearly_separable_data_creator()
    X_xor, y_xor = xor_dataset()

    plot_dataset(X, y)
    plot_dataset(X_xor, y_xor)

    clf = LinearSVC()
    clf.fit(X, y.ravel())
    plot_decision_boundary(X, y, clf)

    # clf_xor = LinearSVC()
    # plot_decision_boundary(X_xor, y_xor, clf_xor)

    X_outliers, y_outliers = create_outliers(X,y, X.shape)
    clf.fit(X, y.ravel())
    plot_decision_boundary(X_outliers, y_outliers, clf)