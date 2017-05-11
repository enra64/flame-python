# -*- coding: utf-8 -*-

import sys

import numpy
from scipy.spatial.distance import squareform, pdist


def extract_structure_information(
        data,
        k,
        outlier_threshold,
        distance_measure,
        minkowski_p= None,
        weighted_minkowsky_weights = None):
    """
    Extract structure information from the dataset

    :param data: a numpy-matrix. each column represents an attribute; each row a data item
    :param k: the amount of neighbours to use for the initial knn graph
    :param outlier_threshold: the maximum density an outlier can have
    :param distance_measure: a str describing the distance measure:  
        ‘braycurtis’, 
        ‘canberra’, 
        ‘chebyshev’, 
        ‘cityblock’, 
        ‘correlation’, 
        ‘cosine’, 
        ‘dice’, 
        ‘euclidean’, 
        ‘hamming’, 
        ‘jaccard’, 
        ‘kulsinski’, 
        ‘mahalanobis’, 
        ‘matching’, 
        ‘minkowski’, 
        ‘rogerstanimoto’, 
        ‘russellrao’, 
        ‘seuclidean’, 
        ‘sokalmichener’, 
        ‘sokalsneath’, 
        ‘sqeuclidean’, 
        ‘yule’
    :param minkowski_p: The p-norm to apply Only for Minkowski, weighted and unweighted. Mandatory
    :param weighted_minkowsky_weights: The weight vector. Only for weighted Minkowski. Mandatory
    :return: a tuple of lists: (cluster supporting objects, cluster outliers, rest), 
        where each list contains the indices of the objects in the data matrix 
        * Cluster Supporting Object (CSO): object with density higher than all its neighbors
        * Cluster Outliers: object with density lower than all its neighbors, and lower than a predefined threshold
        * Rest Object: object not assigned to one of the previous groups
    """
    # check that a p value exists if minkowski distance is used
    if distance_measure == 'minkowski' or distance_measure == 'wminkowski':
        assert minkowski_p is not None, "Minkowski distance requires a p value to be supplied!"
    # check that a weight vector exists if weighted minkowski is used
    elif distance_measure == 'wminkowski':
        assert weighted_minkowsky_weights is not None, "Weighted Minkowski distance requires a weight vector!"

    item_count = data.shape[0]
    assert 0 < k < item_count, "0 < k({}) < #items({}) must hold!".format(k, item_count)

    # get a distance matrix describing our data from scipy, square it so creating the knn graph is easy
    distance_matrix = squareform(pdist(data, distance_measure, p=minkowski_p, w=weighted_minkowsky_weights))

    # create an adjacency list describing the k nearest neighbours for each item
    knn_graph = numpy.apply_along_axis(lambda row: row.argsort()[1:k + 1], arr=distance_matrix, axis=1)

    # calculate the density for each item
    max_distance = numpy.max(distance_matrix)

    densities = numpy.empty((item_count,), dtype=float)
    for i in range(item_count):
        distance_sum = (numpy.sum(distance_matrix[i].take(knn_graph[i])) / k)
        if distance_sum > 0:
            densities[i] = max_distance / distance_sum
        else:
            densities[i] = sys.float_info.max

    # create item bins
    cluster_supporting_objects = []
    outliers = []
    rest = []

    # sort items
    for i in range(densities.shape[0]):
        knn_densities = densities.take(knn_graph[i])
        item_density = densities[i]
        # print "item {} with density {}; max neighbour density {}, min nd {}, sorted into".format(i, item_density, knn_densities.max(), knn_densities.min()), end=""
        if item_density <= outlier_threshold and densities[i] < knn_densities.min():
            outliers.append(i)
            # print " outliers"
        elif item_density > knn_densities.max():
            cluster_supporting_objects.append(i)
            # print " csos"
        else:
            # print " rest"
            rest.append(i)

    # print "{} csos, {} outliers, {} rest".format(len(cluster_supporting_objects), len(outliers), len(rest))

    return cluster_supporting_objects, outliers, rest
