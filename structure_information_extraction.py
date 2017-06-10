# -*- coding: utf-8 -*-

import sys

import numpy

from scipy.spatial.distance import squareform, pdist

from flame_error import FlameError


def extract_structure_information(
        data,
        k,
        outlier_threshold,
        distance_measure,
        minkowski_p=None,
        weighted_minkowski_weights=None):
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
    :param minkowski_p: The p-norm to apply. Mandatory for un/weighted Minkowski distance
    :param weighted_minkowski_weights: The weight vector. Mandatory for weighted Minkowski
    :return: a tuple of lists: (cluster supporting objects, cluster outliers, rest),
        where each list contains the indices of the objects in the data matrix
        * Cluster Supporting Object (CSO): object with density higher than all its neighbors
        * Cluster Outliers: object with density lower than all its neighbors, and lower than a predefined threshold
        * Rest Object: object not assigned to one of the previous groups
        * Distance matrix: a matrix of distances between data points
        * K-Nearest neighbours: a python list of numpy arrays of the nearest neighbours of each element
    """
    # check that a p value exists if minkowski distance is used
    if (distance_measure == 'minkowski' or distance_measure == 'wminkowski') and minkowski_p is None:
        raise FlameError("Minkowski distance requires a p value to be supplied!")
    # check that a weight vector exists if weighted minkowski is used
    if distance_measure == 'wminkowski' and weighted_minkowski_weights is None:
        raise FlameError("Weighted Minkowski distance requires a weight vector!")

    item_count = data.shape[0]

    if k > item_count:
        raise FlameError("More clusters than data points requested!")
    if k <= 0:
        raise FlameError("Requested cluster center count is " + str(k) + "...")

    # get a distance matrix describing our data from scipy, square it so creating the knn graph is easy
    distance_matrix = squareform(pdist(data, distance_measure, p=minkowski_p, w=weighted_minkowski_weights))

    # creates an adjacency list where each row contains the k nearest neighbours. this is extended by those neighbours
    # that have more than 1 element with the distance of the k^th neighbour
    knn_graph = []
    for i in range(item_count):
        distance_matrix_row = distance_matrix[i]
        knns = distance_matrix_row.argsort()[1:]

        same_distance_k = k
        last_neighbour_distance = distance_matrix_row[knns[k-1]]
        for i in range(k, item_count):
            if distance_matrix_row[knns[i]] == last_neighbour_distance:
                same_distance_k += 1
            else:
                break

        knn_graph.append(knns[:same_distance_k])

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

    #print "{} csos, {} outliers, {} rest".format(len(cluster_supporting_objects), len(outliers), len(rest))


    return cluster_supporting_objects, outliers, rest, distance_matrix, knn_graph
