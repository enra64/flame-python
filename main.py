# -*- coding: utf-8 -*-
import timeit

from scipy.io import arff
import numpy
import time
from numpy.core.multiarray import ndarray

from typing import List

from structure_information_extraction import extract_structure_information


def fuzzy_approximation(
        data,
        cluster_supporting_objects,
        cluster_outliers,
        the_rest):
    """
    Approximate the fuzzy memberships of each data item
    
    :param data: a numpy-matrix. each column represents an attribute; each row a data item
    :param cluster_supporting_objects: objects with density higher than all its neighbors
    :param cluster_outliers: objects with density lower than all its neighbors, and lower than a predefined threshold
    :param the_rest: objects not assigned to one of the previous groups
    :return: list of labels. index i contains the label of object i from the original data set 
    """
    pass


def flame_cluster(data, k, outlier_threshold, distance_measure):
    """
    Main function. Coordinates the two phases required by the algorithm
    
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
    :return: a list of labels.
    """
    structure_information = extract_structure_information(data, k, outlier_threshold, distance_measure)
    return fuzzy_approximation(data, *structure_information)


def time_flame(data, count):
    """
    Time our implementation of the algorithm
    
    :param data: a numpy-matrix. each column represents an attribute; each row a data item
    :param count: over how many iterations the average should be built
    :return: average execution duration in seconds
    """
    execution_duration_sum = 0
    for i in range(count):
        start = time.time()
        flame_cluster(data, 3, .1, "euclidean")
        execution_duration_sum += time.time() - start
    return execution_duration_sum / count

if __name__ == "__main__":
    """
    If run as main, this script will try to cluster some data set
    """
    # load iris test set, but cut off the last column, since that contains the class label
    data, meta = arff.loadarff(open("letters.arff", 'r'))
    data = data[meta.names()[:-1]].view(numpy.float).reshape(data.shape + (-1,))

    # get best result of 10 averaged iterations
    print("Average exec duration {} [s]".format(min([time_flame(data, 10) for i in range(5)])))
