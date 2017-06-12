# -*- coding: utf-8 -*-
import numpy

import test
from flame_error import FlameError

from structure_information_extraction import extract_structure_information
from fuzzy_approximation import fuzzy_approximation
from cluster_construction import cluster_construction


def flame_cluster(data, k, outlier_threshold, distance_measure, iterations, minkowski_p=None):
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
    :param iterations: number of updates based on membership of nearest neighbors
    :param minkowski_p: The p-norm to apply. Mandatory for Minkowski distance. Ignored otherwise.
    :return: a list of labels: a vector where at index i is the index of the cluster the object belongs to
    """
    try:
        structure_information = extract_structure_information(
            data,
            k,
            outlier_threshold,
            distance_measure,
            minkowski_p)
        approximation_information = fuzzy_approximation(data, k, iterations, *structure_information)
        return cluster_construction(*approximation_information)

    except numpy.linalg.LinAlgError as err:
        if err.message == "Singular matrix" and distance_measure == "mahalanobis":
            raise FlameError("Mahalanobis distance used for dataset with singular distance matrix. That will not work. "
                             "May be caused by having more dimensions than data points.")
    except ValueError as err:
        if "the covariance matrix is singular" in err.message and distance_measure == "mahalanobis":
            raise FlameError(
                "The number of data points is too small; the covariance matrix is singular. Try not using "
                "mahalanobis distance.")


if __name__ == "__main__":
    """
    If run as main, all tests will be run
    """
    #test.test(lambda data, measure: flame_cluster(data, 3, 0.1, measure, 100), "Banana--training.arff", "braycurtis")
    #test.test_iris_euclidean(lambda data, measure: flame_cluster(data, 20, 10, measure, 100))
    test.run_tests(lambda data, measure: flame_cluster(data, 3, 10, measure, 100, 17), process_count=8)
    #test.test_all_measure(lambda data, measure: flame_cluster(data, 3, 0.1, measure, 10, 17), "braycurtis", process_count=1)
