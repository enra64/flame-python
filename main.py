# -*- coding: utf-8 -*-
import test

from structure_information_extraction import extract_structure_information
from fuzzy_approximation import fuzzy_approximation


# from cluster_construction import cluster_construction

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
    return fuzzy_approximation(data, k, 10, *structure_information)


if __name__ == "__main__":
    """
    If run as main, all tests will be run
    """
    test.test_iris_euclidean(lambda data, measure: flame_cluster(data, 10, 0.1, measure))
    # test.run_tests(lambda data, measure: flame_cluster(data, 3, 0.1, measure))
