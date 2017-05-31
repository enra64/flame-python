# -*- coding: utf-8 -*-

import numpy
from os import listdir
from os.path import isfile, join
import time
from scipy.io import arff

__available_measures = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
                        'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
                        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
                        'yule']


def __load_file(path):
    """
    Load a data set from path. Data set must be arff format.
    
    :param path: path to the data set
    :return: a numpy-matrix. each column represents an attribute; each row a data item
    """
    data, meta = arff.loadarff(open(path, 'r'))
    if data.shape == (0,):
        print("EMPTY DATA SET:\t\t\t\t\t\t\t\t\t\t" + meta.name.strip("\""))
        return numpy.empty((0, len(meta._attributes))), 0
    else:
        return data[meta.names()[:-1]].view(numpy.float).reshape(data.shape + (-1,)), data.shape[0]


def time_flame(cluster_function, data, measure, count):
    """
    Time our implementation of the algorithm. Averages over count executions of cluster_function(data, measure)

    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str)
    :param data: a numpy-matrix. each column represents an attribute; each row a data item
    :param measure: the distance measure to be used; as seen in scipys pdist
    :param count: over how many iterations the average should be built
    :return: average execution duration in seconds
    """
    execution_duration_sum = 0
    for i in range(count):
        start = time.time()
        cluster_function(data, measure)
        execution_duration_sum += time.time() - start
    return execution_duration_sum / count


def test_exec_duration(cluster_function, measure, path, iterations, sub_iterations):
    """
    Test the execution duration. The printed value is the minimum value required to run the algorithm when executing it
    #iterations time. Each of these iterations averages over the execution duration of #sub_iterations runs.
    
    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str)
    :param measure: the distance measure to be used; as seen in scipys pdist
    :param path: full path to the dataset that should be tested
    :param iterations: how often should the sub-iterations be run
    :param sub_iterations: how often each sub iteration times the algorithm
    :return: 
    """
    data, length = __load_file(path)

    if length <= 0:
        return

    # get best result of 10 averaged iterations
    print("Average exec duration {} [s]".format(
        min([time_flame(cluster_function, data, measure, sub_iterations) for _ in range(iterations)])))


def test_all_measure(cluster_function, measure):
    """
    Test cluster_function for all available data sets. The distance_measure parameter will always be measure.

    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str)
    :param measure: the distance measure to be used; as seen in scipys pdist
    :return: nothing
    """
    print("test all sets with " + measure)
    for data_set_path in [f for f in listdir("datasets") if isfile(join("datasets", f))]:
        dataset, length = __load_file("datasets/" + data_set_path)
        print("\trunning " + data_set_path)
        if length > 0:
            cluster_function(dataset, measure)


def test_dataset_all(cluster_function, data_set_path):
    """
    Test all available measures using one data set.
    
    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str)
    :param data_set_path: path to the data set
    :return: nothing
    """
    data, length = __load_file(data_set_path)
    if length > 0:
        for measure in __available_measures:
            cluster_function(data, measure)


def test_iris_euclidean(cluster_function):
    """
    Test cluster function with a single dataset and euclidean measure 
    
    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str) 
    :return: nothing
    """
    dataset, length = __load_file("datasets/c_Iris_test.arff")
    if length > 0:
        cluster_function(dataset, "euclidean")


def run_tests(cluster_function):
    """
    Run all available tests
    
    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str), 
        where distance_measure is a distance measurement function name as seen in scipys pdist
    :return: nothing
    """
    for measure in __available_measures:
        test_all_measure(cluster_function, measure)
