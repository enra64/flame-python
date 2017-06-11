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
        return numpy.empty((0, len(meta._attributes))), 0
    else:
        data_matrix = numpy.zeros(shape=(data.shape[0], len(data[0]) - 1))

        for i in range(len(data)):
            arff_row = data[i]

            for j in range(len(arff_row) - 1):
                data_matrix[i][j] = arff_row[j]

        return data_matrix, data.shape[0]


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


def test_all_measure(cluster_function, measure, process_count):
    """
    Test cluster_function for all available data sets. The distance_measure parameter will always be measure.

    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str)
    :param measure: the distance measure to be used; as seen in scipys pdist
    :return: nothing
    """
    print("test all sets with " + measure)
    data_set_paths = [f for f in listdir("datasets") if isfile(join("datasets", f))]

    # function for verbosely testing the clustering function
    def cluster_with_catching(path):
        try:
            data, length = __load_file("datasets/" + path)
            if length > 0:
                print "clustering " + str(path),
                cluster_function(data, measure)
                print("; did not die")
            else:
                print(str(path) + " is empty")
        except Exception as e:
            print(str(path) + " threw an EXCEPTION: " + str(type(e)) + ": " + str(e))

    # run the tests in sequence or parallel
    if process_count > 1:
        from multiprocess import Pool
        Pool(process_count).map(cluster_with_catching, data_set_paths)
    else:
        for data_set_path in data_set_paths:
            cluster_with_catching(data_set_path)


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
    Test cluster function with iris dataset and euclidean measure
    
    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str) 
    :return: nothing
    """
    test(cluster_function, "Iris_training.arff", "euclidean")


def test(cluster_function, dataset_name, measure):
    """
    Test cluster function with a single dataset and measure

    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str)
    :param dataset_name: name of the dataset file to load
    :param measure: measure to use
    :return: nothing
    """
    dataset, length = __load_file("datasets/" + dataset_name)
    if length > 0:
        cluster_function(dataset, measure)


def run_tests(cluster_function, process_count):
    """
    Run all available tests
    
    :param cluster_function: the clustering function. Signature is clustering(data: ndarray, distance_measure: str), 
        where distance_measure is a distance measurement function name as seen in scipys pdist
    :param process_count: number of parallel processes to use for testing. 1 will run sequentially.
    :return: nothing
    """
    for measure in __available_measures:
        test_all_measure(cluster_function, measure, process_count)
