# -*- coding: utf-8 -*-
import json
import urllib
import test

from structure_information_extraction import extract_structure_information
from fuzzy_approximation import fuzzy_approximation
#from cluster_construction import cluster_construction

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


def dl_sets():
    """Helper function to try and download all data sets for testing purposes"""
    with open('datasets.json') as data_file:
        data = json.load(data_file)

    no = 1
    for set in data.itervalues():
        name = set["name"]
        training = "http://www.learning-challenge.de" + set["url_training_arff"].rstrip("/")
        test = "http://www.learning-challenge.de" + set["url_test_arff"].rstrip("/")

        if "data" in test:
            print(test)
        if "data" in training:
            print(training)
        urllib.urlretrieve(test, "datasets/" + name + '_test.arff')
        urllib.urlretrieve(training, "datasets/" + name + '_training.arff')

        no += 1

def dl_missing():
    import os
    from scipy.io import arff
    for file_name in os.listdir("datasets/"):
        no_test_points, no_training_points = False, False
        with open("datasets/" + file_name, "r") as file:
            d = file.readlines()
            no_training_points = d[10].endswith(" 0\n")
            no_test_points = d[11].endswith(" 0\n")


        if "training" in file_name and no_training_points:
            print(file_name + " has no training points")
            os.rename("datasets/" + file_name, "datasets/empty_" + file_name)

        if "test" in file_name and no_test_points:
            print(file_name + " has no test points")
            os.rename("datasets/" + file_name, "datasets/empty_" + file_name)



if __name__ == "__main__":
    """
    If run as main, all tests will be run
    """
    #test.test_iris_euclidean(lambda data, measure: flame_cluster(data, 10, 0.1, measure))
    test.run_tests(lambda data, measure: flame_cluster(data, 3, 0.1, measure))
    dl_missing()
