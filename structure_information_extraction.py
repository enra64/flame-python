from typing import List, Tuple, Dict

from numpy.core.multiarray import ndarray

from distance_measures import *

def get_distance_matrix(data: ndarray) -> ndarray:
    """
    n^2 distance matrix calculation. i think half of the matrix isnt required, but im not quite sure. would probably
    be faster if i used numpy/scipy
    
    :param data: 
    :return: 
    """
    # können wir scipy nutzen? haben wir die labels richtig verstanden?
    distances = numpy.ndarray(shape=data.shape, dtype=float)
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            distances[i, j] = distance_euclidean(data, i, j)

    return distances

def get_knn_graph(data: ndarray, k: int, distances: ndarray) -> Tuple[List[ndarray], float]:
    # the knn graph should contain the n nearest neighbours for each row
    knn_graph = []

    # get the k lowest distances in the distance vector for this row
    for i in range(data.shape[0]):
        knn_graph.append(distances[i].argsort[:k])

    return knn_graph, numpy.max(distances)


def get_densities(data: ndarray, k: int, max_distance: float, knn_graph: Dict[int, ndarray], distances: ndarray) -> ndarray:
    number_of_items = data.shape[0]
    assert number_of_items == len(knn_graph), "data has %i items, knn graph has %i".format(number_of_items,
                                                                                           len(knn_graph))
    densities = numpy.array(number_of_items)

    for i in range(number_of_items):
        densities[i] = (max_distance / (numpy.sum(distances[i].take(knn_graph[i])) / k))

    return densities


def get_cluster_supporting_objects(knn_graph, densities) -> List[int]:
    cluster_supporting_objects = []
    for i in range(len(knn_graph)):
        if densities[i] > densities.take(knn_graph[i]).max():
            cluster_supporting_objects.append(i)
    return cluster_supporting_objects


def get_outliers(knn_graph, densities, outlier_threshold) -> List[int]:
    outliers = []
    for i in range(len(knn_graph)):
        if densities[i] <= outlier_threshold and densities[i] < densities.take(knn_graph[i]).max():
            outliers.append(i)
    return outliers


def get_rest(cluster_supporting_objects, outliers, data_length) -> List[int]:
    rest = []
    for i in range(data_length):
        if i not in cluster_supporting_objects and i not in outliers:
            rest.append(i)
    return rest


def extract_structure_information(data: ndarray, k: int, outlier_threshold: float) -> Tuple[
        List[int], List[int], List[int]]:
    """
    Extract structure information from the dataset

    :param data: a numpy-matrix. each column represents an attribute; each row a data item
    :param k: the amount of neighbours to use for the initial knn graph
    :param outlier_threshold: the maximum density an outlier can have
    :return: a tuple of lists: (cluster supporting objects, cluster outliers, rest), 
        where each list contains the indices of the objects in the data matrix 
        * Cluster Supporting Object (CSO): object with density higher than all its neighbors
        * Cluster Outliers: object with density lower than all its neighbors, and lower than a predefined threshold
        * Rest Object: object not assigned to one of the previous groups
    """
    # get the knn graph
    distance_matrix = get_distance_matrix(data)

    knn_graph, maximum_distance = get_knn_graph(data, k)
    densities = get_densities(data, k, maximum_distance, knn_graph)

    cluster_supporting_objects = []
    outliers = []
    rest = []

    for i in range(len(knn_graph)):
        if densities[i] > densities.take(knn_graph[i]).max():
            cluster_supporting_objects.append(i)
        elif densities[i] <= outlier_threshold and densities[i] < densities.take(knn_graph[i]).max():
            outliers.append(i)
        else:
            rest.append(i)

    return cluster_supporting_objects, outliers, rest
