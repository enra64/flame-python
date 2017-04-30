from typing import List, Tuple, Callable

from distance_measures import *


def get_distance_matrix(data: ndarray, distance_function: Callable[[ndarray, int, int], float]) -> ndarray:
    """
    Get a distance matrix for the data given.
    
    :param data: a numpy matrix. each column represents an attribute; each row a data item
    :param distance_function: a function that takes the data matrix and two indices, and returns the distance between 
        the objects at index one and two
    :return: an array of distances between i,j data elements where i,j are valid data object indices
    """
    item_count = data.shape[0]
    distances = numpy.ndarray(shape=(item_count, item_count), dtype=float)
    for i in range(item_count):
        for j in range(item_count):
            distances[i, j] = distance_function(data, i, j)
    return distances


def get_knn_graph(k: int, distances: ndarray) -> ndarray:
    """
    Create a k-nearest-neighbour-graph as a numpy matrix
     
    :param k: the "k" in k-nearest-neighbours
    :param distances: an array of distances between i,j data elements where i,j are valid data object indices
    :return: a 2D ndarray containing a row for each data item. Each row contains k columns denoting the nearest neighbours
    """
    # fish out the indices of the k lowest distances, store as the k nearest neighbours, ignoring the first one, hoping
    # because that is guaranteed to be the (i,i) distance
    return numpy.apply_along_axis(lambda row: row.argsort()[1:k + 1], arr=distances, axis=1)


def get_densities(knn_graph: ndarray, distances: ndarray) -> ndarray:
    """
    Calculate a 1D array of item densities
    
    :param knn_graph: a 2D ndarray containing a row for each data item. Each row contains k columns denoting the nearest neighbours
    :param distances: an array of distances between i,j data elements where i,j are valid data object indices
    :return: a 1D numpy array containing the density for object i at index i 
    """
    number_of_items = knn_graph.shape[0]
    k = knn_graph.shape[1]
    max_distance = numpy.max(distances)

    # calculate the density for each item
    densities = numpy.empty((number_of_items,), dtype=float)
    for i in range(number_of_items):
        densities[i] = (max_distance / (numpy.sum(distances[i].take(knn_graph[i])) / k))
    return densities


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
    # get a distance matrix describing our data
    distance_matrix = get_distance_matrix(data, distance_euclidean)

    # get a matrix describing the k nearest neighbours for each item
    knn_graph = get_knn_graph(k, distance_matrix)

    # calculate the density of each item
    densities = get_densities(knn_graph, distance_matrix)

    # sort the items into their respective bin
    cluster_supporting_objects = []
    outliers = []
    rest = []

    for i in range(densities.shape[0]):
        knn_densities = densities.take(knn_graph[i])
        item_density = densities[i]
        # print("item {} with density {}; max neighbour density {}, min nd {}, sorted into".format(i, item_density, knn_densities.max(), knn_densities.min()), end="")
        if item_density <= outlier_threshold and densities[i] < knn_densities.min():
            outliers.append(i)
            # print(" outliers")
        elif item_density > knn_densities.max():
            cluster_supporting_objects.append(i)
            # print(" csos")
        else:
            # print(" rest")
            rest.append(i)

    # print("{} csos, {} outliers, {} rest".format(len(cluster_supporting_objects), len(outliers), len(rest)))

    return cluster_supporting_objects, outliers, rest
