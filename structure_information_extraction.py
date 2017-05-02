from typing import List, Tuple, Callable

from scipy.spatial.distance import squareform, pdist

from distance_measures import *


def extract_structure_information(
        data: ndarray,
        k: int,
        outlier_threshold: float,
        distance_measure: str,
        minkowski_p: float = None,
        weighted_minkowsky_weights: List[float] = None) -> Tuple[
    List[int], List[int], List[int]]:
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
    # distance_matrix = get_distance_matrix(data, distance_euclidean)
    if distance_measure == 'minkowski' or distance_measure == 'wminkowski':
        assert minkowski_p is not None, "Minkowski distance requires a p value to be supplied!"

    if distance_measure == 'wminkowski':
        assert weighted_minkowsky_weights is not None, "Weighted Minkowski distance requires a weight vector!"

    # get a distance matrix describing our data from scipy, square it so the knn graph is one line
    distance_matrix = squareform(pdist(data, distance_measure, p=minkowski_p, w=weighted_minkowsky_weights))

    # get a matrix describing the k nearest neighbours for each item
    knn_graph = numpy.apply_along_axis(lambda row: row.argsort()[1:k + 1], arr=distance_matrix, axis=1)

    # calculate the density for each item
    max_distance = numpy.max(distance_matrix)
    item_count = knn_graph.shape[0]
    densities = numpy.empty((item_count,), dtype=float)
    for i in range(item_count):
        densities[i] = (max_distance / (numpy.sum(distance_matrix[i].take(knn_graph[i])) / k))

    # create item bins
    cluster_supporting_objects = []
    outliers = []
    rest = []

    # sort items
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
