import arff
import numpy
from numpy.core.multiarray import ndarray

from typing import List

from structure_information_extraction import extract_structure_information


def fuzzy_approximation(
        data: ndarray,
        cluster_supporting_objects: List[int],
        cluster_outliers: List[int],
        the_rest: List[int]) -> List[int]:
    """
    Approximate the fuzzy memberships of each data item
    
    :param data: a numpy-matrix. each column represents an attribute; each row a data item
    :param cluster_supporting_objects: objects with density higher than all its neighbors
    :param cluster_outliers: objects with density lower than all its neighbors, and lower than a predefined threshold
    :param the_rest: objects not assigned to one of the previous groups
    :return: list of labels. index i contains the label of object i from the original data set 
    """
    pass


def flame_cluster(data: ndarray, k: int, outlier_threshold: float, distance_measure: str) -> List[int]:
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


if __name__ == "__main__":
    """
    If run as main, this script will try to cluster the iris data set
    """
    # load iris test set, but cut off the last column, since that contains the class label
    data = numpy.array(arff.load(open("iris.arff", 'r'))["data"], dtype=float)[:, :-1]
    flame_cluster(data, 3, .1, "euclidean")
