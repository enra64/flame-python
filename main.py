from numpy.core.multiarray import ndarray

from typing import List, Tuple, Dict

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


def flame_cluster(data: ndarray, k: int, outlier_threshold: float) -> List[int]:
    """
    Main function. Coordinates the two phases required by the algorithm
    
    :param data: a numpy-matrix. each column represents an attribute; each row a data item
    :param k: the amount of neighbours to use for the initial knn graph
    :param outlier_threshold: the maximum density an outlier can have
    :return: a list of labels.
    """
    structure_information = extract_structure_information(data, k, outlier_threshold)
    return fuzzy_approximation(*structure_information)
