from typing import List, Tuple

from numpy.core.multiarray import ndarray

def extract_structure_information(data: ndarray) -> Tuple[List[int], List[int], List[int]]:
    """
    Extract structure information from the dataset
    
    :param data: a numpy-matrix. each column represents an attribute; each row a data item
    :return: a tuple of lists: (cluster supporting objects, cluster outliers, rest), 
        where each list contains the indices of the objects in the data matrix 
        * Cluster Supporting Object (CSO): object with density higher than all its neighbors
        * Cluster Outliers: object with density lower than all its neighbors, and lower than a predefined threshold
        * Rest Object: object not assigned to one of the previous groups
    """
    pass


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


def flame_cluster(data: ndarray) -> List[int]:
    """
    Main function. Coordinates the two phases required by the algorithm
    
    :param data: a numpy-matrix. each column represents an attribute; each row a data item
    :return: a list of labels.
    """
    structure_information = extract_structure_information(data)
    return fuzzy_approximation(*structure_information)
