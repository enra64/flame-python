import numpy
from numpy.core.multiarray import ndarray


def distance_euclidean(data: ndarray, index_a: int, index_b: int) -> float:
    """
    Calculate the euclidean distance between two data items in the
    """
    assert data is not None and index_a < data.shape[0] and index_b < data.shape[0], "invalid distance function arguments"
    item_a = data[index_a]
    item_b = data[index_b]
    return numpy.linalg.norm(item_b - item_a)