import numpy


def cluster_construction(fuzzyship, cluster_supporting_objects, cluster_outliers, the_rest):
    """
    Construct a vector describing the cluster memberships

    :param fuzzyship: fuzzy membership vector
    :param cluster_supporting_objects: objects with density higher than all their neighbors
    :param cluster_outliers:  objects with density lower than all their neighbors, and lower than a predefined threshold
    :param the_rest: objects not assigned to one of the previous groups
    :return: a vector where at index i is the index of the cluster the object belongs to
    """
    result = numpy.empty((len(fuzzyship, )))

    # set the cluster for each cso
    for i in range(len(cluster_supporting_objects)):
        result[cluster_supporting_objects[i]] = i

    # all outliers are in the last group
    outlier_cluster = len(cluster_supporting_objects)
    for outlier_index in cluster_outliers:
        result[outlier_index] = outlier_cluster

    # all other objects are assigned to the cluster with the greatest fuzzy membership
    for obj_index in the_rest:
        result[obj_index] = numpy.argmax(fuzzyship[obj_index])

    # debugging purposes
    if True:
        print (
            "{} csos, {} outliers, {} rest\n".format(len(cluster_supporting_objects), len(cluster_outliers),
                                                     len(the_rest)))

        for index, cluster in enumerate(cluster_supporting_objects):
            cluster_objects = numpy.where(result == index)
            print ("Cluster {}, Members: {}\n{}".format(index + 1, len(cluster_objects[0]), cluster_objects))

        print ("outliers: {}\n{}".format(len(cluster_outliers), cluster_outliers))

    return result
