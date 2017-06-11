import numpy


def cluster_construction(fuzzyship, cluster_supporting_objects, cluster_outliers, the_rest):
    result = numpy.empty((len(fuzzyship, )))
    for i in range(len(cluster_supporting_objects)):
        result[cluster_supporting_objects[i]] = i

    outlier_cluster = len(cluster_supporting_objects)
    for outlier_index in cluster_outliers:
        result[outlier_index] = outlier_cluster

    for obj_index in the_rest:
        result[obj_index] = numpy.argmax(fuzzyship[obj_index])

    if True:
        print (
            "{} csos, {} outliers, {} rest\n".format(len(cluster_supporting_objects), len(cluster_outliers),
                                                     len(the_rest)))

        for index, cluster in enumerate(cluster_supporting_objects):
            cluster_objects = numpy.where(result == index)
            print ("Cluster {}, Members: {}\n{}".format(index + 1, len(cluster_objects[0]), cluster_objects))

    return result
