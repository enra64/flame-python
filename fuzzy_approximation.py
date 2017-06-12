# -*- coding: utf-8 -*-

import sys

def fuzzy_approximation(data, iterations, cluster_supporting_objects, cluster_outliers, the_rest, distance_matrix, knn_graph):
    """
    Approximate the fuzzy memberships of each data item

    :param data: a numpy-matrix. each column represents an attribute each row a data item
    :param iterations: number of updates based on membership of nearest neighbors
    :param cluster_supporting_objects: objects with density higher than all its neighbors
    :param cluster_outliers: objects with density lower than all its neighbors, and lower than a predefined threshold
    :param the_rest: objects not assigned to one of the previous groups
    :param distance_matrix: a matrix of distances between data points from structure_information_extraction
    :param knn_graph: neighbours with the same distance as the k-nearest-neighbour
    :return: a tuple of lists: fuzzyship and from structure_information_extraction (cluster supporting objects, cluster outliers, rest),
        where each list contains the indices of the objects in the data matrix
        * fuzzyship: object with relative membership to each cso
        * cluster_supporting_objects (CSO): object with density higher than all its neighbors
        * cluster_outliers: object with density lower than all its neighbors, and lower than a predefined threshold
        * the_rest: object not assigned to one of the previous groups
    """

    """
    PART 1: Initialization of fuzzy membership
    """
    cso_count=len(cluster_supporting_objects)
    item_count = data.shape[0]
    cso_counter=0
    fuzzyship = [[0 for x in range(cso_count+1)]for y in range(item_count)]
    initFuzzy = [[0 for x in range(cso_count+1)]for y in range(item_count)]

    for i in range(0, item_count, 1):
        # each CSO is assigned with fixed and full membership to itself to represent one cluster
        if(i in cluster_supporting_objects):
            fuzzyship[i][cso_counter] = 1
            initFuzzy[i][cso_counter] = 1
            cso_counter+=1
        # all outliers are assigned with fixed and full membership to the outlier group
        elif(i in cluster_outliers):
            fuzzyship[i][cso_count] = 1
            initFuzzy[i][cso_count] = 1
        # the rest are assigned with equal memberships to all clusters and the outlier group
        else:
            for j in range(0, cso_count+1, 1):
                fuzzyship[i][j] = 1.0/(cso_count+1)

    # weights are only dependent on the ranking of distances of the neighbors
    weight = []
    for knn_row in knn_graph:
        neighbour_count = len(knn_row)
        calculation = 0.5 * neighbour_count * (neighbour_count + 1.0)
        weight.append([(neighbour_count - j) / calculation for j in range(neighbour_count)])

    """
    PART 2: Fuzzy membership update
    """
    for i in range(0, iterations, 1):
        deviation=0

        for j in range(0, item_count, 1):
            if(j in the_rest):
                knn = len(knn_graph[j])

                # the fuzzy membership of each object is updated by a linear combination of the fuzzy memberships of its nearest neighbors
                for k in range(0, cso_count+1, 1):
                    tmp=0

                    if(i%2==0):
                        fuzzyship[j][k] = 0;
                        for n in range(0,knn,1):
                            fuzzyship[j][k] += weight[j][n] * initFuzzy[knn_graph[j][n]][k]
                            tmp+=weight[j][n]*fuzzyship[ knn_graph[j][n] ][k]
                    else:
                        initFuzzy[j][k] = 0;
                        for n in range(0,knn,1):
                            initFuzzy[j][k] += weight[j][n]*fuzzyship[knn_graph[j][n]][k]
                            tmp+=weight[j][n]*fuzzyship[ knn_graph[j][n] ][k]

                    # calculate the deviation every item - every iteration
                    deviation+=(fuzzyship[j][k]-tmp)*(fuzzyship[j][k]-tmp)

        # break at acceptable precision
        if(deviation < 1e-6):
            break

    return fuzzyship ,cluster_supporting_objects, cluster_outliers, the_rest
