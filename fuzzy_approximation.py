# -*- coding: utf-8 -*-

import sys

def fuzzy_approximation(data, knn, iterations, cluster_supporting_objects, cluster_outliers, the_rest, distance_matrix, knn_graph):
    """
    Approximate the fuzzy memberships of each data item

    :param data: a numpy-matrix. each column represents an attribute each row a data item
    :param iterations: number of updates based on membership of nearest neighbors
    :param cluster_supporting_objects: objects with density higher than all its neighbors
    :param cluster_outliers: objects with density lower than all its neighbors, and lower than a predefined threshold
    :param the_rest: objects not assigned to one of the previous groups
    :return: list of labels. index i contains the label of object i from the original data set
    """

    item_count = data.shape[0]
    cso_count=len(cluster_supporting_objects)
    k=0

    """
    PART 1: Initialization of fuzzy membership
    """
    membership = [[0 for x in range(cso_count+1)]for y in range(item_count)]
    membership2 = membership;

    for i in range(0, item_count, 1):
        if(i in cluster_supporting_objects):
            """
            Each CSO is assigned with fixed and full membership to itself to represent one cluster
            """
            membership[i][k] = membership2[i][k] = 1.0
            k+=1
        elif(i in cluster_outliers):
            """
            All outliers are assigned with fixed and full membership to the outlier group
            """
            membership[i][cso_count] = membership2[i][cso_count] = 1.0
        else:
            """
            The rest are assigned with equal memberships to all clusters and the outlier group
            """
            for fill in range(0, cso_count+1, 1):
                membership[i][fill] = membership2[i][fill] = 1.0/(cso_count+1)

    """
    Weights are only dependent on
    the ranking of distances of the neighbors, so it is more
    robust against distance transformations.
    """
    weights = [[0 for x in range(knn)]for y in range(item_count)]

    for i in range(0, item_count, 1):
        calculation = 0.5*knn*(knn+1.0)
        for j in range(0, knn, 1):
            weights[i][j] = (knn-j) / calculation

    """
    PART 2: Fuzzy membership update
    """
    for t in range(0, iterations, 1):
        acc = 0

        for i in range(0, item_count, 1):
            graph = knn_graph[i]
            weight = weights[i]
            fuzzy = membership[i]
            fuzzy2 = membership2
            calculation = 0.0

            #Check if value is even
            if(t%2==0):
                fuzzy = membership2[i]
                fuzzy2 = membership

            """
            The fuzzy membership of each object is updated by a linear combination of the fuzzy memberships of its nearest neighbors
            """
            for j in range(0, cso_count+1, 1):
                fuzzy[j] = 0.0

                for k in range(0,knn,1):
                    fuzzy[j] += weight[k] * fuzzy2[ graph[k] ][j]

                acc += (fuzzy[j] - fuzzy2[i][j]) * (fuzzy[j] - fuzzy2[i][j])
                calculation += fuzzy[j]

            for j in range(0, cso_count+1, 1):
                fuzzy[j] = fuzzy[j] / calculation

        if( acc < 1e-6 ):
            break

    #for num in fuzzy2:
        #print num

    """
    Update the membership of all objects to remove
    clusters that contains only the CSO.
    """
    for i in range(0, item_count, 1):
        graph = knn_graph[i]
        weight = weights[i]
        fuzzy = membership[i]
        fuzzy2 = membership2

        for j in range(0, cso_count+1, 1):
            fuzzy[j] = 0.0
            for k in range(0,knn,1):
                fuzzy[j] += weight[k] * fuzzy2[ graph[k] ][j]
            acc += (fuzzy[j] - fuzzy2[i][j]) * (fuzzy[j] - fuzzy2[i][j])

    #for num in fuzzy2:
        #print num
