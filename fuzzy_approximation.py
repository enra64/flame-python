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

    """
    PART 1: Initialization of fuzzy membership
    """
    cso_count=len(cluster_supporting_objects)
    item_count = data.shape[0]
    cso_counter=0
    fuzzyship = [[0 for x in range(cso_count+1)]for y in range(item_count)]
    initFuzzy = [[0 for x in range(cso_count+1)]for y in range(item_count)]

    for i in range(0, item_count, 1):
        #Each CSO is assigned with fixed and full membership to itself to represent one cluster
        if(i in cluster_supporting_objects):
            fuzzyship[i][cso_counter] = 1
            initFuzzy[i][cso_counter] = 1
            cso_counter+=1
        #All outliers are assigned with fixed and full membership to the outlier group
        elif(i in cluster_outliers):
            fuzzyship[i][cso_count] = 1
            initFuzzy[i][cso_count] = 1
        #The rest are assigned with equal memberships to all clusters and the outlier group
        else:
            for j in range(0, cso_count+1, 1):
                fuzzyship[i][j] = 1.0/(cso_count+1)

    #for num in membership2:
        #print num

    """
    Weights are only dependent on
    the ranking of distances of the neighbors, so it is more
    robust against distance transformations.
    """
    weight = [[0 for x in range(knn)]for y in range(item_count)]

    for i in range(0, item_count, 1):
        k = knn
        d = distance_matrix[i][knn-1]

        for j in range(knn+1, item_count,1):
            if(distance_matrix[i][j] == d):
                k += 1
            else:
                break

        calculation = 0.5*k*(k+1.0)
        for j in range(0, knn, 1):
            weight[i][j] = (k-j) / calculation

    """
    PART 2: Fuzzy membership update
    """
    for i in range(0, iterations, 1):

        for j in range(0, item_count, 1):

            if(j in the_rest):
                sum_fuzzy = 0.0

                #The fuzzy membership of each object is updated by a linear combination of the fuzzy memberships of its nearest neighbors
                for k in range(0, cso_count+1, 1):

                    if(i%2==0):
                        fuzzyship[j][k] = 0;
                        for n in range(0,knn,1):
                            fuzzyship[j][k] += weight[j][n] * initFuzzy[knn_graph[j][n]][k]
                    else:
                        initFuzzy[j][k] = 0;
                        for n in range(0,knn,1):
                            initFuzzy[j][k] += weight[j][n]*fuzzyship[knn_graph[j][n]][k]

                    sum_fuzzy += fuzzyship[j][k]

        if(i%10 == 0):
            deviation=0
            for j in range(0, item_count, 1):
                if(j in the_rest):
                    for k in range(0, cso_count+1, 1):
                        tmp=0
                        for n in range(0,knn,1):
                            tmp+=weight[j][n]*fuzzyship[ knn_graph[j][n] ][ k ]
                        deviation+=(fuzzyship[j][k]-tmp)*(fuzzyship[j][k]-tmp)

        if(deviation < 1e-6):
            break


    for num in fuzzyship:
        print num
