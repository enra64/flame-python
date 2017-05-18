# -*- coding: utf-8 -*-

import sys
import numpy as np

def fuzzy_approximation(data,cluster_supporting_objects,cluster_outliers,the_rest):

    #  tuple (m, n), where m is the number of rows, and n is the number of columns
    data = np.matrix(data)
    print (data)
    print (data.shape[0])
    c=0
    m=len(cluster_supporting_objects)

    """
    Approximate the fuzzy memberships of each data item

    :param data: a numpy-matrix. each column represents an attribute each row a data item
    :param cluster_supporting_objects: objects with density higher than all its neighbors
    :param cluster_outliers: objects with density lower than all its neighbors, and lower than a predefined threshold
    :param the_rest: objects not assigned to one of the previous groups
    :return: list of labels. index i contains the label of object i from the original data set
    """

    '''

    """
    PART 1: Initialization of fuzzy membership
    """
    for i in range(0, data.shape[0], 1):
        if( i in cluster_supporting_objects ):
            """
            Each CSO is assigned with fixed and full membership to itself to represent one cluster
            """
            membership[i][c] = 1.0
            membership2[i][c] = 1.0
            c ++
        elif( i in cluster_outliers ):
            """
            All outliers are assigned with fixed and full membership to the outlier group
            """
            membership[i][m] = 1.0
            membership2[i][m] = 1.0
        else:
            """
            The rest are assigned with equal memberships to all clusters and the outlier group
            """
            for j in range(0, m+1, 1):
                membership[i][j] = membership2[i][j] = 1.0/(m+1)

    """
    PART 2: Fuzzy membership update
    """
    for t in range(0, steps, 1):
        for i in range(0, n, 1):
            knn = self->nncounts[i]
            ids = self->graph[i]
            wt = self->weights[i]
            fuzzy = membership[i]
            fuzzy2 = membership2
            double sum = 0.0
            if( self->obtypes[i] != the_rest )
                continue
            if( even ):
                fuzzy = membership2[i]
                fuzzy2 = membership

            """
            The fuzzy membership of each object is updated by a linear combination of the fuzzy memberships of its nearest neighbors
            """
            for j in range(0, m+1, 1):
                fuzzy[j] = 0.0
                for(k=0 k<knn k++) fuzzy[j] += wt[k] * fuzzy2[ ids[k] ][j]
                dev += (fuzzy[j] - fuzzy2[i][j]) * (fuzzy[j] - fuzzy2[i][j])
                sum += fuzzy[j]

            for j in range(0, m+1, 1):
                fuzzy[j] = fuzzy[j] / sum

        even = ! even
        if( dev < epsilon )
            break

    """
    update the membership of all objects to remove
    clusters that contains only the CSO.
    """
    for i in range(0, n, 1):
        knn = self->nncounts[i]
        ids = self->graph[i]
        wt = self->weights[i]
        fuzzy = membership[i]
        fuzzy2 = membership2
        for j in range(0, m+1, 1):
            fuzzy[j] = 0.0
            for(k=0 k<knn k++)
                fuzzy[j] += wt[k] * fuzzy2[ ids[k] ][j]
            dev += (fuzzy[j] - fuzzy2[i][j]) * (fuzzy[j] - fuzzy2[i][j])
    '''
if __name__ == "__main__":
    fuzzy_approximation('1 2; 3 4;1 2; 3 4;1 2; 3 4;1 2; 3 4')
