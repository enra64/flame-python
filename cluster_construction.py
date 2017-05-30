def cluster_construction(data, cso_count, members):

    N = data.shape[0]
    C = cso_count+1
    memberships = members
    clusters = []

    """
    Sort objects based on the "entropy" of fuzzy memberships.
    """
    for i in range(0, N, 1):
        vals[i].index = i
        vals[i].value = 0.0
        for j in range(0, C, 1):
            fs = memberships[i][j]
            if( fs > 1E-9):
                vals[i].value -= fs * log( fs )

    #PartialQuickSort( vals, 0, N-1, N )

    if( thd <0 || thd > 1.0 ):
        """
        Assign each object to the cluster
         * in which it has the highest membership.
         """
        for i in range(0, N, 1):
            id = vals[i].index
            fmax = 0
            imax = -1
            for j in range(0, C, 1):
                if( memberships[id][j] > fmax ):
                    imax = j
                    fmax = memberships[id][j]

            #IntArray_Push( self->clusters + imax, id )

    else:
        """
        Assign each object to all the clusters
         in which it has membership higher than thd,
         otherwise, assign it to the outlier group.
         """
        for i in range(0, N, 1):
            id = vals[i].index
            imax = -1
            for j in range(0, C, 1):
                if( memberships[id][j] > thd || ( j == C-1 && imax <0 ) ):
                    imax = j
                    clust = clusters + j
                    #IntArray_Push( self->clusters + j, id )
    """
    removing empty clusters
    """
    C = 0;
    for i in range(0, cso_count, 1):
        if( clusters[i].size >0 ):
            clusters[C] = clusters[i]
            C +=1


    """
    keep the outlier group, even if its empty
    """
    clusters[C] = clusters[cso_count]
    C ++
    self->count = C
