def cluster_construction(
        data,
        *params):

    i, j, imax;
    N = self->N;
    C = self->cso_count+1;
    fmax;
    fuzzyships = self->fuzzyships;
    clust;

    """
    Sort objects based on the "entropy" of fuzzy memberships.
    """
    for i in range(0, N, 1):
        vals[i].index = i;
        vals[i].value = 0.0;
        for j in range(0, C, 1):
            fs = fuzzyships[i][j];
            if( fs > EPSILON ) vals[i].value -= fs * log( fs );

    PartialQuickSort( vals, 0, N-1, N );

    if( thd <0 || thd > 1.0 ):
        """
        Assign each object to the cluster
         * in which it has the highest membership.
         """
        for i in range(0, N, 1):
            id = vals[i].index;
            fmax = 0;
            imax = -1;
            for j in range(0, C, 1):
                if( fuzzyships[id][j] > fmax ):
                    imax = j;
                    fmax = fuzzyships[id][j];

            IntArray_Push( self->clusters + imax, id );

    else:
        """
        Assign each object to all the clusters
         in which it has membership higher than thd,
         otherwise, assign it to the outlier group.
         """
        for i in range(0, N, 1):
            id = vals[i].index;
            imax = -1;
            for j in range(0, C, 1):
                if( fuzzyships[id][j] > thd || ( j == C-1 && imax <0 ) ):
                    imax = j;
                    clust = self->clusters + j;
                    IntArray_Push( self->clusters + j, id );
    """
    removing empty clusters
    """
    C = 0;
    for i in range(0, cso_count, 1):
        if( self->clusters[i].size >0 ):
            self->clusters[C] = self->clusters[i];
            C ++;


    """
    keep the outlier group, even if its empty
    """
    self->clusters[C] = self->clusters[self->cso_count];
    C ++;
    self->count = C;
