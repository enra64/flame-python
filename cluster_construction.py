import sys

def cluster_construction(fuzzyship,cluster_supporting_objects, cluster_outliers, the_rest):

    if (len(cluster_supporting_objects)>0):
        clusters = [[] for i in range(len(cluster_supporting_objects))]

        for index, num in enumerate(fuzzyship):
            clusters[num.index(max(num))].append(index)

        for index, cluster in enumerate(clusters):
            print ("\nCluster: {} Members: {}\n{}".format(index+1, len(clusters[index]), cluster))

        print ("\n")
    else:
        print ("\nERROR: No CSO's found -> {} csos, {} outliers, {} rest\n".format(len(cluster_supporting_objects), len(cluster_outliers), len(the_rest)))
