import numpy as np
cimport numpy as np
from cpython cimport array
from sklearn.metrics.pairwise import euclidean_distances
import array

cdef float euclidean_distance(np.ndarray[np.float_t, ndim = 1] x_i, np.ndarray[np.float_t, ndim = 1] x_j):
    cdef int t
    cdef double res = 0.0
    for t in range(x_i.shape[0]):
        res += (x_i[t] - x_j[t]) ** 2
    return np.sqrt(res)


cpdef np.ndarray[np.int_t, ndim = 2] c_find_new_clusters(np.ndarray[np.float_t, ndim = 2] x, int l, np.ndarray[np.int_t, ndim = 1] rejected_inds, int n_neighbors):
    cdef:
        int u = x.shape[0] - l
        int num_pairs = rejected_inds.shape[0] * n_neighbors
        np.ndarray[np.int_t, ndim = 2] pairs = np.full((num_pairs, 2), -1, dtype=np.int)
        np.ndarray[np.int_t, ndim = 1] neighbors
        np.ndarray[np.float_t, ndim = 1] dist_neighbors
        np.ndarray[np.float_t, ndim = 2] dist_matrix = np.empty((u, l+u), dtype=np.float)
        int i, j, k
        int it = 0
        double tmp

    dist_matrix = euclidean_distances(x[rejected_inds+l], x)
    # return dist_matrix
    # for i in range(l, l+u):
    for i in range(rejected_inds.shape[0]):
        # if i - l not in rejected_inds:
        #     continue
        neighbors = np.zeros(n_neighbors, dtype=np.int)
        dist_neighbors = np.repeat(1.0e+31, n_neighbors)
        for j in range(l+u):
            if rejected_inds[i] + l == j:
                continue
            # tmp = euclidean_distance(x[i], x[j])
            tmp = dist_matrix[i, j]
            for k in range(n_neighbors):
                if tmp < dist_neighbors[k]:
                    dist_neighbors[k:] = np.roll(dist_neighbors[k:], 1)
                    neighbors[k:] = np.roll(neighbors[k:], 1)
                    dist_neighbors[k] = tmp
                    neighbors[k] = j
                    break
        for k in range(n_neighbors):
            if neighbors[k] - l in rejected_inds:
                pairs[it, 0] = rejected_inds[i]
                pairs[it, 1] = neighbors[k] - l
                it = it + 1
    # return pairs
    cdef list clusters = []
    for j in range(it):
        if len(clusters) == 0:
            clusters.append([pairs[j, 0], pairs[j, 1]])
        else:
            i = 0
            for cluster in clusters:
                if pairs[j, 0] in cluster:
                    if pairs[j, 1] not in cluster:
                        cluster.append(pairs[j, 1])
                elif pairs[j, 1] in cluster:
                    if pairs[j, 0] not in cluster:
                        cluster.append(pairs[j, 0])
                # it is a new cluster
                else:
                    i += 1
            if i == len(clusters):
                clusters.append([pairs[j, 0], pairs[j, 1]])
    # merge clusters if they have intersections
    cdef int num_clusters = len(clusters)
    cdef np.ndarray[np.int_t, ndim = 1] to_keep = np.ones(num_clusters, dtype=np.int)
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            if to_keep[j] == 0 or to_keep[i] == 0:
                continue
            if np.intersect1d(clusters[i], clusters[j]).size != 0:
                clusters[i].extend(np.setdiff1d(clusters[j], clusters[i]))
                to_keep[j] = 0
    # clusters_arr = np.array(clusters)[to_keep]
    return np.array(clusters)[to_keep == 1]
