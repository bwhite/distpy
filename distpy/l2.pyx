import numpy as np
cimport numpy as np

cdef class L2Base(object):
    """L2 Norm implemented to facilitate simple metric implementation
        
    To create a new norm extend this class and implement 'dist'
    """

    def __init__(self):
        super(L2Base, self).__init__()

    cpdef double dist(self,
                      np.ndarray[np.float64_t, ndim=1, mode='c'] v0,
                      np.ndarray[np.float64_t, ndim=1, mode='c'] v1):
        """Compute distance between two vectors
        
        :param v0: Vector
        :param v1: Vector
        :returns: Real valued distance (greater is further)
        """
        return np.linalg.norm(v0 - v1)

    cpdef np.ndarray[np.float64_t, ndim=1, mode='c'] nn(self,
             np.ndarray[np.float64_t, ndim=2, mode='c'] neighbors,
             np.ndarray[np.float64_t, ndim=1, mode='c'] vector):
        """Returns the index of the nearest neighbor to the vector

        :param neighbors: Iteratable of list-like objects
        :param vector: List-like object
        :returns: ndarray of (distance, index)
        """
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] dists
        cdef int ind
        dists = np.asfarray([self.dist(x, vector) for x in neighbors])
        ind = np.argmin(dists)
        return np.asfarray([dists[ind], ind])

    cpdef np.ndarray[np.float64_t, ndim=2, mode='c'] knn(self,
              np.ndarray[np.float64_t, ndim=2, mode='c'] neighbors,
              np.ndarray[np.float64_t, ndim=1, mode='c'] vector, int k):
        """Returns the k nearest neighbors to the vector
        

        :param neighbors: Iteratable of list-like objects
        :param vector: List-like object
        :param k: Number of neighbors desired
        :returns: ndarray (distance, index) (k x 2)
        """
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] dists
        dists = np.asfarray([(self.dist(x, vector), ind)
                            for ind, x in enumerate(neighbors)])
        dists = dists[dists[:, 0].argsort(), :]
        return dists[:k]

cdef extern from "knearest_neighbor.h":
    int knnl2sqr(np.float64_t *test_point, np.float64_t *train_points, np.int32_t *neighbor_indeces, np.float64_t *neighbor_dists,
                 int num_train_points, int num_dims, int num_neighbors)
    void nnsl2sqr(np.float64_t *test_points, np.float64_t *train_points, np.float64_t *neighbor_dist_indeces, int num_test_points, int num_train_points, int num_dims)

cdef class L2Sqr(object):
    """L2 Norm Squared
    """

    def __init__(self):
        super(L2Sqr, self).__init__()

    cpdef double dist(self,
                      np.ndarray[np.float64_t, ndim=1, mode='c'] v0,
                      np.ndarray[np.float64_t, ndim=1, mode='c'] v1):
        """Compute distance between two vectors
        

        :param v0: Vector
        :param v1: Vector
        :returns: Real valued distance (greater is further)
        """
        d = v0 - v1
        return np.dot(d, d)

    cpdef np.ndarray[np.float64_t, ndim=1, mode='c'] nn(self,
             np.ndarray[np.float64_t, ndim=2, mode='c'] neighbors,
             np.ndarray[np.float64_t, ndim=1, mode='c'] vector):
        """Returns the index of the nearest neighbor to the vector
        

        :param neighbors: Iteratable of list-like objects
        :param vector: List-like object
        :returns: ndarray of (distance, index)
        """
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] out = self.knn(neighbors, vector, 1)
        return out[0]


    cpdef np.ndarray[np.float64_t, ndim=2, mode='c'] nns(self,
             np.ndarray[np.float64_t, ndim=2, mode='c'] neighbors,
             np.ndarray[np.float64_t, ndim=2, mode='c'] vectors):
        """Returns the index of the nearest neighbor to the vector

        :param neighbors: Iteratable of list-like objects
        :param vectors: List-like object
        :returns: Ndarray of (distance, index)
        """
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] out = np.zeros((len(vectors), 2))
        try:
            assert neighbors.shape[1] == vectors.shape[1]
        except:
            raise ValueError('Shape Mismatch: [%s] [%s]' % (neighbors.shape[1], vectors.shape[1]))
        nnsl2sqr(<np.float64_t *>vectors.data, <np.float64_t *>neighbors.data, <np.float64_t *>out.data,
                 vectors.shape[0], neighbors.shape[0], neighbors.shape[1])
        return out

    cpdef np.ndarray[np.float64_t, ndim=2, mode='c'] knn(self,
              np.ndarray[np.float64_t, ndim=2, mode='c'] neighbors,
              np.ndarray[np.float64_t, ndim=1, mode='c'] vector, int k):
        """Returns the k nearest neighbors to the vector

        :param neighbors: Iteratable of list-like objects
        :param vector: List-like object
        :param k: Number of neighbors desired
        :returns: ndarray (distance, index) (k x 2)
        """
        cdef np.ndarray[np.int32_t, ndim=1, mode='c'] neighbor_indeces
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] neighbor_dists
        neighbor_indeces = np.zeros(k, dtype=np.int32)
        neighbor_dists = np.zeros(k, dtype=np.float64)
        assert neighbors.shape[1] == vector.shape[0]
        ind = knnl2sqr(<np.float64_t *>vector.data,
                       <np.float64_t *>neighbors.data,
                       <np.int32_t *>neighbor_indeces.data,
                       <np.float64_t *>neighbor_dists.data,
                       len(neighbors), len(vector), k)
        return np.ascontiguousarray(np.asfarray([neighbor_dists[:ind + 1],
                                                 neighbor_indeces[:ind + 1]]).T)
