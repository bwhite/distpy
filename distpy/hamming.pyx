import numpy as np
cimport numpy as np

cdef extern from "hamming_aux.h":
    void hamdist_cmap_lut16(np.uint16_t *xs, np.uint16_t *ys, np.int32_t *out, int size, int num_xs, int num_ys)


cdef class Hamming(object):
    """Hamming distance computer
    """

    def __init__(self):
        super(Hamming, self).__init__()

    cpdef np.ndarray[np.int32_t, ndim=2, mode='c'] cdist(self, np.ndarray[np.uint8_t, ndim=2, mode='c'] a,
                                                         np.ndarray[np.uint8_t, ndim=2, mode='c'] b):
        """Computes the cartesian product between two vectors

        :param a: Matrix of a_samples x dims
        :param b: Matrix of b_samples x dims
        :returns: ndarray (a_samples x b_samples)
        """
        assert a.shape[1] == b.shape[1]
        cdef np.ndarray out = np.zeros((a.shape[0], b.shape[0]), dtype=np.int32)
        hamdist_cmap_lut16(<np.uint16_t *>a.data, <np.uint16_t *>b.data, <np.int32_t *>out.data,
                           a.shape[1], a.shape[0], b.shape[0])
        return out

    cpdef int dist(self,
                   np.ndarray[np.uint8_t, ndim=1, mode='c'] v0,
                   np.ndarray[np.uint8_t, ndim=1, mode='c'] v1):
        """Compute distance between two vectors
        

        :param v0: Vector
        :param v1: Vector
        :returns: Integer values (greater is further)
        """
        assert v0.size == v1.size
        cdef np.int32_t out = 0
        hamdist_cmap_lut16(<np.uint16_t *>v0.data, <np.uint16_t *>v1.data, &out, v0.size, 1, 1)
        return out

    cpdef np.ndarray[np.int32_t, ndim=2, mode='c'] knn(self,
                                                       np.ndarray[np.uint8_t, ndim=2, mode='c'] neighbors,
                                                       np.ndarray[np.uint8_t, ndim=1, mode='c'] vector, int k):
        """Returns the k nearest neighbors to the vector
        

        :param neighbors: Iteratable of list-like objects
        :param vector: List-like object
        :param k: Number of neighbors desired
        :returns: ndarray (distance, index) (k x 2)
        """
        assert vector.size == neighbors.shape[1]
        cdef np.ndarray[np.int32_t, ndim=1, mode='c'] dists = np.zeros(neighbors.shape[0], dtype=np.int32)
        hamdist_cmap_lut16(<np.uint16_t *>neighbors.data, <np.uint16_t *>vector.data, <np.int32_t *>dists.data,
                           neighbors.shape[1], neighbors.shape[0], 1)
        indeces = dists.argsort()[:k]
        return np.ascontiguousarray(np.dstack([dists[indeces], indeces])[0])
