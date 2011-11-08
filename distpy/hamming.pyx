import numpy as np
cimport numpy as np

cdef extern from "hamming_aux.h":
    int hamdist8(void *x, void *y, int num_ints)
    int hamdist16(void *x, void *y, int num_ints)
    int hamdist32(void *x, void *y, int num_ints)
    int hamdist64(void *x, void *y, int num_ints)
    void hamdist_batch(void *xs, void *y, np.int32_t *out, int num_ints, int num_xs, int num_bytes, int (*hamdist)(void *, void *, int))

cdef class Hamming(object):
    """Hamming distance computer
    """
    cdef int (*hamdist)(void*,  void*, int)
    cdef int shift

    def __init__(self, int num_bytes=0):
        super(Hamming, self).__init__()
        if num_bytes % 8 == 0:
            self.hamdist = hamdist64
            self.shift = 3
        elif num_bytes % 4 == 0:
            self.hamdist = hamdist32
            self.shift = 2
        elif num_bytes % 2 == 0:
            self.hamdist = hamdist16
            self.shift = 1
        elif num_bytes > 0:
            self.hamdist = hamdist8
            self.shift = 0
        else:  # NOTE(brandyn): Special case for default, will be used later
            self.hamdist = hamdist8
            self.shift = 0

    cpdef int dist(self,
                   np.ndarray[np.uint8_t, ndim=1, mode='c'] v0,
                   np.ndarray[np.uint8_t, ndim=1, mode='c'] v1):
        """Compute distance between two vectors
        
        Args:
            v0: Vector
            v1: Vector
        
        Returns:
            Integer values (greater is further)
        """
        assert v0.size == v1.size
        return self.hamdist(v0.data, v1.data, v0.size >> self.shift)

    cpdef np.ndarray[np.int32_t, ndim=2, mode='c'] knn(self,
                                                       np.ndarray[np.uint8_t, ndim=2, mode='c'] neighbors,
                                                       np.ndarray[np.uint8_t, ndim=1, mode='c'] vector, int k):
        """Returns the k nearest neighbors to the vector
        
        Args:
            neighbors: Iteratable of list-like objects
            vector: List-like object
            k: Number of neighbors desired
        
        Returns:
            ndarray (distance, index) (k x 2)
        """
        cdef np.ndarray[np.int32_t, ndim=1, mode='c'] dists = np.zeros(neighbors.shape[0], dtype=np.int32)
        hamdist_batch(neighbors.data, vector.data, <np.int32_t *>dists.data, vector.size >> self.shift, neighbors.shape[0], vector.size, self.hamdist)
        indeces = dists.argsort()[:k]
        return np.ascontiguousarray(np.dstack([dists[indeces], indeces])[0])
