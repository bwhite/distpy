import numpy as np
cimport numpy as np

cdef extern from "hamming_aux.h":
    int hamdist8(void *x, void *y, int num_ints)
    int hamdist16(void *x, void *y, int num_ints)
    int hamdist32(void *x, void *y, int num_ints)
    int hamdist64(void *x, void *y, int num_ints)
    void hamdist_batch(void *xs, void *y, np.int32_t *out, int num_ints, int num_xs, int num_bytes, int (*hamdist)(void *, void *, int))
    void make_lut16bit(np.uint8_t *lut16bit)
    void hamdist_cmap_lut16(np.uint16_t *xs, np.uint16_t *ys, np.int32_t *out, int size_by_2, int num_xs, int num_ys, np.uint8_t *lut16bit)


cdef int hamdist_shift(int num_bytes):
    if num_bytes % 8 == 0:
        return 3
    elif num_bytes % 4 == 0:
        return 2
    elif num_bytes % 2 == 0:
        return 1
    return 0

cdef (int(*)(void*, void*, int)) hamdist_selector(int shift):
    if shift == 0:
        return hamdist8
    elif shift == 1:
        return hamdist16
    elif shift == 2:
        return hamdist32
    elif shift == 3:
        return hamdist64
    raise ValueError('Shift expected to be 0 <= x <= 3')


cdef class Hamming(object):
    """Hamming distance computer
    """
    cdef int shift
    cdef np.ndarray lut16bit

    def __init__(self, num_bytes=0):
        super(Hamming, self).__init__()
        if num_bytes > 0:
            self.shift = hamdist_shift(num_bytes)
        else:
            self.shift = -1
        self.lut16bit = np.zeros(2**16, dtype=np.uint8)
        make_lut16bit(<np.uint8_t *>self.lut16bit.data)
        #np.testing.assert_equal(self.lut16bit, np.fromiter(((np.sum(np.unpackbits(np.fromstring(np.uint16(x).tostring(), dtype=np.uint8))))
        #                                                    for x in xrange(2**16)), dtype=np.uint8))

    cpdef np.ndarray[np.int32_t, ndim=2, mode='c'] cdist(self, np.ndarray[np.uint8_t, ndim=2, mode='c'] a,
                                                         np.ndarray[np.uint8_t, ndim=2, mode='c'] b):
        """Computes the cartesian product between two vectors

        :param a: Matrix of a_samples x dims
        :param b: Matrix of b_samples x dims
        :returns: ndarray (a_samples x b_samples)
        """
        assert a.shape[1] == b.shape[1]
        assert a.shape[1] % 2 == 0
        cdef np.ndarray out = np.zeros((a.shape[0], b.shape[0]), dtype=np.int32)
        hamdist_cmap_lut16(<np.uint16_t *>a.data, <np.uint16_t *>b.data, <np.int32_t *>out.data,
                           a.shape[1] / 2, a.shape[0], b.shape[0], <np.uint8_t *>self.lut16bit.data)
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
        assert v0.size % 2 == 0
        cdef np.int32_t out = 0
        hamdist_cmap_lut16(<np.uint16_t *>v0.data, <np.uint16_t *>v1.data, &out, v0.size / 2, 1, 1, <np.uint8_t *>self.lut16bit.data)
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
        cdef int shift
        shift = self.shift if self.shift >= 0 else hamdist_shift(vector.size)
        cdef int (*hamdist)(void*,  void*, int)
        hamdist = hamdist_selector(shift)
        hamdist_batch(neighbors.data, vector.data, <np.int32_t *>dists.data, vector.size >> shift, neighbors.shape[0], vector.size, hamdist)
        indeces = dists.argsort()[:k]
        return np.ascontiguousarray(np.dstack([dists[indeces], indeces])[0])
