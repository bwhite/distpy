import numpy as np
cimport numpy as np

cdef extern from "jaccard_weighted_aux.h":
    void jaccard_weighted_cmap_lut16(np.uint8_t *xs, np.uint8_t *ys, np.double_t *out, int size, int num_xs, int num_ys, np.double_t *chunks, int num_chunks)
    void make_lut16_chunk(np.double_t *weights, np.double_t *chunk)


cdef class JaccardWeighted(object):
    cdef np.ndarray w, chunks

    def __init__(self, weights):
        super(JaccardWeighted, self).__init__()
        cdef np.ndarray w, chunks, cur_chunk, cur_w
        assert len(weights) % 16 == 0
        w = np.asfarray(weights).reshape((-1, 16))
        chunks = np.zeros((w.shape[0], 2**16), dtype=np.double)
        for chunk in range(w.shape[0]):
            cur_w = w[chunk, :]
            cur_chunk = chunks[chunk, :]
            make_lut16_chunk(<np.double_t*>cur_w.data,
                             <np.double_t*>cur_chunk.data)
        self.chunks = chunks

    cpdef np.ndarray[np.double_t, ndim=2, mode='c'] cdist(self, np.ndarray[np.uint8_t, ndim=2, mode='c'] a,
                                                          np.ndarray[np.uint8_t, ndim=2, mode='c'] b):
        """Computes the cartesian product between two vectors

        :param a: Matrix of a_samples x dims
        :param b: Matrix of b_samples x dims
        :returns: ndarray (a_samples x b_samples)
        """
        assert a.shape[1] == b.shape[1]
        assert a.shape[1] % 2 == 0
        assert a.shape[1] / 2 == self.chunks.shape[0]
        cdef np.ndarray out = np.zeros((a.shape[0], b.shape[0]), dtype=np.double)
        jaccard_weighted_cmap_lut16(<np.uint8_t *>a.data, <np.uint8_t *>b.data, <np.double_t *>out.data,
                                    a.shape[1], a.shape[0], b.shape[0], <np.double_t *>self.chunks.data, self.chunks.shape[0])
        return out

    cpdef np.double_t dist(self,
                           np.ndarray[np.uint8_t, ndim=1, mode='c'] v0,
                           np.ndarray[np.uint8_t, ndim=1, mode='c'] v1):
        """Compute distance between two vectors
        

        :param v0: Vector
        :param v1: Vector
        :returns: Integer values (greater is further)
        """
        assert v0.size == v1.size
        assert v0.size % 2 == 0
        assert v0.size / 2 == self.chunks.shape[0]
        cdef np.double_t out = 0
        jaccard_weighted_cmap_lut16(<np.uint8_t *>v0.data, <np.uint8_t *>v1.data, &out, v0.size, 1, 1, <np.double_t *>self.chunks.data, self.chunks.shape[0])
        return out

    cpdef np.ndarray[np.double_t, ndim=2, mode='c'] knn(self,
                                                        np.ndarray[np.uint8_t, ndim=2, mode='c'] neighbors,
                                                        np.ndarray[np.uint8_t, ndim=1, mode='c'] vector, int k):
        """Returns the k nearest neighbors to the vector
        

        :param neighbors: Iteratable of list-like objects
        :param vector: List-like object
        :param k: Number of neighbors desired
        :returns: ndarray (distance, index) (k x 2)
        """
        assert vector.size == neighbors.shape[1]
        assert vector.size % 2 == 0
        assert vector.size / 2 == self.chunks.shape[0]
        cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] dists = np.zeros(neighbors.shape[0], dtype=np.double)
        jaccard_weighted_cmap_lut16(<np.uint8_t *>neighbors.data, <np.uint8_t *>vector.data, <np.double_t *>dists.data,
                                    neighbors.shape[1], neighbors.shape[0], 1, <np.double_t *>self.chunks.data, self.chunks.shape[0])
        indeces = dists.argsort()[:k]
        return np.ascontiguousarray(np.dstack([dists[indeces], indeces])[0])
