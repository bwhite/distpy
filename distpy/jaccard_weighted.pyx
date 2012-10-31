import numpy as np
cimport numpy as np

cdef extern from "jaccard_weighted_aux.h":
    void jaccard_weighted_cmap_lut16(np.uint8_t *xs, np.uint8_t *ys, np.double_t *out, int size, int num_xs, int num_ys, np.double_t *chunks, int num_chunks)
    void make_lut16_chunk(np.double_t *weights, np.double_t *chunk)


cdef class JaccardWeighted(object):
    cdef np.ndarray w, chunks
    cdef int true_size, new_size, true_bytes, new_bytes

    def __init__(self, weights):
        super(JaccardWeighted, self).__init__()
        cdef np.ndarray w, chunks, cur_chunk, cur_w
        weights = np.asfarray(weights)
        self.true_size = weights.size
        self.true_bytes = int(np.ceil(weights.size / 8.))
        self.new_size = int(np.ceil(weights.size / 16.) * 16)
        if self.true_size != self.new_size:
            weights = np.ascontiguousarray(np.hstack([weights, np.zeros(self.new_size - self.true_size)]), dtype=np.double)
        self.new_bytes = self.new_size / 8
        print((self.true_size, self.new_size, self.true_bytes, self.new_bytes))
        print(weights)
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
        print(a.shape)
        print(b.shape)
        print((self.true_size, self.new_size, self.true_bytes, self.new_bytes))
        print(weights)
        assert a.shape[1] == b.shape[1]
        assert a.shape[1] == self.true_bytes
        if self.true_bytes != self.new_bytes:  # Resize if we need to
            a = np.ascontiguousarray(np.hstack([a, np.zeros((a.shape[0], self.new_bytes - self.true_bytes), dtype=np.uint8)]))
            b = np.ascontiguousarray(np.hstack([b, np.zeros((b.shape[0], self.new_bytes - self.true_bytes), dtype=np.uint8)]))
        assert a.shape[1] == b.shape[1]
        assert a.shape[1] == self.new_bytes
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
        v0_2 = v0.reshape((1, -1))
        v1_2 = v1.reshape((1, -1))
        return self.cdist(v0_2, v1_2).flat[0]

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
        vector_2 = vector.reshape((1, -1))
        cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] dists = self.cdist(neighbors, vector_2).ravel()
        indeces = dists.argsort()[:k]
        return np.ascontiguousarray(np.dstack([dists[indeces], indeces])[0])
