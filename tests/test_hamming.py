#!/usr/bin/env python
# (C) Copyright 2010 Brandyn A. White
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Test
"""
__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import unittest
import numpy as np
import distpy
import types
import time
from scipy.spatial.distance import hamming as hamming_sp
import scipy as sp
import scipy.spatial
import contextlib


@contextlib.contextmanager
def timer(name):
    st = time.time()
    yield
    print('[%s]: %s' % (name, time.time() - st))



def unpack_vector(vec):
    return np.ascontiguousarray(np.hstack([np.unpackbits(x) for x in vec]))


def slow_hamming_knn(neighbors, vector, k):
    vector = unpack_vector(vector)
    dists = np.array([hamming_sp(np.unpackbits(x), vector) * vector.size for x in neighbors], dtype=np.int32)
    indeces = dists.argsort()[:k]
    return np.ascontiguousarray(np.dstack([dists[indeces], indeces])[0])


class Test(unittest.TestCase):
    def test_0(self):
        num_vecs = 1000
        dist = distpy.Hamming()
        for k in [1, 2, 5, 10, 20, num_vecs + 10]:
            for num_dims in [1, 2, 3, 4, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128]:
                neighbors = np.array(np.random.randint(0, 256, (num_vecs, num_dims)), dtype=np.uint8)
                vector = np.array(np.random.randint(0, 256, num_dims), dtype=np.uint8)
                with timer('distpy[%d]' % num_dims):
                    out = dist.knn(neighbors, vector, k)
                with timer('slow[%d]' % num_dims):
                    out2 = slow_hamming_knn(neighbors, vector, k)
                np.testing.assert_equal(out, out2)
                self.assertEqual(out.shape[0], min(num_vecs, k))
                hdist = np.sum(unpack_vector(neighbors[out[0, 1]]) != unpack_vector(vector))
                self.assertEqual(hdist, out[0, 0])

    def test_1(self):
        num_samples = 1000
        for num_dims in [1, 2, 3, 4, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128]:
            print('NumDims[%d]' % num_dims)
            vecs0 = (np.random.random((num_samples, num_dims)) < .5).astype(np.int)
            vecs_packed0 = np.packbits(vecs0, axis=1)
            vecs1 = (np.random.random((num_samples, num_dims)) < .5).astype(np.int)
            vecs_packed1 = np.packbits(vecs1, axis=1)
            with timer('SP[%d]' % num_dims):
                sp_out = (sp.spatial.distance.cdist(vecs0, vecs1, 'hamming') * num_dims).astype(np.int)
            with timer('DP[%d]' % num_dims):
                dp_out = distpy.Hamming().cdist(vecs_packed0, vecs_packed1)
            np.testing.assert_equal(sp_out, dp_out)

if __name__ == '__main__':
    unittest.main()
