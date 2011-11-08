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
        dist_generic = distpy.Hamming()
        dist_8 = distpy.Hamming(1)
        for k in [1, 2, 5, 10, 20, num_vecs + 10]:
            for num_dims in [1, 2, 3, 4, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128]:
                neighbors = np.array(np.random.randint(0, 256, (num_vecs, num_dims)), dtype=np.uint8)
                vector = np.array(np.random.randint(0, 256, num_dims), dtype=np.uint8)
                dist = distpy.Hamming(num_dims)
                st0 = time.time()
                out = dist.knn(neighbors, vector, k)
                st0 = time.time() - st0
                st1 = time.time()
                out1 = dist_generic.knn(neighbors, vector, k)
                st1 = time.time() - st1
                st2 = time.time()
                out2 = slow_hamming_knn(neighbors, vector, k)
                st2 = time.time() - st2
                st3 = time.time()
                out3 = dist_8.knn(neighbors, vector, k)
                st3 = time.time() - st3
                np.testing.assert_equal(out, out1)
                np.testing.assert_equal(out, out2)
                np.testing.assert_equal(out, out3)
                self.assertEqual(out.shape[0], min(num_vecs, k))
                hdist = np.sum(unpack_vector(neighbors[out[0, 1]]) != unpack_vector(vector))
                self.assertEqual(hdist, out[0, 0])
                print('k[%d] Bytes[%d] Out[0] = [%s] hdists[%d] t0[%f] t1[%f] t2[%f] t3[%f] r_generic[%f] r8[%f] r_slow[%f]' % (k, num_dims, out[0, :], hdist, st0, st1, st2, st3, st1 / st0, st3 / st0, st2 / st0))

if __name__ == '__main__':
    unittest.main()
