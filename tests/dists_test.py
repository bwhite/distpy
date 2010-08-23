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

class Test(unittest.TestCase):
    def test_0(self):
        num_vecs = 100
        num_dims = 1000
        neighbors = np.random.random((num_vecs, num_dims))
        vector = np.random.random(num_dims)
        dist_metrics = [getattr(distpy, dist_module) for dist_module in dir(distpy)]
        dist_metrics = [x for x in dist_metrics
                        if isinstance(x, types.TypeType)]
        dist_metrics = [x for x in dist_metrics
                        if issubclass(x, distpy.BaseDistance)]
        for dist_metric in dist_metrics:
            print(dist_metric)
            st = time.time()
            dist = dist_metric()
            self.assertAlmostEqual(dist.dist(vector, vector), 0.)
            self.assertAlmostEqual(dist.nn(neighbors, vector)[0],
                                   dist.knn(neighbors, vector, 1)[0][0])
            self.assertEqual(dist.nn(neighbors, vector)[1],
                                   dist.knn(neighbors, vector, 1)[0][1])
            for x in range(num_vecs + 1):
                self.assertEqual(len(dist.knn(neighbors, vector, x)), x)
            self.assertEqual(len(dist.knn(neighbors, vector, num_vecs + 2)), num_vecs)
            self.assertTrue(isinstance(dist.knn(neighbors, vector, num_vecs + 2)[0][0], float))
            self.assertTrue(isinstance(dist.knn(neighbors, vector, num_vecs + 2)[0][1], int))
            self.assertTrue(isinstance(dist.nn(neighbors, vector)[0], float))
            self.assertTrue(isinstance(dist.nn(neighbors, vector)[1], int))
            self.assertFalse(isinstance(dist.knn(neighbors, vector, num_vecs + 2)[0][0], np.generic))
            self.assertFalse(isinstance(dist.knn(neighbors, vector, num_vecs + 2)[0][1], np.generic))
            self.assertFalse(isinstance(dist.nn(neighbors, vector)[0], np.generic))
            self.assertFalse(isinstance(dist.nn(neighbors, vector)[1], np.generic))
            for x in dist.knn(neighbors, vector, num_vecs + 2):
                self.assertEqual(len(x), 2)
            print('Vecs[%d] Dims[%d] - Time[%f] - %s' % (num_vecs, num_dims, time.time() - st, str(dist_metric)))

if __name__ == '__main__':
    unittest.main()
