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
                        if issubclass(x, distpy.L2)]
        dist0 = distpy.L2()
        dist1 = distpy.L2Sqr()
        # Test NN vs 1-KNN
        np.testing.assert_equal(dist0.knn(neighbors, vector, 1)[0],
                                dist0.nn(neighbors, vector))
        np.testing.assert_equal(dist1.knn(neighbors, vector, 1)[0],
                                dist1.nn(neighbors, vector))
        # Test KNN
        for k in range(num_vecs + 2):
            d0 = dist0.knn(neighbors, vector, k)
            d1 = dist1.knn(neighbors, vector, k)
            self.assertEqual(len(d0), len(d1))
            for x, y in zip(d0, d1):
                self.assertEqual(x[1], y[1])
                self.assertAlmostEqual(x[0] * x[0], y[0])

    def test_nns(self):
        d = distpy.L2Sqr()
        vs0 = np.random.random((50, 50))
        vs1 = np.random.random((50, 50))
        out0 = d.nns(vs0, vs1)
        out1 = [d.nn(vs0, x) for x in vs1]
        np.testing.assert_almost_equal(out0, out1)

if __name__ == '__main__':
    unittest.main()
