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
"""L2Sqr Norm
"""
__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import numpy as np
import distpy
import ctypes
from . import __path__

_int_ptr = ctypes.POINTER(ctypes.c_int32)
_double_ptr = ctypes.POINTER(ctypes.c_double)
_int = ctypes.c_int32
try:
    _knn = np.ctypeslib.load_library('libknearest_neighbor', __path__[0] + '/lib/')
except OSError:
    _knn = np.ctypeslib.load_library('libknearest_neighbor', '.')
_knn.knnl2sqr.restype = ctypes.c_int
_knn.knnl2sqr.argtypes = [_double_ptr, _double_ptr, _int_ptr, _double_ptr,
                          _int, _int, _int]

class L2Sqr(distpy.BaseDistance):
    def __init__(self):
        super(L2Sqr, self).__init__()

    def dist(self, v0, v1):
        """Compute distance between two vectors
        Args:
            v0: Vector (list-like object)
            v1: Vector (list-like object)
        Returns:
            Real valued distance (greater is further)
        """
        v0 = np.asfarray(v0)
        v1 = np.asfarray(v1)
        d = v0 - v1
        return float(np.dot(d, d))

    def nn(self, neighbors, vector):
        """Returns the index of the nearest neighbor to the vector
        Args:
            neighbors: Iteratable of list-like objects
            vector: List-like object
        Returns:
            Tuple of nearest neighbor (distance, index)
        """
        return self.knn(neighbors, vector, 1)[0]

    def knn(self, neighbors, vector, k):
        """Returns the k nearest neighbors to the vector
        Args:
            neighbors: Iteratable of list-like objects
            vector: List-like object
            k: Number of neighbors desired
        Returns:
            List of (distance, index)
        """
        neighbors = np.ascontiguousarray(np.asfarray(neighbors))
        vector = np.ascontiguousarray(np.asfarray(vector))
        neighbor_indeces = np.zeros(k, dtype=np.int32)
        neighbor_dists = np.zeros(k, dtype=np.double)
        ind = _knn.knnl2sqr(vector.ctypes.data_as(_double_ptr),
                            neighbors.ctypes.data_as(_double_ptr),
                            neighbor_indeces.ctypes.data_as(_int_ptr),
                            neighbor_dists.ctypes.data_as(_double_ptr),
                            len(neighbors), len(vector), k)
        return zip(neighbor_dists.tolist(), neighbor_indeces.tolist())[:ind + 1]
