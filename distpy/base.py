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

import numpy as np

def BaseDistance(object):
    def __init__(self):
        super(BaseDistance, self).__init__()

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
        return np.linalg.norm(v0 - v1)

    def nn(self, neighbors, vector):
        """Returns the index of the nearest neighbor to the vector
        Args:
            neighbors: Iteratable of list-like objects
            vector: List-like object
        Returns:
            Tuple of nearest neighbor (distance, index)
        """
        neighbors = np.asfarray(neighbors)
        vector = np.asfarray(vector)
        dists = [(self.dist(x, vector), ind) for ind, x in enumerate(neighbors)]
        ind = np.argmin(dists)
        return dists[ind], ind

    def knn(self, neighbors, vector, k):
        """Returns the k nearest neighbors to the vector
        Args:
            neighbors: Iteratable of list-like objects
            vector: List-like object
            k: Number of neighbors desired
        Returns:
            List of (distance, index)
        """
        neighbors = np.asfarray(neighbors)
        vector = np.asfarray(vector)
        dists = [(self.dist(x, vector), ind) for ind, x in enumerate(neighbors)]
        dists.sort()
        return dists
        

        
