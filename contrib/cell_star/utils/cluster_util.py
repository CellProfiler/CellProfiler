# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

# External imports
import numpy as np
# Internal imports
from calc_util import euclidean_norm
from cell_star.core.seed import Seed


def pdist(list_points, _):
        n = len(list_points)
        result = np.zeros((n, n))
        for i in xrange(n):
            for j in xrange(i + 1, n):
                dist = euclidean_norm(list_points[i], list_points[j])
                result[i][j] = dist
                result[j][i] = dist
        return result


def square_form(pairz):
    return np.array(pairz)


def point_list_as_seeds(points, origin):
    return [Seed(point[0], point[1], origin) for point in points]