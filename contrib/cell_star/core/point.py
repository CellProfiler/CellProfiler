# -*- coding: utf-8 -*-
"""
Integer point wrapper.
Date: 2013-2016
Website: http://cellstar-algorithm.org/
"""

import math


class Point(object):
    """
    @ivar x: x coordinate of point
    @ivar y: y coordinate of point
    """

    def __init__(self, x, y):
        """
        @type x: int
        @type y: int
        """
        self.x = x
        self.y = y

    def __repr__(self):
        return "Seed(x={0},y={1})".format(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def polar_coords(self, origin):
        """
        @type origin : Point
        @return: radius, angle
        @rtype: (int,int)
        """
        r_x = self.x - origin.x
        r_y = self.y - origin.y

        radius = math.sqrt(r_x ** 2 + r_y ** 2)
        angle = math.atan2(r_y, r_x)

        return radius, angle

    def as_xy(self):
        return self.x, self.y

    def euclidean_distance_to(self, other_point):
        return math.sqrt((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2)
