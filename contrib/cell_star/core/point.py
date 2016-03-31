# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import math

import numpy as np


class Point(object):
    """
    Object of a point
    @ivar x: x coordinate of point
    @ivar y: y coordinate of point
    """

    @classmethod
    def from_polar_coords(cls, angle, radius, origin):
        x = origin.x + np.cos(angle) * radius
        y = origin.y + np.sin(angle) * radius
        return cls(x, y)

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Seed(x={0},y={1})".format(self.x, self.y)

    def polar_coords(self, origin):
        r_x = self.x - origin.x
        r_y = self.y - origin.y

        radius = math.sqrt(r_x**2 + r_y**2)
        angle = math.atan2(r_y, r_x)

        return radius, angle

    def as_pair(self):
        return self.x, self.y

    def euclidean_distance_to(self, other_point):
        return math.sqrt((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y