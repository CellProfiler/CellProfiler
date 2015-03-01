# -*- coding: utf-8 -*-
"""
Contour seed.
Date: 2013-2016
Website: http://cellstar-algorithm.org/
"""

from contrib.cell_star.core.point import Point


class Seed(Point):
    """
    Object of a seed.
    @ivar x: x coordinate of seed
    @ivar y: y coordinate of seed
    @ivar origin: where seed comes from ('content' or 'background' or 'snakes')
    """

    def __init__(self, x, y, origin):
        """
        @type x: int
        @type y: int
        """
        super(Seed, self).__init__(x, y)
        self.origin = origin
