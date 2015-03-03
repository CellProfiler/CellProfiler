# -*- coding: utf-8 -*-
from cell_star.core.point import Point

__author__ = 'Adam Kaczmarek, Filip Mróz'


class Seed(Point):
    """
    Object of a seed
    @ivar x: x coordinate of seed
    @ivar y: y coordinate of seed
    @ivar origin: where seed comes from ('content' or 'background' or 'snakes')
    """

    def __init__(self, x, y, origin):
        super(Seed, self).__init__(x, y)
        self.origin = origin