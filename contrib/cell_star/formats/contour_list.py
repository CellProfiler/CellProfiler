# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

from contrib.cell_star.core.snake import Snake
import contrib.cell_star.yaml as yaml


class ContourListFormat(object):

    def __init__(self, segmentation, snakes):
        self.segmentation = segmentation
        self.snakes = snakes

    def read(self, filename):
        self.snakes = []
        with file(filename, 'r') as src:
            snake_dump = yaml.load(src)
        for snake_points in snake_dump:
            self.snakes.append(Snake({}, snake_points))

    def write(self, filename):
        snake_dump = []
        for snake in self.snakes:
            snake_dump.append(snake.points)
        with file(filename, 'w') as dst:
            yaml.dump(snake_dump, dst)