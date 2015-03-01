# -*- coding: utf-8 -*-
"""
Utilities with tools that can help with debuging / profiling CellStar
Date: 2016
Website: http://cellstar-algorithm.org/
"""
import image_util


def show_snake(snake, name):
    polygon = image_util.draw_polygons(snake.images.image, [zip(snake.xs, snake.ys)])
    image_out = polygon + (1 - polygon) * snake.images.image
    image_util.image_show(image_out, name)
