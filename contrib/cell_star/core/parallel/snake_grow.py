# -*- coding: utf-8 -*-
"""
Snake grow is an attempt to run time expensive snake in parallel using multiprocessing package.
Currently not tested and on hold. Tried only once for parameter fitting.
Date: 2015-2016
Website: http://cellstar-algorithm.org/
"""

import ctypes
from copy import copy
from multiprocessing import Pool, Array, Manager

import numpy as np

from contrib.cell_star.core.image_repo import ImageRepo
from contrib.cell_star.core.polar_transform import PolarTransform
from contrib.cell_star.core.snake import Snake
from contrib.cell_star.parameter_fitting.pf_snake import PFSnake


def conv_single_image(image):
    shared_array_base = Array(ctypes.c_double, image.size)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(image.shape)
    shared_array[:] = image

    return shared_array


def conv_image_repo(images):
    return map(conv_single_image, [
        images.foreground_mask,
        images.brighter,
        images.darker,
        images.image_back_difference_blurred,
        images.image_back_difference,
        images.cell_content_mask,
        images.cell_border_mask,
        images.background,
        images.background_mask
    ])


def grow_single_snake(frame, images, parameters, seed):
    #
    #
    # RECONSTRUCT INPUT
    #
    #

    ir = ImageRepo(frame, parameters)
    ir._foreground_mask, ir._brighter, ir._darker, ir._clean, ir._clean_original, \
    ir._cell_content_mask, ir._cell_border_mask, ir._background, ir._background_mask = images

    #
    #
    # CREATE AND GROW SNAKE
    #
    #

    polar_transform = PolarTransform.instance(parameters["segmentation"]["avgCellDiameter"],
                                              parameters["segmentation"]["stars"]["points"],
                                              parameters["segmentation"]["stars"]["step"],
                                              parameters["segmentation"]["stars"]["maxSize"])

    s = Snake.create_from_seed(parameters, seed, parameters["segmentation"]["stars"]["points"], ir)

    size_weight_list = parameters["segmentation"]["stars"]["sizeWeight"]
    snakes_to_grow = [(copy(s), w) for w in size_weight_list]

    for snake, weight in snakes_to_grow:
        snake.star_grow(size_weight=weight, polar_transform=polar_transform)
        snake.calculate_properties_vec(polar_transform)

    best_snake = sorted(snakes_to_grow, key=lambda (sn, _): sn.rank)[0][0]

    pf_s = PFSnake(None, None, None, best_snake=best_snake)
    pf_s.best_snake = best_snake

    return pf_s


def grow_fun((seed, frame, images, parameters)):
    return grow_single_snake(frame, images, parameters, seed)


def add_snake(snakes, snake):
    snakes.append(snake)


def mp_snake_grow(images, parameters, seeds):
    snakes = []
    manager = Manager()
    shared_parameters = manager.dict(parameters)
    shared_images = conv_image_repo(images)
    shared_frame = conv_single_image(images.image)

    snakes = []

    pool = Pool(processes=8)
    snakes = pool.map_async(grow_fun, [(seed, shared_frame, shared_images, shared_parameters) for seed in seeds]).get()
    pool.close()
    pool.join()

    return snakes

