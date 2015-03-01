# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import copy
import random
random.seed(1)  # make it deterministic
import numpy as np
import scipy.ndimage.measurements as measure

from contrib.cell_star.core.seed import Seed
from contrib.cell_star.core.snake import Snake
from contrib.cell_star.core.polar_transform import PolarTransform


class PFSnake(object):

    def __init__(self, seed, image_repo, params, best_snake=None):
        if seed is not None:

            self.fit = 0.0  # MINIMAL FIT
            self.seed = seed
            self.snakes = []
            self.images = image_repo
            self.initial_parameters = params
            self.point_number = params["segmentation"]["stars"]["points"]
            self.orig_size_weight_list = params["segmentation"]["stars"]["sizeWeight"]
            if isinstance(self.orig_size_weight_list, float):
                self.orig_size_weight_list = [self.orig_size_weight_list]
            self.avg_cell_diameter = params["segmentation"]["avgCellDiameter"]
            self.step = params["segmentation"]["stars"]["step"]
            self.max_size = params["segmentation"]["stars"]["maxSize"]
            self.polar_transform = PolarTransform.instance(params["segmentation"]["avgCellDiameter"],
                                                           params["segmentation"]["stars"]["points"],
                                                           params["segmentation"]["stars"]["step"],
                                                           params["segmentation"]["stars"]["maxSize"])

        self.best_snake = best_snake

    @staticmethod
    def merge_parameters(initial_parameters, new_params):
        params = copy.deepcopy(initial_parameters)
        for k, v in new_params.iteritems():
            params["segmentation"]["stars"][k] = v

        return params

    def merge_parameters_with_me(self, new_params):
        return PFSnake.merge_parameters(self.initial_parameters, new_params)

    def grow(self, supplementary_parameters):
        new_parameters = self.merge_parameters_with_me(supplementary_parameters)
        s = Snake.create_from_seed(new_parameters, self.seed, self.point_number, self.images)

        size_weight_list = new_parameters["segmentation"]["stars"]["sizeWeight"]
        snakes_to_grow = [(copy.copy(s), w) for w in size_weight_list]

        for snake, weight in snakes_to_grow:
            snake.star_grow(size_weight=weight, polar_transform=self.polar_transform)
            snake.calculate_properties_vec(self.polar_transform)

        self.snakes = [grown_snake for grown_snake, _ in snakes_to_grow]
        self.best_snake = sorted(snakes_to_grow, key=lambda (sn, _): sn.rank)[0][0]

        return self

    def extract_total_mask(self, tot_shape):
        return PFSnake.extract_total_mask_of_snake(self.best_snake, tot_shape)

    @staticmethod
    def extract_total_mask_of_snake(snake, tot_shape):
        yx = snake.in_polygon_yx
        in_polygon = snake.in_polygon
        in_polygon_bounds = np.array([yx, np.array(yx) + in_polygon.shape]).flatten()

        mask = np.zeros(tot_shape, dtype=bool)
        mask[in_polygon_bounds[0]:in_polygon_bounds[2], in_polygon_bounds[1]:in_polygon_bounds[3]] = in_polygon

        return mask

    @staticmethod
    def gt_snake_intersection(snake, gt):
        yx = snake.in_polygon_yx
        in_polygon = snake.in_polygon
        in_polygon_bounds = np.array([yx, np.array(yx) + in_polygon.shape]).flatten()
        intersection_local = gt.binary_mask[in_polygon_bounds[0]:in_polygon_bounds[2],
                             in_polygon_bounds[1]:in_polygon_bounds[3]] * in_polygon
        return np.count_nonzero(intersection_local)

    @staticmethod
    def fitness_with_gt(snake, gt_snake):
        intersection = PFSnake.gt_snake_intersection(snake, gt_snake)
        return intersection / (snake.area + gt_snake.area - intersection)

    def multi_fitness(self, gt_snake):
        return max([PFSnake.fitness_with_gt(pf_snake, gt_snake) for pf_snake in self.snakes])



class GTSnake(object):

    def __init__(self, binary_mask, seed=None):
        self.binary_mask = binary_mask
        self.area = np.count_nonzero(self.binary_mask)
        if seed is not None:
            self.seed = seed
            self.centroid_x, self.centroid_y = seed.x, seed.y
        else:
            self._calculate_centroids(binary_mask)
            self.seed = Seed(self.centroid_x, self.centroid_y, "gt_snake")

    def _calculate_centroids(self, binary_mask):
        self.centroid_y, self.centroid_x = measure.center_of_mass(binary_mask, binary_mask, [1])[0]