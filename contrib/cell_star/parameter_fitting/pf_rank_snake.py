# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import copy
import numpy as np
import scipy.ndimage.measurements as measure

from cell_star.core.seed import Seed
from cell_star.core.snake import Snake
from cell_star.core.polar_transform import PolarTransform
from cell_star.parameter_fitting.pf_snake import PFSnake


class PFRankSnake(object):
    def __init__(self, gt_snake, grown_snake, avg_cell_diameter, params):
        self.gt_snake = gt_snake
        self.grown_snake = grown_snake
        self.avg_cell_diameter = avg_cell_diameter
        self.initial_parameters = params
        self.fitness = PFSnake.fitness_with_gt(grown_snake, gt_snake)
        self.rank_vector = grown_snake.properties_vector(avg_cell_diameter)

    @staticmethod
    def create_all(gt_snake, grown_pf_snake, params):
        return [PFRankSnake(gt_snake, snake, grown_pf_snake.avg_cell_diameter, params) for snake in grown_pf_snake.snakes]

    def merge_rank_parameters(self, new_params):
        params = copy.deepcopy(self.initial_parameters)
        for k, v in new_params.iteritems():
            params["segmentation"]["ranking"][k] = v

        return params

    def calculate_ranking(self, ranking_params):
        return self.grown_snake.star_rank(ranking_params, self.avg_cell_diameter)