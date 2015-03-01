# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import copy
import random
random.seed(1)  # make it deterministic
import numpy as np

from contrib.cell_star.core.polar_transform import PolarTransform
from contrib.cell_star.parameter_fitting.pf_snake import PFSnake


class PFRankSnake(object):
    def __init__(self, gt_snake, grown_snake, avg_cell_diameter, params):
        self.gt_snake = gt_snake
        self.grown_snake = grown_snake
        self.avg_cell_diameter = avg_cell_diameter
        self.initial_parameters = params
        self.fitness = PFSnake.fitness_with_gt(grown_snake, gt_snake)
        self.rank_vector = grown_snake.properties_vector(avg_cell_diameter)
        self.polar_transform = PolarTransform.instance(params["segmentation"]["avgCellDiameter"],
                                                           params["segmentation"]["stars"]["points"],
                                                           params["segmentation"]["stars"]["step"],
                                                           params["segmentation"]["stars"]["maxSize"])

    @staticmethod
    def create_all(gt_snake, grown_pf_snake, params):
        return [(gt_snake, PFRankSnake(gt_snake, snake, grown_pf_snake.avg_cell_diameter, params)) for snake in grown_pf_snake.snakes]

    def create_mutation(self, dilation, rand_range=[0, 0]):
        mutant_snake = copy.copy(self.grown_snake)
        # zero rank so it recalculates
        mutant_snake.rank = None

        boundary_change = np.array([dilation + random.randrange(rand_range[0], rand_range[1] + 1)
                                    for _ in range(mutant_snake.polar_coordinate_boundary.size)])

        mutant_snake.polar_coordinate_boundary = np.maximum(np.minimum(
            mutant_snake.polar_coordinate_boundary + boundary_change,
            len(self.polar_transform.R) - 1), 3)
        # TODO need to update self.final_edgepoints to calculate properties (for now we ignore this property)
        mutant_snake.calculate_properties_vec(self.polar_transform)

        return PFRankSnake(self.gt_snake,mutant_snake,self.avg_cell_diameter,self.initial_parameters)

    @staticmethod
    def merge_rank_parameters(initial_parameters, new_params):
        params = copy.deepcopy(initial_parameters)
        for k, v in new_params.iteritems():
            params["segmentation"]["ranking"][k] = v

        return params

    def merge_parameters_with_me(self, new_params):
        return PFRankSnake.merge_rank_parameters(self.initial_parameters, new_params)


    def calculate_ranking(self, ranking_params):
        return self.grown_snake.star_rank(ranking_params, self.avg_cell_diameter)