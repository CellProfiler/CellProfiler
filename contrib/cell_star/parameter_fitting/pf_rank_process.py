# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import numpy as np
import operator as op
from cell_star.utils.params_util import *
from scipy.linalg import norm
import scipy.optimize as opt
from cell_star.core.seed import Seed
from cell_star.core.image_repo import ImageRepo
from cell_star.parameter_fitting.pf_snake import PFSnake
from cell_star.core.seeder import Seeder

from cell_star.parameter_fitting.pf_process import distance_norm, get_gt_snake_seeds, grow_single_seed
from cell_star.parameter_fitting.pf_rank_snake import PFRankSnake
from cell_star.parameter_fitting.pf_auto_params import pf_parameters_encode, pf_parameters_decode, \
    pf_rank_parameters_encode, pf_rank_parameters_decode

#
#
# COST FUNCTION AND FITNESS
#
#
best_so_far = 1000000000
calculations = 0


def distance_norm_list(expected, result):
    """
    Calculates number of derangments between two sequences
    @param expected: expected order
    @param result: given order
    @return:
    """
    global best_so_far, calculations
    exp_position = dict([(obj, i) for (i, obj) in enumerate(expected)])
    positions = enumerate(result)
    distance = sum([abs(exp_position[obj] - i)**2 for (i, obj) in positions])
    best_so_far = min(best_so_far, distance)
    calculations += 1
    print "Rank current:", distance, ", Best:", best_so_far
    return distance


def calc_ranking(rank_snakes, pf_param_vector):
    fitness_order = sorted(rank_snakes, key=lambda x: -x.fitness)
    ranking_order = sorted(rank_snakes, key=lambda x: x.calculate_ranking(pf_rank_parameters_decode(pf_param_vector)))
    print pf_rank_parameters_decode(pf_param_vector)
    return distance_norm_list(fitness_order, ranking_order)


def pf_rank_get_ranking(rank_snakes, initial_parameters):
    print "ranksnake len =", len(rank_snakes)

    fitness = lambda partial_parameters, debug=False: \
        calc_ranking(
            rank_snakes,
            partial_parameters
        )

    return fitness


#
#
# PREPARE DATA
#
#


def add_mutations(gt_and_grown):
    mutants = []
    for (gt, grown) in gt_and_grown:
        mutants += [(gt, grown.create_mutation(20)), (gt, grown.create_mutation(-20)),
                    (gt, grown.create_mutation(10)), (gt, grown.create_mutation(-10)),
                    (gt, grown.create_mutation(3, rand_range=(-5, 5))),
                    (gt, grown.create_mutation(-3, rand_range=(-5, 5)))]
    return gt_and_grown + mutants


#
#
# OPTIMIZATION
#
#

def run(image, gt_snakes, precision, avg_cell_diameter, method):
    global calculations
    params = default_parameters(segmentation_precision=precision, avg_cell_diameter=avg_cell_diameter)
    images = ImageRepo(image, params)

    # prepare seed and grow snakes
    encoded_star_params = pf_parameters_encode(params)
    gt_snake_seed_pairs = [(gt_snake, seed) for gt_snake in gt_snakes for seed in get_gt_snake_seeds(gt_snake)]
    gt_snake_grown_seed_pairs = \
        [(gt_snake, grow_single_seed(seed, images, params, encoded_star_params)) for gt_snake, seed in
         gt_snake_seed_pairs]

    gts_snakes_with_mutations = add_mutations(gt_snake_grown_seed_pairs)
    ranked_snakes = reduce(op.add,
                           [PFRankSnake.create_all(gt, grown, params) for (gt, grown) in gts_snakes_with_mutations])

    best_params_full = params
    best_params = pf_rank_parameters_encode(params)
    for _ in range(1):
        calculations = 0
        best_params = \
            pf_rank_parameters_decode(
                optimize(
                    method,
                    pf_rank_parameters_encode(best_params_full),
                    pf_rank_get_ranking(ranked_snakes, best_params_full)
                )
            )
        best_params_full = PFSnake.merge_parameters(params, best_params)
        print "Snakes:", len(ranked_snakes), "Calculations:", calculations

    print "Best ranking: "
    print "\n".join([k + ": " + str(v) for k, v in best_params.iteritems()])


def optimize(method_name, encoded_params, distance_function):
    if method_name == 'brute':
        return optimize_brute(encoded_params, distance_function)
    else:
        raise


def optimize_brute(params_to_optimize, distance_function):
    lower_bound = [0] * len(params_to_optimize)
    upper_bound = [1] * len(params_to_optimize)

    result = opt.brute(distance_function, zip(lower_bound, upper_bound), finish=None, Ns=8, disp=True, full_output=False)
    print "Opt finished:", result
    # distance_function(result[0], debug=True)
    return result[0], result[1]