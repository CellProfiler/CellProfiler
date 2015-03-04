# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import numpy as np
import copy
import operator as op
from contrib.cell_star.utils.params_util import *
from scipy.linalg import norm
import scipy.optimize as opt
from contrib.cell_star.core.seed import Seed
from contrib.cell_star.core.image_repo import ImageRepo
from contrib.cell_star.core.seeder import Seeder

from contrib.cell_star.parameter_fitting.pf_process import distance_norm, get_gt_snake_seeds, grow_single_seed
from contrib.cell_star.parameter_fitting.pf_rank_snake import PFRankSnake
from contrib.cell_star.parameter_fitting.pf_auto_params import pf_parameters_encode, pf_parameters_decode, \
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
    print sorted(pf_rank_parameters_decode(pf_param_vector).iteritems())
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
                    #(gt, grown.create_mutation(3, rand_range=(-5, 5))),
                    #(gt, grown.create_mutation(-3, rand_range=(-5, 5)))
                    ]
    return gt_and_grown + mutants


#
#
# OPTIMIZATION
#
#

def run(image, gt_snakes, precision=-1, avg_cell_diameter=-1, method='brute', initial_params=None):
    global calculations
    """
    :param image: input image
    :param gt_snakes: gt snakes label image
    :param precision: if initial_params is None then it is used to calculate parameters
    :param avg_cell_diameter: if initial_params is None then it is used to calculate parameters
    :param method: optimization engine
    :param initial_params: overrides precision and avg_cell_diameter
    :return:
    """
    if initial_params is None:
        params = default_parameters(segmentation_precision=precision, avg_cell_diameter=avg_cell_diameter)
    else:
        params = copy.deepcopy(initial_params)

    images = ImageRepo(image, params)

    # prepare seed and grow snakes
    encoded_star_params = pf_parameters_encode(params)
    gt_snake_seed_pairs = [(gt_snake, seed) for gt_snake in gt_snakes for seed in
                           get_gt_snake_seeds(gt_snake, radius=8, times=8)]
    gt_snake_grown_seed_pairs = \
        [(gt_snake, grow_single_seed(seed, images, params, encoded_star_params)) for gt_snake, seed in
         gt_snake_seed_pairs]

    gt_snake_grown_seed_pairs_all = reduce(op.add,
                                           [PFRankSnake.create_all(gt, grown, params) for (gt, grown) in
                                            gt_snake_grown_seed_pairs])

    gts_snakes_with_mutations = add_mutations(gt_snake_grown_seed_pairs_all)
    ranked_snakes = zip(*gts_snakes_with_mutations)[1]

    calculations = 0

    best_params_encoded, distance = optimize(
        method,
        pf_rank_parameters_encode(params),
        pf_rank_get_ranking(ranked_snakes, params)
    )
    best_params = pf_rank_parameters_decode(best_params_encoded)
    best_params_full = PFRankSnake.merge_rank_parameters(params, best_params)
    print "Snakes:", len(ranked_snakes), "Calculations:", calculations

    print "Best ranking: "
    print "\n".join([k + ": " + str(v) for k, v in sorted(best_params.iteritems())])
    print ",".join([str(v) for _,v in sorted(best_params.iteritems())])

    return best_params_full, best_params, distance


def optimize(method_name, encoded_params, distance_function):
    if method_name == 'brute':
        return optimize_brute(encoded_params, distance_function)
    else:
        raise


def optimize_brute(params_to_optimize, distance_function):
    import time
    lower_bound = [0] * len(params_to_optimize)
    upper_bound = [1] * len(params_to_optimize)

    start = time.clock()
    result = opt.brute(distance_function, zip(lower_bound, upper_bound), finish=None, Ns=4, disp=True, full_output=True)
    elapsed = time.clock() - start

    print "Opt finished:", result[:2], "Elapsed[s]:", elapsed
    # distance_function(result[0], debug=True)
    return result[0], result[1]