# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import copy
import time
from multiprocessing import Process, Queue
import operator as op
import scipy.optimize as opt

import random
random.seed(1)

import logging
logger = logging.getLogger(__name__)

from contrib.cell_star.utils.params_util import *
from contrib.cell_star.core.image_repo import ImageRepo
from contrib.cell_star.parameter_fitting.pf_process import get_gt_snake_seeds, grow_single_seed
from contrib.cell_star.parameter_fitting.pf_rank_snake import PFRankSnake
from contrib.cell_star.parameter_fitting.pf_auto_params import pf_parameters_encode, pf_rank_parameters_encode, pf_rank_parameters_decode

from cellprofiler.preferences import get_max_workers

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
    distance = sum([abs(exp_position[obj] - i) ** 2 for (i, obj) in positions])
    best_so_far = min(best_so_far, distance)
    calculations += 1
    if calculations % 100 == 0:
        logger.debug("Rank current: %f, Best: %f, Calc %d" % (distance, best_so_far, calculations))
    return distance


def calc_ranking(rank_snakes, pf_param_vector):
    fitness_order = sorted(rank_snakes, key=lambda x: -x.fitness)
    ranking_order = sorted(rank_snakes, key=lambda x: x.calculate_ranking(pf_rank_parameters_decode(pf_param_vector)))
    #logger.debug(sorted(pf_rank_parameters_decode(pf_param_vector).iteritems()))
    return distance_norm_list(fitness_order, ranking_order)


def pf_rank_get_ranking(rank_snakes, initial_parameters):
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

    if method == 'mp':
        return multiproc_optimize_brute(image, gt_snakes, precision, avg_cell_diameter)

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
    #print "Snakes:", len(ranked_snakes), "Calculations:", calculations

    logger.debug("Best ranking: \n" + "\n".join([k + ": " + str(v) for k, v in sorted(best_params.iteritems())]))
    return best_params_full, best_params, distance

#
#
#   OPTIMISATION METHODS
#
#

def optimize(method_name, encoded_params, distance_function):
    if method_name == 'brute':
        return optimize_brute(encoded_params, distance_function)
    else:
        raise # return multiproc_optimize_brute(encoded_params, distance_function)


def optimize_brute(params_to_optimize, distance_function):

    lower_bound = np.zeros(len(params_to_optimize), dtype=float)
    upper_bound = np.ones(len(params_to_optimize), dtype=float)

    # introduce random shift (0,grid step) # max 10%
    number_of_steps = 6
    step = (upper_bound - lower_bound) / float(number_of_steps)
    random_shift = np.array([random.random() * 1 / 10 for _ in range(len(lower_bound))], dtype=float)
    lower_bound += random_shift * step
    upper_bound += random_shift * step

    print lower_bound, upper_bound

    start = time.clock()
    result = opt.brute(distance_function, zip(lower_bound, upper_bound), finish=None, Ns=number_of_steps, disp=True, full_output=True)
    elapsed = time.clock() - start

    logger.debug("Opt finished: " + str(result[:2]) + " Elapsed[s]: " + str(elapsed))
    # distance_function(result[0], debug=True)
    return result[0], result[1]


def run_wrapper(queue, image, gt_snakes, precision, avg_cell_diameter, method):
    random.seed()  # reseed with random
    result = run(image, gt_snakes, precision, avg_cell_diameter, method)
    queue.put(result)


def multiproc_optimize_brute(image, gt_snakes, precision, avg_cell_diameter):
    result_queue = Queue()
    workers_num = max(1,get_max_workers())
    optimizers = [
        Process(target=run_wrapper, args=(result_queue, image, gt_snakes, precision, avg_cell_diameter, 'brute')) for _ in range(workers_num)]

    for optimizer in optimizers:
        optimizer.start()

    results = [result_queue.get() for _ in optimizers]

    for optimizer in optimizers:
        optimizer.join()

    sorted_results = sorted(results, key=lambda x: x[2])
    logger.debug(str(sorted_results[0]))
    return sorted_results[0][1], sorted_results[0][2]