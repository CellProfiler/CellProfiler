# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mr�z'

import copy
import time
import numpy as np
from contrib.cell_star.utils.params_util import *
from scipy.linalg import norm
import scipy.optimize as opt
from contrib.cell_star.core.seed import Seed
from contrib.cell_star.core.image_repo import ImageRepo
from contrib.cell_star.parameter_fitting.pf_snake import PFSnake
from contrib.cell_star.core.seeder import Seeder
from contrib.cell_star.process.segmentation import Segmentation
from contrib.cell_star.utils.image_util import image_show
from contrib.cell_star.core.parallel.snake_grow import mp_snake_grow
from contrib.cell_star.parameter_fitting.pf_auto_params import parameters_range, pf_parameters_encode, pf_parameters_decode
from multiprocessing import Process, Queue

snakes_multiprocessing = False

#
#
# COST FUNCTION AND FITNESS
#
#
best_so_far = 1


def distance_norm(fitnesses):
    global best_so_far
    # Mean-Squared Error
    distance = norm((np.ones(fitnesses.shape) - fitnesses)) / np.sqrt(fitnesses.size)
    best_so_far = min(best_so_far, distance)
    print "Current distance:", distance, ", Best:", best_so_far
    return distance


def grow_single_seed(seed, images, init_params, pf_param_vector):
    pfsnake = PFSnake(seed, images, init_params)
    return pfsnake.grow(pf_parameters_decode(pf_param_vector, pfsnake.orig_size_weight_list, init_params["segmentation"]["stars"]["step"], init_params["segmentation"]["avgCellDiameter"]))


def snakes_fitness(gt_snake_seed_pairs, images, parameters, pf_param_vector, debug=False):
    if snakes_multiprocessing:
        gt_snakes, seeds = zip(*gt_snake_seed_pairs)
        merged_parameters = PFSnake.merge_parameters(
            parameters,
            pf_parameters_decode(pf_param_vector, parameters["segmentation"]["stars"]["sizeWeight"], parameters["segmentation"]["stars"]["step"], parameters["segmentation"]["avgCellDiameter"])
        )
        snakes = mp_snake_grow(images, merged_parameters, seeds)
        gt_snake_grown_seed_pairs = zip(gt_snakes, snakes)
    else:
        gt_snake_grown_seed_pairs = [(gt_snake, grow_single_seed(seed, images, parameters, pf_param_vector)) for
                                     gt_snake, seed in gt_snake_seed_pairs]

    print pf_parameters_decode(pf_param_vector, parameters["segmentation"]["stars"]["sizeWeight"], parameters["segmentation"]["stars"]["step"], parameters["segmentation"]["avgCellDiameter"])
    return np.array([pf_s.multi_fitness(gt_snake) for gt_snake, pf_s in gt_snake_grown_seed_pairs])


#
#
# PREPARE DATA
#
#


def get_gt_snake_seeds(gt_snake, times=3, radius=5):
    seed = [Seed(gt_snake.centroid_x, gt_snake.centroid_y, "optimize_star_parameters")]
    return seed + Seeder.rand_seeds(radius, times, seed)


def get_size_weight_list(params):
    l = params["segmentation"]["stars"]["sizeWeight"]
    if isinstance(l, float):
        l = [l]
    return l


def pf_get_distance(gt_snakes, images, initial_parameters):
    gt_snake_seed_pairs = [(gt_snake, seed) for gt_snake in gt_snakes for seed in get_gt_snake_seeds(gt_snake)]

    distance = lambda partial_parameters, debug=False: \
        distance_norm(
            snakes_fitness(gt_snake_seed_pairs, images, initial_parameters, partial_parameters, debug=debug)
        )

    return distance


#
#
# VISUALIZATION
#
#


def test_trained_parameters(image, parameters, precision, avg_cell_diameter):
    seg = Segmentation(segmentation_precision=precision, avg_cell_diameter=avg_cell_diameter)
    for k, v in parameters.iteritems():
        seg.parameters["segmentation"]["stars"][k] = v
    seg.set_frame(image)
    seg.run_segmentation()
    image_show(seg.images.segmentation, 1)


#
#
# OPTIMIZATION
#
#

def run(image, gt_snakes, precision=-1, avg_cell_diameter=-1, method='brute', initial_params=None):
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

    start = time.clock()
    optimized = optimize(method, gt_snakes, images, params, precision, avg_cell_diameter)

    best_params = pf_parameters_decode(optimized[0], get_size_weight_list(params), params["segmentation"]["stars"]["step"], avg_cell_diameter)
    best_score = optimized[1]

    stop = time.clock()

    print "Best: "
    print "\n".join([k + ": " + str(v) for k, v in sorted(best_params.iteritems())])
    print "Time: ", stop - start

    # test_trained_parameters(image, best_params, precision, avg_cell_diameter)
    return PFSnake.merge_parameters(params, best_params), best_params, best_score


def optimize(method_name, gt_snakes, images, params, precision, avg_cell_diameter):
    if method_name == "mp":
        fitted_params, score = multiproc_multitype_fitness(images.image, gt_snakes, precision, avg_cell_diameter)
        test_trained_parameters(images.image, params, precision, avg_cell_diameter)
        return fitted_params, score
    else:
        encoded_params = pf_parameters_encode(params)
        distance_function = pf_get_distance(gt_snakes, images, params)
        if method_name == 'brute':
            return optimize_brute(encoded_params, distance_function)
        elif method_name == 'anneal':
            return optimize_anneal(encoded_params, distance_function)
        elif method_name == 'diffevo':
            return optimize_de(encoded_params, distance_function)
        else:
            return optimize_basinhopping(encoded_params, distance_function)


def optimize_brute(params_to_optimize, distance_function):
    broadness = 0.1
    search_range = 10 * broadness

    lower_bound = params_to_optimize - np.maximum(np.abs(params_to_optimize), 0.1)
    lower_bound[params_to_optimize == 0] = -100 * search_range

    upper_bound = params_to_optimize + np.maximum(np.abs(params_to_optimize), 0.1)
    upper_bound[params_to_optimize == 0] = 100 * search_range

    result = opt.brute(distance_function, zip(lower_bound, upper_bound), Ns=5, disp=True, full_output=True)
    print "Opt finished:", result
    # distance_function(result[0], debug=True)
    return result[0], result[1]


def optimize_de(params_to_optimize, distance_function):
    broadness = 0.1
    search_range = 10 * broadness

    lower_bound = params_to_optimize - np.maximum(np.abs(params_to_optimize), 0.1)
    lower_bound[params_to_optimize == 0] = -100 * search_range

    upper_bound = params_to_optimize + np.maximum(np.abs(params_to_optimize), 0.1)
    upper_bound[params_to_optimize == 0] = 100 * search_range

    bounds = zip(lower_bound, upper_bound)
    result = opt.differential_evolution(distance_function, bounds, maxiter=10, popsize=30, init='latinhypercube')
    print "Opt finished:", result
    # fitness(result.x, debug=True)
    return result.x, result.fun


def optimize_basinhopping(params_to_optimize, distance_function):
    minimizer_kwargs = {"method": "BFGS"}
    result = opt.basinhopping(distance_function, params_to_optimize, minimizer_kwargs=minimizer_kwargs, niter=20)
    print "Opt finished:", result
    return result.x, result.fun


def optimize_anneal(params_to_optimize, distance_function):
    # DEPRECATED
    broadness = 0.1
    search_range = 10 * broadness
    temperature = 500 + (search_range * 100) ** 2

    stall_iter = 300  # int(temperature / 4)

    lower_bound = params_to_optimize - np.maximum(np.abs(params_to_optimize), 0.1)
    lower_bound[params_to_optimize == 0] = -100 * search_range

    upper_bound = params_to_optimize + np.maximum(np.abs(params_to_optimize), 0.1)
    upper_bound[params_to_optimize == 0] = 100 * search_range

    # lower_bound = [0]*len(params_to_optimize)
    # upper_bound = [1]*len(params_to_optimize)
    result = opt.anneal(distance_function, params_to_optimize, full_output=True, lower=lower_bound, upper=upper_bound, maxiter=10, schedule='cauchy')
    print "Opt finished:", result
    # fitness_function(result[0], debug=True)
    return result[0], result[1]


#
#
# CALLBACKS
#
#

# def universal_callback(method_name, params, value):
# best_so_far = min(best_so_far, distance)
#     print "Current:", distance, ", Best:", best_so_far
#     return distance


#
#
#   MULTIPROCESSING - MULTIPLE METHODS
#
#


def run_wrapper(queue, image, gt_snakes, precision, avg_cell_diameter, method):
    result = run(image, gt_snakes, precision, avg_cell_diameter, method)
    queue.put(result)


def multiproc_multitype_fitness(image, gt_snakes, precision, avg_cell_diameter):
    result_queue = Queue()

    optimizers = \
        [
            # Process(target=run_wrapper, args=(result_queue, image, gt_snakes, precision, avg_cell_diameter, "anneal")),
            # Process(target=run_wrapper, args=(result_queue, image, gt_snakes, precision, avg_cell_diameter, "basin")),
            Process(target=run_wrapper, args=(result_queue, image, gt_snakes, precision, avg_cell_diameter, "brute")),
            # Process(target=run_wrapper, args=(result_queue, image, gt_snakes, precision, avg_cell_diameter, "diffevo"))
        ]

    for optimizer in optimizers:
        optimizer.start()

    for optimizer in optimizers:
        optimizer.join()

    results = [result_queue.get() for o in optimizers]

    sorted_results = sorted(results, key=lambda x: x[2])

    return sorted_results[0][1], sorted_results[0][2]