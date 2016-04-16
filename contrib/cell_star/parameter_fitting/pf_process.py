# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mr�z'

import copy
import random
import time
from multiprocessing import Process, Queue

import scipy.optimize as opt
from scipy.linalg import norm

random.seed(1)

import logging
logger = logging.getLogger(__name__)

from contrib.cell_star.utils.params_util import *
from contrib.cell_star.core.seed import Seed
from contrib.cell_star.core.image_repo import ImageRepo
from contrib.cell_star.parameter_fitting.pf_snake import PFSnake
from contrib.cell_star.core.seeder import Seeder
from contrib.cell_star.process.segmentation import Segmentation
from contrib.cell_star.utils.image_util import image_show, image_save
from contrib.cell_star.parameter_fitting.pf_auto_params import pf_parameters_encode, pf_parameters_decode

from cellprofiler.preferences import get_max_workers

min_number_of_chosen_seeds = 6
max_number_of_chosen_snakes = 20

#
#
# COST FUNCTION AND FITNESS
#
#
best_so_far = 1
calculations = 0
best_3 = []

def keep_3_best(partial_parameters, distance):
    global best_3
    best_3.append((distance, partial_parameters))
    best_3.sort(key=lambda x: x[0])
    best_3 = best_3[:3]
    if best_3[0][0] == best_3[-1][0]:
        best_3 = [best_3[0]]

def distance_norm(fitnesses):
    global calculations, best_so_far
    # Mean-Squared Error
    distance = norm((np.ones(fitnesses.shape) - fitnesses)) / np.sqrt(fitnesses.size)
    best_so_far = min(best_so_far, distance)
    calculations += 1
    if calculations % 100 == 0:
        logger.debug("Current distance: %f, Best: %f, Calc %d"%(distance, best_so_far,calculations))
    return distance


def grow_single_seed(seed, images, init_params, pf_param_vector):
    pfsnake = PFSnake(seed, images, init_params)
    return pfsnake.grow(pf_parameters_decode(pf_param_vector, pfsnake.orig_size_weight_list, init_params["segmentation"]["stars"]["step"], init_params["segmentation"]["avgCellDiameter"], init_params["segmentation"]["stars"]["maxSize"]))


def snakes_fitness(gt_snake_seed_pairs, images, parameters, pf_param_vector, debug=False):
    gt_snake_grown_seed_pairs = [(gt_snake, grow_single_seed(seed, images, parameters, pf_param_vector)) for
                                     gt_snake, seed in gt_snake_seed_pairs]

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


def pf_get_distance(gt_snakes, images, initial_parameters, callback = keep_3_best):
    gt_snake_seed_pairs = [(gt_snake, seed) for gt_snake in gt_snakes for seed in get_gt_snake_seeds(gt_snake)]
    random.shuffle(gt_snake_seed_pairs)
    pick_seed_pairs = max(min_number_of_chosen_seeds, max_number_of_chosen_snakes / len(
        initial_parameters["segmentation"]["stars"]["sizeWeight"]))
    gt_snake_seed_pairs = gt_snake_seed_pairs[:pick_seed_pairs]

    def distance(partial_parameters, debug=False):
        current_distance = distance_norm(
            snakes_fitness(gt_snake_seed_pairs, images, initial_parameters, partial_parameters, debug=debug)
        )
        if callback is not None:
            callback(partial_parameters, current_distance)

        return current_distance

    return distance


#
#
# VISUALIZATION
#
#

def test_trained_parameters(image, star_params, precision, avg_cell_diameter, output_name=None):
    seg = Segmentation(segmentation_precision=precision, avg_cell_diameter=avg_cell_diameter)
    for k, v in star_params.iteritems():
        seg.parameters["segmentation"]["stars"][k] = v
    seg.set_frame(image)
    seg.run_segmentation()
    if output_name is None:
        image_show(seg.images.segmentation, 1)
    else:
        image_save(seg.images.segmentation, output_name)

#
#
# OPTIMIZATION
#
#

def run(image, gt_snakes, precision, avg_cell_diameter, method='brute', initial_params=None):
    global best_3
    """
    :param image: input image
    :param gt_snakes: gt snakes label image
    :param precision: if initial_params is None then it is used to calculate parameters
    :param avg_cell_diameter: if initial_params is None then it is used to calculate parameters
    :param method: optimization engine
    :param initial_params: overrides precision and avg_cell_diameter
    :return:
    """
    logger.info("Parameter fitting started...")
    if initial_params is None:
        params = default_parameters(segmentation_precision=precision, avg_cell_diameter=avg_cell_diameter)
    else:
        params = copy.deepcopy(initial_params)
    images = ImageRepo(image, params)

    start = time.clock()
    best_3 = []
    optimized = optimize(method, gt_snakes, images, params, precision, avg_cell_diameter)

    best_arg = optimized[0]
    best_params = pf_parameters_decode(optimized[0], get_size_weight_list(params), params["segmentation"]["stars"]["step"], avg_cell_diameter, params["segmentation"]["stars"]["maxSize"])
    best_score = optimized[1]

    stop = time.clock()

    logger.debug("Best: \n" + "\n".join([k + ": " + str(v) for k, v in sorted(best_params.iteritems())]))
    logger.debug("Time: %d" % (stop - start))
    logger.info("Parameter fitting finished with best score %f" % best_score)
    # test_trained_parameters(image, best_params, precision, avg_cell_diameter)
    return PFSnake.merge_parameters(params, best_params), best_arg, best_score


def optimize(method_name, gt_snakes, images, params, precision, avg_cell_diameter):
    encoded_params = pf_parameters_encode(params)
    distance_function = pf_get_distance(gt_snakes, images, params)
    initial_distance = distance_function(encoded_params)
    logger.debug("Initial parameters distance is (%f)." % (initial_distance))
    if method_name == "mp" and getattr(sys, "frozen", False) and sys.platform == 'win32':
        # multiprocessing do not work then
        method_name = "brutemaxbasin"
    if method_name == "mp":
        best_params_encoded, distance = multiproc_multitype_fitness(images.image, gt_snakes, precision, avg_cell_diameter, params)
        # test_trained_parameters(images.image, params, precision, avg_cell_diameter)
        # return fitted_params, score
    else:
        if method_name == 'brute':
            best_params_encoded, distance = optimize_brute(encoded_params, distance_function)
        elif method_name == 'brutemaxbasin':
            best_params_encoded, distance = optimize_brute(encoded_params, distance_function)
            logger.debug("Best grid parameters distance is (%f)." % distance)
            best_params_encoded, distance = optimize_basinhopping(best_params_encoded, distance_function, time_percent=100)
        elif method_name == 'brutemax3basin':
            _, _ = optimize_brute(encoded_params, distance_function)
            logger.debug("Best grid parameters distance are %s." %  str(zip(*best_3)[0]))
            logger.debug("Best grid parameters parameters are %s." %  str(zip(*best_3)[1]))

            best_basins = []
            for candidate in list(best_3):
                best_basins.append(optimize_basinhopping(candidate[1], distance_function, time_percent = 33))
            best_basins.sort(key=lambda x: x[1])

            best_params_encoded, distance = best_basins[0]
        elif method_name == 'basin':
            best_params_encoded, distance = optimize_basinhopping(encoded_params, distance_function)
        elif method_name == 'diffevo':
            best_params_encoded, distance = optimize_de(encoded_params, distance_function)

    if initial_distance <= distance:
        logger.debug("Initial parameters (%f) are not worse than the best found (%f)." % (initial_distance, distance))
        return encoded_params, initial_distance
    else:
        return best_params_encoded, distance


def optimize_brute(params_to_optimize, distance_function):
    broadness = 0.1
    search_range = 10 * broadness

    lower_bound = params_to_optimize - np.maximum(np.abs(params_to_optimize), 0.1)
    #lower_bound[params_to_optimize == 0] = -100 * search_range

    upper_bound = params_to_optimize + np.maximum(np.abs(params_to_optimize), 0.1)
    #upper_bound[params_to_optimize == 0] = 100 * search_range

    # introduce random shift (0,grid step) # max 20%
    number_of_steps = 3
    step = (upper_bound - lower_bound) / float(number_of_steps)
    random_shift = np.array([random.random() * 2 / 10 for _ in range(len(lower_bound))])
    lower_bound += random_shift * step
    upper_bound += random_shift * step

    print lower_bound, upper_bound

    logger.debug("Search range: " + str(zip(lower_bound,upper_bound)))
    result = opt.brute(distance_function, zip(lower_bound, upper_bound), Ns=number_of_steps, disp=True, finish=None, full_output=True)
    logger.debug("Opt finished:" + str(result[:2]))
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
    logger.debug("Opt finished: " + str(result))
    # fitness(result.x, debug=True)
    return result.x, result.fun


def optimize_basinhopping(params_to_optimize, distance_function, time_percent = 100):
    minimizer_kwargs = {"method": "COBYLA"}
    result = opt.basinhopping(distance_function, params_to_optimize, minimizer_kwargs=minimizer_kwargs, niter=44*time_percent/100)
    logger.debug("Opt finished: " + str(result))
    return result.x, result.fun



#
#
#   MULTIPROCESSING - MULTIPLE METHODS
#
#


def run_wrapper(queue, image, gt_snakes, precision, avg_cell_diameter, method, init_params):
    random.seed()  # reseed with random
    result = run(image, gt_snakes, precision, avg_cell_diameter, method, init_params)
    queue.put(result)


def multiproc_multitype_fitness(image, gt_snakes, precision, avg_cell_diameter, init_params=None):
    result_queue = Queue()
    workers_num = get_max_workers()

    optimizers = [
        Process(target=run_wrapper, args=(result_queue, image, gt_snakes, precision, avg_cell_diameter, "brute", init_params))
        for _ in range(workers_num)]

    for optimizer in optimizers:
        optimizer.start()

    results = [result_queue.get() for o in optimizers]

    for optimizer in optimizers:
        optimizer.join()

    sorted_results = sorted(results, key=lambda x: x[2])
    logger.debug(str(sorted_results[0]))
    return sorted_results[0][1], sorted_results[0][2]