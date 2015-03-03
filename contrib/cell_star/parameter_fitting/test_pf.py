__author__ = 'Adam'

import sys
import os
import os.path as path
import numpy as np
import scipy as sp
from contrib.cell_star.core.seed import Seed
#from contrib.cell_star.test import test_utils
from contrib.cell_star.parameter_fitting.pf_process import run
from contrib.cell_star.parameter_fitting.pf_snake import GTSnake

corpus_path = "yeast_corpus/data/"


def single_mask_to_snake(bool_mask, seed=None):
    return GTSnake(bool_mask, seed)


def gt_mask_to_snakes(gt_mask):
    components, num_components = sp.ndimage.label(gt_mask, np.ones((3, 3)))
    return [single_mask_to_snake(components == label) for label in range(1, num_components + 1)]


def load_from_testset(filepath):
    """
    @param filepath: TestSetX/frame/BF_frame001.tif
    @return: loaded image
    """
    #return test_utils.load_image(path.join(corpus_path, filepath))
    return None

def try_load_image(image_path):
    try:
        pass
        #image = test_utils.load_frame(image_path)
    except:
        image = load_from_testset(image_path)
    return image

def run_pf(input_image, gt_mask, parameters):
    """

    :param input_image:
    :param gt_mask:
    :param parameters:
    :return: Best complete parameters settings, best distance
    """
    gt_snakes = gt_mask_to_snakes(gt_mask)
    best_complete_params, _, best_score = run(input_image, gt_snakes, initial_params=parameters)
    return best_complete_params, best_score

def test_pf(image_path, mask_path, precision, avg_cell_diameter, method):
    frame = try_load_image(image_path)
    gt_mask = np.array(try_load_image(mask_path), dtype=bool)

    gt_snakes = gt_mask_to_snakes(gt_mask)

    run(frame, gt_snakes, precision, avg_cell_diameter, method)


if __name__ == "__main__":
    test_pf(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5])