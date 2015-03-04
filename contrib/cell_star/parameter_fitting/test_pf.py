__author__ = 'Adam'

import sys
import os
import os.path as path
import numpy as np
import scipy as sp
from contrib.cell_star.core.seed import Seed
from contrib.cell_star.utils import image_util
from contrib.cell_star.parameter_fitting.pf_process import run
from contrib.cell_star.parameter_fitting.pf_snake import GTSnake

corpus_path = "C:/workspace/CellStar - Python/Yeast/trunk/cell_star_plugins/yeast_corpus/data/"
#corpus_path = "D:/Fafa/Drozdze/Repozytorium/cell_star_plugins/yeast_corpus/data/"


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
    return image_util.load_image(path.join(corpus_path, filepath))


def try_load_image(image_path):
    return image_util.load_frame(corpus_path, image_path)


def run_pf(input_image, gt_mask, parameters, precision, avg_cell_diameter):
    """

    :param input_image:
    :param gt_mask:
    :param parameters:
    :return: Best complete parameters settings, best distance
    """
    gt_snakes = gt_mask_to_snakes(gt_mask)
    best_complete_params, _, best_score = run(input_image, gt_snakes, precision=precision, avg_cell_diameter=avg_cell_diameter, initial_params=parameters, method='brute')
    return best_complete_params, best_score


def test_pf(image_path, mask_path, precision, avg_cell_diameter, method):
    frame = try_load_image(image_path)
    gt_mask = np.array(try_load_image(mask_path), dtype=bool)

    gt_snakes = gt_mask_to_snakes(gt_mask)

    run(frame, gt_snakes, precision, avg_cell_diameter, method)


if __name__ == "__main__":
    test_pf(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5])