__author__ = 'Adam'

import sys
import numpy as np
import contrib.cell_star.parameter_fitting.pf_rank_process as pf_rank
from contrib.cell_star.parameter_fitting.test_pf import try_load_image, gt_mask_to_snakes

import logging
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger = logging.getLogger('contrib.cell_star.parameter_fitting')
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)

corpus_path = "yeast_corpus/data/"

def run_rank_pf(input_image, gt_mask, parameters):
    """
    :param input_image:
    :param gt_mask:
    :param parameters:
    :return: Best complete parameters settings, best distance
    """
    gt_snakes = gt_mask_to_snakes(gt_mask)
    best_complete_params, _, best_score = pf_rank.run(input_image, gt_snakes, initial_params=parameters)
    return best_complete_params, best_score

def test_rank_pf(image_path, mask_path, precision, avg_cell_diameter, method):
    frame = try_load_image(image_path)
    gt_mask = np.array(try_load_image(mask_path), dtype=bool)

    gt_snakes = gt_mask_to_snakes(gt_mask)

    pf_rank.run(frame, gt_snakes, precision, avg_cell_diameter, method)


if __name__ == "__main__":
    test_rank_pf(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5])