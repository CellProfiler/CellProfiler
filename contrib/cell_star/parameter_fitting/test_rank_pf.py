__author__ = 'Adam'

import sys
import numpy as np
import contrib.cell_star.parameter_fitting.pf_rank_process as pf_rank
import contrib.cell_star.parameter_fitting.test_pf as test_pf
from contrib.cell_star.parameter_fitting.test_pf import try_load_image, gt_mask_to_snakes
from cellprofiler.preferences import get_max_workers

import logging

def run_rank_pf(input_image, gt_mask, parameters):
    """
    :param input_image:
    :param gt_mask:
    :param parameters:
    :return: Best complete parameters settings, best distance
    """
    gt_snakes = gt_mask_to_snakes(gt_mask)
    if get_max_workers() > 1:
        best_complete_params, _, best_score = pf_rank.run_multiprocess(input_image, gt_snakes, initial_params=parameters, method='brute')
    else:
        best_complete_params, _, best_score = pf_rank.run_singleprocess(input_image, gt_snakes, initial_params=parameters, method='brute')

    return best_complete_params, best_score

def test_rank_pf(image_path, mask_path, precision, avg_cell_diameter, method):
    frame = try_load_image(image_path)
    gt_mask = np.array(try_load_image(mask_path), dtype=bool)

    gt_snakes = gt_mask_to_snakes(gt_mask)

    if method == "mp":
        pf_rank.run_multiprocess(frame, gt_snakes, precision, avg_cell_diameter, 'brute')
    else:
        pf_rank.run_singleprocess(frame, gt_snakes, precision, avg_cell_diameter, method)


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print "Usage: <script> base_path image_path mask_path precision avg_cell_diameter method"
        print "Given: " + " ".join(sys.argv)
        sys.exit(-1)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger = logging.getLogger('contrib.cell_star.parameter_fitting')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    test_pf.corpus_path = sys.argv[1]
    test_rank_pf(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), sys.argv[6])