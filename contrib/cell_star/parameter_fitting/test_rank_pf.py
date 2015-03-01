__author__ = 'Adam'

import sys
import numpy as np
import contrib.cell_star.parameter_fitting.pf_rank_process as pf_rank
import contrib.cell_star.parameter_fitting.test_pf as test_pf
from contrib.cell_star.parameter_fitting.test_pf import try_load_image, image_to_label, cropped_to_gt, gt_label_to_snakes
from cellprofiler.preferences import get_max_workers

import logging

def run_rank_pf(input_image, gt_mask, parameters):
    """
    :param input_image:
    :param gt_mask:
    :param parameters:
    :return: Best complete parameters settings, best distance
    """
    cropped_image, cropped_gt_label = cropped_to_gt(parameters["segmentation"]["avgCellDiameter"], input_image, gt_mask)

    gt_snakes = gt_label_to_snakes(cropped_gt_label)
    if get_max_workers() > 1 and not(getattr(sys, "frozen", False) and sys.platform == 'win32'):
        # multiprocessing do not work if frozen on win32
        best_complete_params, _, best_score = pf_rank.run_multiprocess(cropped_image, gt_snakes, initial_params=parameters, method='brutemaxbasin')
    else:
        best_complete_params, _, best_score = pf_rank.run_singleprocess(cropped_image, gt_snakes, initial_params=parameters, method='brutemaxbasin')

    return best_complete_params, best_score


def test_rank_pf(image_path, mask_path, precision, avg_cell_diameter, method, initial_params=None):
    frame = try_load_image(image_path)
    gt_image = np.array(try_load_image(mask_path) * 255, dtype=int)

    cropped_image, cropped_gt_label = cropped_to_gt(avg_cell_diameter, frame, gt_image)

    gt_snakes = gt_label_to_snakes(cropped_gt_label)
    if method == "mp":
        return pf_rank.run_multiprocess(cropped_image, gt_snakes, precision, avg_cell_diameter, 'brutemaxbasin', initial_params=initial_params)
    else:
        return pf_rank.run_singleprocess(cropped_image, gt_snakes, precision, avg_cell_diameter, method, initial_params=initial_params)


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