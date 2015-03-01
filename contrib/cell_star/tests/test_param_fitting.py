__author__ = 'Adam Kaczmarek, Filip Mroz'

import logging
import sys

import contrib.cell_star.parameter_fitting.test_pf as test_pf
from contrib.cell_star.parameter_fitting.test_rank_pf import test_rank_pf
from contrib.cell_star.process.segmentation import Segmentation

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print "Usage: <script> base_path image_path mask_path precision avg_cell_diameter method {image_result_path}"
        print "Given: " + " ".join(sys.argv)
        sys.exit(-1)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger = logging.getLogger('contrib.cell_star.parameter_fitting')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    image_result_path = None
    if len(sys.argv) >= 8:
        image_result_path = sys.argv[7]

    test_pf.corpus_path = sys.argv[1]
    image_path = sys.argv[2]
    mask_path = sys.argv[3]
    precision = int(sys.argv[4])
    avg_cell_diameter = float(sys.argv[5])

    full_params_contour, _, _ = test_pf.test_pf(image_path, mask_path, precision, avg_cell_diameter, sys.argv[6])
    complete_params, _, _ = test_rank_pf(image_path, mask_path, precision, avg_cell_diameter, sys.argv[6], initial_params=full_params_contour)
    print "Best_params:", complete_params
    print
    print "CellProfiler autoparams:", Segmentation.encode_auto_params_from_all_params(complete_params)

    test_pf.test_parameters(image_path, mask_path, precision, avg_cell_diameter, complete_params, image_result_path)