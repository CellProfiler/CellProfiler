__author__ = 'Adam Kaczmarek, Filip Mroz'

import logging
import os.path as path
import sys

import contrib.cell_star.parameter_fitting.test_pf as test_pf
from contrib.cell_star.parameter_fitting.test_rank_pf import test_rank_pf
from contrib.cell_star.process.segmentation import Segmentation

logger = logging.getLogger(__name__)

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
    full_params_contour, _, _ = test_pf.test_pf(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), sys.argv[6])
    complete_params, _, _ = test_rank_pf(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), sys.argv[6], initial_params=full_params_contour)
    print "Best_params:", complete_params
    print
    print "CellProfiler autoparams:", Segmentation.encode_auto_params_from_all_params(complete_params)