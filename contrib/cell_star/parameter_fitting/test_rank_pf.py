__author__ = 'Adam'

import sys
import os
import os.path as path
import numpy as np
import scipy as sp
from contrib.cell_star.core.seed import Seed
from contrib.cell_star.test import test_utils
from contrib.cell_star.parameter_fitting.pf_process import run
from contrib.cell_star.parameter_fitting.pf_snake import GTSnake
import contrib.cell_star.parameter_fitting.pf_rank_process as pf_rank
from contrib.cell_star.parameter_fitting.test_pf import try_load_image, gt_mask_to_snakes

corpus_path = "yeast_corpus/data/"

def test_rank_pf(image_path, mask_path, precision, avg_cell_diameter, method):
    frame = try_load_image(image_path)
    gt_mask = np.array(try_load_image(mask_path), dtype=bool)

    gt_snakes = gt_mask_to_snakes(gt_mask)

    pf_rank.run(frame, gt_snakes, precision, avg_cell_diameter, method)


if __name__ == "__main__":
    test_rank_pf(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5])