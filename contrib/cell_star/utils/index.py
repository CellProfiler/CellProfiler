# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import numpy as np


class Index(object):

    @staticmethod
    def _decode_single_index(single_index, dim):
        px, py = divmod(single_index - 1, dim)
        return np.array([int(py), int(px)])

    @staticmethod
    def to_numpy(index):
        if len(index.shape) == 2:
            return index[:, 0], index[:, 1]
        elif len(index.shape) == 3:
            return index[:, :, 0], index[:, :, 1]
        else:
            return index



