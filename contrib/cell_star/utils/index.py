# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import numpy as np


class Index(object):

    @staticmethod
    def _decode_single_index(single_index, dim):
        px, py = divmod(single_index - 1, dim)
        return np.array([int(py), int(px)])

    @staticmethod
    def to_matlab(numpy_index, dim):
        # TODO: check
        return [y * dim + x + 1 for x, y in numpy_index]

    @staticmethod
    def from_matlab_flat(matlab_index, dim):
        x = matlab_index.shape[0]
        decoded_index = np.zeros((x, 2), dtype=np.int32)
        for i, v in enumerate(matlab_index):
            decoded_index[i] = Index._decode_single_index(v, dim)

        return decoded_index

    @staticmethod
    def from_matlab(matlab_index, dim):
        x = matlab_index.shape[0]
        y = matlab_index.shape[1]
        decoded_index = np.zeros((x * y, 2), dtype=np.int32)
        index_matrix = matlab_index.reshape(-1)
        for i, v in enumerate(index_matrix):
            decoded_index[i] = Index._decode_single_index(v, dim)

        return decoded_index.reshape((x, y, 2))

    @staticmethod
    def to_numpy(index):
        if len(index.shape) == 2:
            return index[:, 0], index[:, 1]
        elif len(index.shape) == 3:
            return index[:, :, 0], index[:, :, 1]
        else:
            return index



