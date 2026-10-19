from __future__ import absolute_import
import numpy as np
import unittest

import centrosome.index as I


class TestIndexes(unittest.TestCase):
    def test_00_00_oh_so_empty(self):
        ind = I.Indexes([[]])
        self.assertEqual(ind.length, 0)
        self.assertEqual(len(ind.fwd_idx), 0)
        self.assertEqual(len(ind.rev_idx), 0)
        self.assertEqual(tuple(ind.idx.shape), (1, 0))
        np.testing.assert_array_equal([[]], ind.counts)

    def test_00_01_all_are_empty(self):
        counts = [[0]]
        ind = I.Indexes(counts)
        self.assertEqual(ind.length, 0)
        self.assertEqual(len(ind.fwd_idx), 1)
        self.assertEqual(ind.fwd_idx[0], 0)
        self.assertEqual(len(ind.rev_idx), 0)
        self.assertEqual(tuple(ind.idx.shape), (1, 0))
        np.testing.assert_array_equal(counts, ind.counts)

    def test_00_02_other_ways_to_be_empty(self):
        ind = I.Indexes([(0, 1), (1, 0)])
        self.assertEqual(ind.length, 0)
        self.assertEqual(len(ind.fwd_idx), 2)
        self.assertTrue(np.all(ind.fwd_idx == 0))
        self.assertEqual(len(ind.rev_idx), 0)
        self.assertEqual(tuple(ind.idx.shape), (2, 0))

    def test_01_01_one_object_1_subobject(self):
        ind = I.Indexes([[1]])
        self.assertEqual(ind.length, 1)
        self.assertEqual(len(ind.fwd_idx), 1)
        self.assertEqual(ind.fwd_idx[0], 0)
        self.assertEqual(len(ind.rev_idx), 1)
        self.assertEqual(ind.rev_idx, 0)
        self.assertEqual(tuple(ind.idx.shape), (1, 1))
        self.assertEqual(ind.idx[0, 0], 0)

    def test_01_02_one_object_1x1_subobject(self):
        ind = I.Indexes([[1], [1]])
        self.assertEqual(ind.length, 1)
        self.assertEqual(len(ind.fwd_idx), 1)
        self.assertEqual(ind.fwd_idx[0], 0)
        self.assertEqual(len(ind.rev_idx), 1)
        self.assertEqual(ind.rev_idx, 0)
        self.assertEqual(tuple(ind.idx.shape), (2, 1))
        self.assertEqual(ind.idx[0, 0], 0)
        self.assertEqual(ind.idx[1, 0], 0)

    def test_01_03_one_object_NxM(self):
        counts = np.array([[4], [3]])
        hits = np.zeros(counts[:, 0], int)
        ind = I.Indexes(counts)
        self.assertEqual(ind.length, np.prod(counts))
        self.assertEqual(len(ind.fwd_idx), 1)
        self.assertEqual(ind.fwd_idx[0], 0)
        self.assertEqual(len(ind.rev_idx), ind.length)
        np.testing.assert_array_equal(ind.rev_idx, 0)
        self.assertEqual(tuple(ind.idx.shape), (2, ind.length))
        hits[ind.idx[0], ind.idx[1]] = np.arange(ind.length) + 1
        self.assertEqual(len(np.unique(hits.ravel())), ind.length)

    def test_02_01_two_objects_NxM(self):
        counts = [[4, 2], [3, 6]]
        c0 = counts[0][0] * counts[1][0]
        ind = I.Indexes(counts)
        self.assertEqual(ind.length, np.sum(np.prod(counts, 0)))
        self.assertEqual(len(ind.fwd_idx), 2)
        self.assertEqual(ind.fwd_idx[0], 0)
        self.assertEqual(ind.fwd_idx[1], c0)
        self.assertEqual(len(ind.rev_idx), ind.length)
        np.testing.assert_array_equal(ind.rev_idx[:c0], 0)
        np.testing.assert_array_equal(ind.rev_idx[c0:], 1)
        start = 0
        for i, count in enumerate(ind.counts.transpose()):
            hits = np.zeros(count)
            n = np.prod(count)
            hits[
                ind.idx[0, start : (start + n)], ind.idx[1, start : (start + n)]
            ] = np.arange(np.prod(count))
            self.assertEqual(len(np.unique(hits.ravel())), n)
            start += n

    def test_02_02_multiple_objects_and_one_is_0x0(self):
        counts = np.array([[4, 2, 0, 3], [3, 6, 4, 5]])
        ind = I.Indexes(counts)
        self.assertEqual(ind.length, np.sum(np.prod(counts, 0)))
        self.assertEqual(len(ind.fwd_idx), counts.shape[1])
        np.testing.assert_array_equal(ind.fwd_idx, [0, 12, 24, 24])
        self.assertEqual(len(ind.rev_idx), ind.length)
        start = 0
        for i, count in enumerate(ind.counts.transpose()):
            if np.prod(count) == 0:
                continue
            hits = np.zeros(count)
            n = np.prod(count)
            hits[
                ind.idx[0, start : (start + n)], ind.idx[1, start : (start + n)]
            ] = np.arange(np.prod(count))
            self.assertEqual(len(np.unique(hits.ravel())), n)
            start += n

    def test_02_03_one_at_end(self):
        ind = I.Indexes(np.array([[0, 0, 1]]))
        pass

    def test_02_04_none_at_start(self):
        ind = I.Indexes(np.array([[0, 1, 1]]))
        np.testing.assert_array_equal(ind.rev_idx, np.array([1, 2]))

    def test_02_05_none_at_start_and_one(self):
        ind = I.Indexes(np.array([[0, 2]]))
        np.testing.assert_array_equal(ind.rev_idx, np.array([1, 1]))
