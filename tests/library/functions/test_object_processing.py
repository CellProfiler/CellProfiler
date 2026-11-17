import numpy as np

from cellprofiler_library.functions.object_processing import split_objects


class TestSplitObjects:
    def test_touching_objects_stay_separate(self):
        labels = np.zeros((6, 8), int)
        labels[1:4, 1:4] = 1
        labels[1:4, 4:7] = 2
        np.testing.assert_array_equal(split_objects(labels), labels)

    def test_disconnected_pieces_get_new_labels(self):
        labels = np.zeros((10, 20), int)
        labels[2:5, 3:8] = 1
        labels[2:5, 13:18] = 1
        out = split_objects(labels)
        expected = np.zeros((10, 20), int)
        expected[2:5, 3:8] = 1
        expected[2:5, 13:18] = 2
        np.testing.assert_array_equal(out, expected)

    def test_touching_objects_and_split_label(self):
        labels = np.zeros((6, 12), int)
        labels[1:4, 1:4] = 1
        labels[1:4, 4:7] = 2
        labels[1:4, 9:11] = 3
        labels[5, 0:2] = 3
        out = split_objects(labels)
        expected = labels.copy()
        expected[5, 0:2] = 4
        np.testing.assert_array_equal(out, expected)

    def test_diagonal_pieces_of_one_label_stay_together(self):
        labels = np.zeros((4, 4), int)
        labels[0, 0] = 1
        labels[1, 1] = 1
        np.testing.assert_array_equal(split_objects(labels), labels)

    def test_empty(self):
        labels = np.zeros((5, 5), int)
        np.testing.assert_array_equal(split_objects(labels), labels)
