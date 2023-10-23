import base64
import bz2
import io

import centrosome.outline
import numpy
import numpy.testing
import numpy.testing
import pytest
import skimage.measure

import cellprofiler_library.functions.segmentation
import cellprofiler_core.image
import cellprofiler_core.image
import cellprofiler_core.object
import cellprofiler_core.object
import cellprofiler_core.utilities.core.object
import cellprofiler_core.utilities.hdf5_dict
import cellprofiler_core.utilities.hdf5_dict


@pytest.fixture
def image10():
    image = numpy.zeros((10, 10), dtype=bool)

    image[2:4, 2:4] = 1
    image[5:7, 5:7] = 1

    return image


@pytest.fixture
def unedited_segmented10(image10):
    return skimage.measure.label(image10)


@pytest.fixture
def segmented10(unedited_segmented10):
    __segmented10 = unedited_segmented10

    __segmented10[__segmented10 == 2] = 0

    return __segmented10


@pytest.fixture
def small_removed_segmented10(segmented10, unedited_segmented10):
    __small_removed_segmented10 = unedited_segmented10

    __small_removed_segmented10[segmented10 == 1] = 0

    return __small_removed_segmented10


def relate_ijv(parent_ijv, children_ijv):
    p = cellprofiler_core.object.Objects()
    p.ijv = parent_ijv
    c = cellprofiler_core.object.Objects()
    c.ijv = children_ijv
    return p.relate_children(c)


class TestDownsampleLabels:
    def test_01_01_downsample_127(self):
        i, j = numpy.mgrid[0:16, 0:8]
        labels = (i * 8 + j).astype(int)
        result = cellprofiler_library.functions.segmentation.downsample_labels(labels)
        assert result.dtype == numpy.dtype(numpy.int8)
        assert numpy.all(result == labels)

    def test_01_02_downsample_128(self):
        i, j = numpy.mgrid[0:16, 0:8]
        labels = (i * 8 + j).astype(int) + 1
        result = cellprofiler_library.functions.segmentation.downsample_labels(labels)
        assert result.dtype == numpy.dtype(numpy.int16)
        assert numpy.all(result == labels)

    def test_01_03_downsample_32767(self):
        i, j = numpy.mgrid[0:256, 0:128]
        labels = (i * 128 + j).astype(int)
        result = cellprofiler_library.functions.segmentation.downsample_labels(labels)
        assert result.dtype == numpy.dtype(numpy.int16)
        assert numpy.all(result == labels)

    def test_01_04_downsample_32768(self):
        i, j = numpy.mgrid[0:256, 0:128]
        labels = (i * 128 + j).astype(int) + 1
        result = cellprofiler_library.functions.segmentation.downsample_labels(labels)
        assert result.dtype == numpy.dtype(numpy.int32)
        assert numpy.all(result == labels)


class TestCropLabelsAndImage:
    def test_01_01_crop_same(self):
        labels, image = cellprofiler_core.utilities.core.object.crop_labels_and_image(
            numpy.zeros((10, 20)), numpy.zeros((10, 20))
        )
        assert tuple(labels.shape) == (10, 20)
        assert tuple(image.shape) == (10, 20)

    def test_01_02_crop_image(self):
        labels, image = cellprofiler_core.utilities.core.object.crop_labels_and_image(
            numpy.zeros((10, 20)), numpy.zeros((10, 30))
        )
        assert tuple(labels.shape) == (10, 20)
        assert tuple(image.shape) == (10, 20)
        labels, image = cellprofiler_core.utilities.core.object.crop_labels_and_image(
            numpy.zeros((10, 20)), numpy.zeros((20, 20))
        )
        assert tuple(labels.shape) == (10, 20)
        assert tuple(image.shape) == (10, 20)

    def test_01_03_crop_labels(self):
        labels, image = cellprofiler_core.utilities.core.object.crop_labels_and_image(
            numpy.zeros((10, 30)), numpy.zeros((10, 20))
        )
        assert tuple(labels.shape) == (10, 20)
        assert tuple(image.shape) == (10, 20)
        labels, image = cellprofiler_core.utilities.core.object.crop_labels_and_image(
            numpy.zeros((20, 20)), numpy.zeros((10, 20))
        )
        assert tuple(labels.shape) == (10, 20)
        assert tuple(image.shape) == (10, 20)

    def test_01_04_crop_both(self):
        labels, image = cellprofiler_core.utilities.core.object.crop_labels_and_image(
            numpy.zeros((10, 30)), numpy.zeros((20, 20))
        )
        assert tuple(labels.shape) == (10, 20)
        assert tuple(image.shape) == (10, 20)

    def test_relate_children_volume(self):
        parent_labels = numpy.zeros((30, 30, 30), dtype=numpy.uint8)

        k, i, j = numpy.mgrid[-15:15, -15:15, -15:15]
        parent_labels[k ** 2 + i ** 2 + j ** 2 <= 196] = 1

        parent_object = cellprofiler_core.object.Objects()

        parent_object.segmented = parent_labels

        labels = numpy.zeros((30, 30, 30), dtype=numpy.uint8)

        k, i, j = numpy.mgrid[-15:15, -15:15, -7:23]
        labels[k ** 2 + i ** 2 + j ** 2 <= 25] = 1

        k, i, j = numpy.mgrid[-15:15, -15:15, -22:8]
        labels[k ** 2 + i ** 2 + j ** 2 <= 16] = 2

        labels[
            0, 10:20, 10:20
        ] = 3  # not touching a parent, should not be counted as a child

        object = cellprofiler_core.object.Objects()

        object.segmented = labels

        actual_children, actual_parents = parent_object.relate_children(object)

        expected_children = [2]

        expected_parents = [1, 1, 0]

        numpy.testing.assert_array_equal(actual_children, expected_children)

        numpy.testing.assert_array_equal(actual_parents, expected_parents)

    # https://github.com/CellProfiler/CellProfiler/issues/2751
    def test_overlay_objects(self):
        data = numpy.zeros((9, 9, 9))

        labels = numpy.zeros_like(data, dtype=numpy.uint8)
        labels[:3, :3, :3] = 1
        labels[:, 3:-3, 3:-3] = 2
        labels[-3:, -3:, -3:] = 3

        overlay_pixel_data = cellprofiler_core.utilities.core.object.overlay_labels(
            pixel_data=data, labels=labels
        )

        overlay_region_1 = overlay_pixel_data[:3, :3, :3]
        assert numpy.all(overlay_region_1 == overlay_region_1[0, 0, 0])

        overlay_region_2 = overlay_pixel_data[:, 3:-3, 3:-3]
        assert numpy.all(overlay_region_2 == overlay_region_2[0, 0, 0])

        overlay_region_3 = overlay_pixel_data[-3:, -3:, -3:]
        assert numpy.all(overlay_region_3 == overlay_region_3[0, 0, 0])

        assert not numpy.all(overlay_region_1[0, 0, 0] == overlay_region_2[0, 0, 0])
        assert not numpy.all(overlay_region_1[0, 0, 0] == overlay_region_3[0, 0, 0])
        assert not numpy.all(overlay_region_2[0, 0, 0] == overlay_region_3[0, 0, 0])

    # https://github.com/CellProfiler/CellProfiler/issues/3268
    def test_overlay_objects_empty_label(self):
        data = numpy.zeros((9, 9, 9))

        labels = numpy.zeros_like(data, dtype=numpy.uint8)
        labels[:3, :3, :3] = 1
        labels[-3:, -3:, -3:] = 3

        overlay_pixel_data = cellprofiler_core.utilities.core.object.overlay_labels(
            pixel_data=data, labels=labels
        )

        overlay_region_1 = overlay_pixel_data[:3, :3, :3]
        assert numpy.all(overlay_region_1 == overlay_region_1[0, 0, 0])

        overlay_region_3 = overlay_pixel_data[-3:, -3:, -3:]
        assert numpy.all(overlay_region_3 == overlay_region_3[0, 0, 0])

        assert not numpy.all(overlay_region_1[0, 0, 0] == overlay_region_3[0, 0, 0])


class TestSizeSimilarly:
    def test_01_01_size_same(self):
        secondary, mask = cellprofiler_core.utilities.core.object.size_similarly(
            numpy.zeros((10, 20)), numpy.zeros((10, 20))
        )
        assert tuple(secondary.shape) == (10, 20)
        assert numpy.all(mask)

    def test_01_02_larger_secondary(self):
        secondary, mask = cellprofiler_core.utilities.core.object.size_similarly(
            numpy.zeros((10, 20)), numpy.zeros((10, 30))
        )
        assert tuple(secondary.shape) == (10, 20)
        assert numpy.all(mask)
        secondary, mask = cellprofiler_core.utilities.core.object.size_similarly(
            numpy.zeros((10, 20)), numpy.zeros((20, 20))
        )
        assert tuple(secondary.shape) == (10, 20)
        assert numpy.all(mask)

    def test_01_03_smaller_secondary(self):
        secondary, mask = cellprofiler_core.utilities.core.object.size_similarly(
            numpy.zeros((10, 20), int), numpy.zeros((10, 15), numpy.float32)
        )
        assert tuple(secondary.shape) == (10, 20)
        assert numpy.all(mask[:10, :15])
        assert numpy.all(~mask[:10, 15:])
        assert secondary.dtype == numpy.dtype(numpy.float32)

    def test_01_04_size_color(self):
        secondary, mask = cellprofiler_core.utilities.core.object.size_similarly(
            numpy.zeros((10, 20), int), numpy.zeros((10, 15, 3), numpy.float32)
        )
        assert tuple(secondary.shape) == (10, 20, 3)
        assert numpy.all(mask[:10, :15])
        assert numpy.all(~mask[:10, 15:])
        assert secondary.dtype == numpy.dtype(numpy.float32)


class TestObjects:
    def test_01_01_set_segmented(self, segmented10):
        objects = cellprofiler_core.object.Objects()

        segmented = segmented10

        objects.segmented = segmented

        numpy.testing.assert_array_equal(segmented, objects.segmented)

    def test_01_02_segmented(self, segmented10):
        objects = cellprofiler_core.object.Objects()

        segmented = segmented10

        objects.segmented = segmented

        numpy.testing.assert_array_equal(segmented, objects.segmented)

    def test_segmented_volume(self):
        segmentation = numpy.zeros((3, 10, 10), dtype=numpy.uint8)

        segmentation[0:2, 2:4, 2:4] = 1
        segmentation[1:2, 5:7, 5:7] = 2

        x = cellprofiler_core.object.Objects()

        x.segmented = segmentation

        numpy.testing.assert_array_equal(x.segmented, segmentation)

    def test_01_03_set_unedited_segmented(self, unedited_segmented10):
        x = cellprofiler_core.object.Objects()

        x.unedited_segmented = unedited_segmented10

        assert (unedited_segmented10 == x.unedited_segmented).all()

    def test_01_04_unedited_segmented(self, unedited_segmented10):
        x = cellprofiler_core.object.Objects()

        x.unedited_segmented = unedited_segmented10

        assert (unedited_segmented10 == x.unedited_segmented).all()

    def test_unedited_segmented_volume(self):
        segmentation = numpy.zeros((3, 10, 10), dtype=numpy.uint8)

        segmentation[0:2, 2:4, 2:4] = 1

        segmentation[1:2, 5:7, 5:7] = 2

        x = cellprofiler_core.object.Objects()

        x.unedited_segmented = segmentation

        assert numpy.all(x.unedited_segmented == segmentation)

    def test_01_05_set_small_removed_segmented(self, small_removed_segmented10):
        x = cellprofiler_core.object.Objects()

        x.small_removed_segmented = small_removed_segmented10

        assert (small_removed_segmented10 == x.small_removed_segmented).all()

    def test_small_removed_segmented_volume(self):
        segmentation = numpy.zeros((3, 10, 10), dtype=numpy.uint8)

        segmentation[0:2, 2:4, 2:4] = 1

        segmentation[1:2, 5:7, 5:7] = 2

        x = cellprofiler_core.object.Objects()

        x.small_removed_segmented = segmentation

        assert numpy.all(x.small_removed_segmented == segmentation)

    def test_01_06_unedited_segmented(self, small_removed_segmented10):
        x = cellprofiler_core.object.Objects()

        x.small_removed_segmented = small_removed_segmented10

        assert (small_removed_segmented10 == x.small_removed_segmented).all()

    def test_02_01_set_all(
        self, segmented10, unedited_segmented10, small_removed_segmented10
    ):
        x = cellprofiler_core.object.Objects()

        x.segmented = segmented10

        x.unedited_segmented = unedited_segmented10

        x.small_removed_segmented = small_removed_segmented10

    # def test_03_01_default_unedited_segmented(self):
    #     x = cpo.Objects()
    #     x.segmented = __segmented10
    #     assertTrue((x.unedited_segmented==x.segmented).all())

    def test_03_02_default_small_removed_segmented(
        self, segmented10, unedited_segmented10
    ):
        x = cellprofiler_core.object.Objects()

        x.segmented = segmented10

        assert (x.small_removed_segmented == segmented10).all()

        x.unedited_segmented = unedited_segmented10

        assert (x.small_removed_segmented == unedited_segmented10).all()

    def test_shape_image_segmentation(self, segmented10):
        x = cellprofiler_core.object.Objects()

        x.segmented = segmented10

        assert x.shape == (10, 10)

    def test_shape_volume_segmentation(self):
        x = cellprofiler_core.object.Objects()

        x.segmented = numpy.ones((5, 10, 10))

        assert x.shape == (5, 10, 10)

    def test_get_labels_image_segmentation(self, segmented10):
        x = cellprofiler_core.object.Objects()

        x.segmented = segmented10

        [(labels, _)] = x.get_labels()

        assert numpy.all(labels == segmented10)

    def test_get_labels_volume_segmentation(self):
        x = cellprofiler_core.object.Objects()

        segmentation = numpy.ones((5, 10, 10))

        x.segmented = segmentation

        [(labels, _)] = x.get_labels()

        assert segmentation.shape == labels.shape

        assert numpy.all(segmentation == labels)

    def test_05_01_relate_zero_parents_and_children(self):
        """Test the relate method if both parent and child label matrices are zeros"""
        x = cellprofiler_core.object.Objects()
        x.segmented = numpy.zeros((10, 10), int)
        y = cellprofiler_core.object.Objects()
        y.segmented = numpy.zeros((10, 10), int)
        children_per_parent, parents_of_children = x.relate_children(y)
        assert numpy.product(children_per_parent.shape) == 0
        assert numpy.product(parents_of_children.shape) == 0

    def test_05_02_relate_zero_parents_one_child(self):
        x = cellprofiler_core.object.Objects()
        x.segmented = numpy.zeros((10, 10), int)
        y = cellprofiler_core.object.Objects()
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        y.segmented = labels
        children_per_parent, parents_of_children = x.relate_children(y)
        assert numpy.product(children_per_parent.shape) == 0
        assert numpy.product(parents_of_children.shape) == 1
        assert parents_of_children[0] == 0

    def test_05_03_relate_one_parent_no_children(self):
        x = cellprofiler_core.object.Objects()
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        x.segmented = labels
        y = cellprofiler_core.object.Objects()
        y.segmented = numpy.zeros((10, 10), int)
        children_per_parent, parents_of_children = x.relate_children(y)
        assert numpy.product(children_per_parent.shape) == 1
        assert children_per_parent[0] == 0
        assert numpy.product(parents_of_children.shape) == 0

    def test_05_04_relate_one_parent_one_child(self):
        x = cellprofiler_core.object.Objects()
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        x.segmented = labels
        y = cellprofiler_core.object.Objects()
        y.segmented = labels
        children_per_parent, parents_of_children = x.relate_children(y)
        assert numpy.product(children_per_parent.shape) == 1
        assert children_per_parent[0] == 1
        assert numpy.product(parents_of_children.shape) == 1
        assert parents_of_children[0] == 1

    def test_05_05_relate_two_parents_one_child(self):
        x = cellprofiler_core.object.Objects()
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        labels[3:6, 7:9] = 2
        x.segmented = labels
        y = cellprofiler_core.object.Objects()
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 5:9] = 1
        y.segmented = labels
        children_per_parent, parents_of_children = x.relate_children(y)
        assert numpy.product(children_per_parent.shape) == 2
        assert children_per_parent[0] == 0
        assert children_per_parent[1] == 1
        assert numpy.product(parents_of_children.shape) == 1
        assert parents_of_children[0] == 2

    def test_05_06_relate_one_parent_two_children(self):
        x = cellprofiler_core.object.Objects()
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 3:9] = 1
        x.segmented = labels
        y = cellprofiler_core.object.Objects()
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        labels[3:6, 7:9] = 2
        y.segmented = labels
        children_per_parent, parents_of_children = x.relate_children(y)
        assert numpy.product(children_per_parent.shape) == 1
        assert children_per_parent[0] == 2
        assert numpy.product(parents_of_children.shape) == 2
        assert parents_of_children[0] == 1
        assert parents_of_children[1] == 1

    def test_05_07_relate_ijv_none(self):
        child_counts, parents_of = relate_ijv(
            numpy.zeros((0, 3), int), numpy.zeros((0, 3), int)
        )
        assert len(child_counts) == 0
        assert len(parents_of) == 0

        child_counts, parents_of = relate_ijv(
            numpy.zeros((0, 3), int), numpy.array([[1, 2, 3]])
        )
        assert len(child_counts) == 0
        assert len(parents_of) == 3
        assert parents_of[2] == 0

        child_counts, parents_of = relate_ijv(
            numpy.array([[1, 2, 3]]), numpy.zeros((0, 3), int)
        )
        assert len(child_counts) == 3
        assert child_counts[2] == 0
        assert len(parents_of) == 0

    def test_05_08_relate_ijv_no_match(self):
        child_counts, parents_of = relate_ijv(
            numpy.array([[3, 2, 1]]), numpy.array([[5, 6, 1]])
        )
        assert len(child_counts) == 1
        assert child_counts[0] == 0
        assert len(parents_of) == 1
        assert parents_of[0] == 0

    def test_05_09_relate_ijv_one_match(self):
        child_counts, parents_of = relate_ijv(
            numpy.array([[3, 2, 1]]), numpy.array([[3, 2, 1]])
        )
        assert len(child_counts) == 1
        assert child_counts[0] == 1
        assert len(parents_of) == 1
        assert parents_of[0] == 1

    def test_05_10_relate_ijv_many_points_one_match(self):
        r = numpy.random.RandomState()
        r.seed(510)
        parent_ijv = numpy.column_stack(
            (r.randint(0, 10, size=(100, 2)), numpy.ones(100, int))
        )
        child_ijv = numpy.column_stack(
            (r.randint(0, 10, size=(100, 2)), numpy.ones(100, int))
        )
        child_counts, parents_of = relate_ijv(parent_ijv, child_ijv)
        assert len(child_counts) == 1
        assert child_counts[0] == 1
        assert len(parents_of) == 1
        assert parents_of[0] == 1

    def test_05_11_relate_many_many(self):
        r = numpy.random.RandomState()
        r.seed(511)
        parent_ijv = numpy.column_stack(
            (r.randint(0, 10, size=(100, 2)), numpy.ones(100, int))
        )
        child_ijv = numpy.column_stack(
            (r.randint(0, 10, size=(100, 2)), numpy.ones(100, int))
        )
        parent_ijv[parent_ijv[:, 0] >= 5, 2] = 2
        child_ijv[:, 2] = (
            1
            + (child_ijv[:, 0] >= 5).astype(int)
            + 2 * (child_ijv[:, 1] >= 5).astype(int)
        )
        child_counts, parents_of = relate_ijv(parent_ijv, child_ijv)
        assert len(child_counts) == 2
        assert tuple(child_counts) == (2, 2)
        assert len(parents_of) == 4
        assert parents_of[0] == 1
        assert parents_of[1] == 2
        assert parents_of[2] == 1
        assert parents_of[3] == 2

    def test_05_12_relate_many_parent_missing_child(self):
        parent_ijv = numpy.array([[1, 0, 1], [2, 0, 2], [3, 0, 3]])
        child_ijv = numpy.array([[1, 0, 1], [3, 0, 2]])
        child_counts, parents_of = relate_ijv(parent_ijv, child_ijv)
        assert len(child_counts) == 3
        assert tuple(child_counts) == (1, 0, 1)
        assert len(parents_of) == 2
        assert parents_of[0] == 1
        assert parents_of[1] == 3

    def test_05_13_relate_many_child_missing_parent(self):
        child_ijv = numpy.array([[1, 0, 1], [2, 0, 2], [3, 0, 3]])
        parent_ijv = numpy.array([[1, 0, 1], [3, 0, 2]])
        child_counts, parents_of = relate_ijv(parent_ijv, child_ijv)
        assert len(child_counts) == 2
        assert tuple(child_counts) == (1, 1)
        assert len(parents_of) == 3
        assert parents_of[0] == 1
        assert parents_of[1] == 0
        assert parents_of[2] == 2

    def test_05_14_relate_many_parent_missing_child_end(self):
        parent_ijv = numpy.array([[1, 0, 1], [2, 0, 2], [3, 0, 3]])
        child_ijv = numpy.array([[1, 0, 1], [2, 0, 2]])
        child_counts, parents_of = relate_ijv(parent_ijv, child_ijv)
        assert len(child_counts) == 3
        assert tuple(child_counts) == (1, 1, 0)
        assert len(parents_of) == 2
        assert parents_of[0] == 1
        assert parents_of[1] == 2

    def test_05_15_relate_many_child_missing_end(self):
        child_ijv = numpy.array([[1, 0, 1], [2, 0, 2], [3, 0, 3]])
        parent_ijv = numpy.array([[1, 0, 1], [2, 0, 2]])
        child_counts, parents_of = relate_ijv(parent_ijv, child_ijv)
        assert len(child_counts) == 2
        assert tuple(child_counts) == (1, 1)
        assert len(parents_of) == 3
        assert parents_of[0] == 1
        assert parents_of[1] == 2
        assert parents_of[2] == 0

    def test_05_16_relate_uint16(self):
        # Regression test of issue 1285 - uint16 ijv values
        # wrap-around when flattened
        #
        # 4096 * 16 = 0 in uint16 arithmetic
        child_ijv = numpy.array([[4095, 0, 1]], numpy.uint16)
        parent_ijv = numpy.array([[4095, 16, 1]], numpy.uint16)
        child_counts, parents_of = relate_ijv(parent_ijv, child_ijv)
        assert numpy.all(child_counts == 0)

    def test_06_01_segmented_to_ijv(self):
        """Convert the segmented representation to an IJV one"""
        x = cellprofiler_core.object.Objects()
        numpy.random.seed(61)
        labels = numpy.random.randint(0, 10, size=(20, 20))
        x.segmented = labels
        ijv = x.get_ijv()
        new_labels = numpy.zeros(labels.shape, int)
        new_labels[ijv[:, 0], ijv[:, 1]] = ijv[:, 2]
        assert numpy.all(labels == new_labels)

    def test_06_02_ijv_to_labels_empty(self):
        """Convert a blank ijv representation to labels"""
        x = cellprofiler_core.object.Objects()
        x.ijv = numpy.zeros((0, 3), int)
        y = x.get_labels()
        assert len(y) == 1
        labels, indices = y[0]
        assert len(indices) == 0
        assert numpy.all(labels == 0)

    def test_06_03_ijv_to_labels_simple(self):
        """Convert an ijv representation w/o overlap to labels"""
        x = cellprofiler_core.object.Objects()
        numpy.random.seed(63)
        labels = numpy.zeros((20, 20), int)
        labels[1:-1, 1:-1] = numpy.random.randint(0, 10, size=(18, 18))

        x.segmented = labels
        ijv = x.get_ijv()
        x = cellprofiler_core.object.Objects()
        x.ijv = ijv
        x.parent_image = cellprofiler_core.image.Image(numpy.zeros(labels.shape))
        labels_out = x.get_labels()
        assert len(labels_out) == 1
        labels_out, indices = labels_out[0]
        assert numpy.all(labels_out == labels)
        assert len(indices) == 9
        assert numpy.all(numpy.unique(indices) == numpy.arange(1, 10))

    def test_06_04_ijv_to_labels_overlapping(self):
        """Convert an ijv representation with overlap to labels"""
        ijv = numpy.array(
            [
                [1, 1, 1],
                [1, 2, 1],
                [2, 1, 1],
                [2, 2, 1],
                [1, 3, 2],
                [2, 3, 2],
                [2, 3, 3],
                [4, 4, 4],
                [4, 5, 4],
                [4, 5, 5],
                [5, 5, 5],
            ]
        )
        x = cellprofiler_core.object.Objects()
        x.ijv = ijv
        labels = x.get_labels()
        assert len(labels) == 2
        unique_a = numpy.unique(labels[0][0])[1:]
        unique_b = numpy.unique(labels[1][0])[1:]
        for a in unique_a:
            assert a not in unique_b
        for b in unique_b:
            assert b not in unique_a
        for i, j, v in ijv:
            mylabels = labels[0][0] if v in unique_a else labels[1][0]
            assert mylabels[i, j] == v

    def test_06_05_ijv_three_overlapping(self):
        #
        # This is a regression test of a bug where a segmentation consists
        # of only one point, labeled three times yielding two planes instead
        # of three.
        #
        ijv = numpy.array([[4, 5, 1], [4, 5, 2], [4, 5, 3]])
        x = cellprofiler_core.object.Objects()
        x.set_ijv(ijv, (8, 9))
        labels = []
        indices = numpy.zeros(3, bool)
        for l, i in x.get_labels():
            labels.append(l)
            assert len(i) == 1
            assert i[0] in (1, 2, 3)
            indices[i[0] - 1] = True
        assert numpy.all(indices)
        assert len(labels) == 3
        lstacked = numpy.dstack(labels)
        i, j, k = numpy.mgrid[
            0 : lstacked.shape[0], 0 : lstacked.shape[1], 0 : lstacked.shape[2]
        ]
        assert numpy.all(lstacked[(i != 4) | (j != 5)] == 0)
        assert (1, 2, 3) == tuple(sorted(lstacked[4, 5, :]))

    def test_07_00_make_ivj_outlines_empty(self):
        numpy.random.seed(70)
        x = cellprofiler_core.object.Objects()
        x.segmented = numpy.zeros((10, 20), int)
        image = x.make_ijv_outlines(numpy.random.uniform(size=(5, 3)))
        assert numpy.all(image == 0)

    def test_07_01_make_ijv_outlines(self):
        numpy.random.seed(70)
        x = cellprofiler_core.object.Objects()
        ii, jj = numpy.mgrid[0:10, 0:20]
        masks = [
            (ii - ic) ** 2 + (jj - jc) ** 2 < r ** 2
            for ic, jc, r in ((4, 5, 5), (4, 12, 5), (6, 8, 5))
        ]
        i = numpy.hstack([ii[mask] for mask in masks])
        j = numpy.hstack([jj[mask] for mask in masks])
        v = numpy.hstack([[k + 1] * numpy.sum(mask) for k, mask in enumerate(masks)])

        x.set_ijv(numpy.column_stack((i, j, v)), ii.shape)
        x.parent_image = cellprofiler_core.image.Image(numpy.zeros((10, 20)))
        colors = numpy.random.uniform(size=(3, 3)).astype(numpy.float32)
        image = x.make_ijv_outlines(colors)
        i1 = [i for i, color in enumerate(colors) if numpy.all(color == image[0, 5, :])]
        assert len(i1) == 1
        i2 = [
            i for i, color in enumerate(colors) if numpy.all(color == image[0, 12, :])
        ]
        assert len(i2) == 1
        i3 = [
            i for i, color in enumerate(colors) if numpy.all(color == image[-1, 8, :])
        ]
        assert len(i3) == 1
        assert i1[0] != i2[0]
        assert i2[0] != i3[0]
        colors = colors[numpy.array([i1[0], i2[0], i3[0]])]
        outlines = numpy.zeros((10, 20, 3), numpy.float32)
        alpha = numpy.zeros((10, 20))
        for i, (color, mask) in enumerate(zip(colors, masks)):
            my_outline = centrosome.outline.outline(mask)
            outlines[my_outline] += color
            alpha[my_outline] += 1
        alpha[alpha == 0] = 1
        outlines /= alpha[:, :, numpy.newaxis]
        numpy.testing.assert_almost_equal(outlines, image)

    def test_07_02_labels_same_as_ijv(self):
        d = (
            "QlpoOTFBWSZTWeu0qJwGoDt///////////////9///////9//3///3//f3//f/9////4YCAfH0ki"
            "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"
        )
        stream = io.BytesIO(bz2.decompress(base64.b64decode(d)))
        x = cellprofiler_core.object.Objects()
        x.segmented = numpy.load(stream)
        y = cellprofiler_core.object.Objects()
        y.segmented = numpy.load(stream)
        labels_children_per_parent, labels_parents_of_children = x.relate_children(y)
        # force generation of ijv
        x.ijv, y.ijv
        ijv_children_per_parent, ijv_parents_of_children = x.relate_children(y)
        numpy.testing.assert_array_equal(
            labels_children_per_parent, ijv_children_per_parent
        )
        numpy.testing.assert_array_equal(
            labels_parents_of_children, ijv_parents_of_children
        )

    def test_center_of_mass_with_background_label(self):
        labels = numpy.zeros((11, 11), dtype=numpy.uint8)

        labels[3:8, 3:8] = 1

        objects = cellprofiler_core.object.Objects()

        objects.segmented = labels

        centers = objects.center_of_mass()

        numpy.testing.assert_array_equal(centers, [[5, 5]])

    def test_center_of_mass_without_background_label(self):
        labels = numpy.ones((11, 11), dtype=numpy.uint8)

        objects = cellprofiler_core.object.Objects()

        objects.segmented = labels

        centers = objects.center_of_mass()

        numpy.testing.assert_array_equal(centers, [[5, 5]])
