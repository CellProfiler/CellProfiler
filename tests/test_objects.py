import numpy
import numpy.testing

import nucleus.image
import nucleus.object
import nucleus.utilities.hdf5_dict


class TestDownsampleLabels:
    def test_01_01_downsample_127(self):
        i, j = numpy.mgrid[0:16, 0:8]
        labels = (i * 8 + j).astype(int)
        result = nucleus.object.downsample_labels(labels)
        assert result.dtype == numpy.dtype(numpy.int8)
        assert numpy.all(result == labels)

    def test_01_02_downsample_128(self):
        i, j = numpy.mgrid[0:16, 0:8]
        labels = (i * 8 + j).astype(int) + 1
        result = nucleus.object.downsample_labels(labels)
        assert result.dtype == numpy.dtype(numpy.int16)
        assert numpy.all(result == labels)

    def test_01_03_downsample_32767(self):
        i, j = numpy.mgrid[0:256, 0:128]
        labels = (i * 128 + j).astype(int)
        result = nucleus.object.downsample_labels(labels)
        assert result.dtype == numpy.dtype(numpy.int16)
        assert numpy.all(result == labels)

    def test_01_04_downsample_32768(self):
        i, j = numpy.mgrid[0:256, 0:128]
        labels = (i * 128 + j).astype(int) + 1
        result = nucleus.object.downsample_labels(labels)
        assert result.dtype == numpy.dtype(numpy.int32)
        assert numpy.all(result == labels)


class TestCropLabelsAndImage:
    def test_01_01_crop_same(self):
        labels, image = nucleus.object.crop_labels_and_image(
            numpy.zeros((10, 20)), numpy.zeros((10, 20))
        )
        assert tuple(labels.shape) == (10, 20)
        assert tuple(image.shape) == (10, 20)

    def test_01_02_crop_image(self):
        labels, image = nucleus.object.crop_labels_and_image(
            numpy.zeros((10, 20)), numpy.zeros((10, 30))
        )
        assert tuple(labels.shape) == (10, 20)
        assert tuple(image.shape) == (10, 20)
        labels, image = nucleus.object.crop_labels_and_image(
            numpy.zeros((10, 20)), numpy.zeros((20, 20))
        )
        assert tuple(labels.shape) == (10, 20)
        assert tuple(image.shape) == (10, 20)

    def test_01_03_crop_labels(self):
        labels, image = nucleus.object.crop_labels_and_image(
            numpy.zeros((10, 30)), numpy.zeros((10, 20))
        )
        assert tuple(labels.shape) == (10, 20)
        assert tuple(image.shape) == (10, 20)
        labels, image = nucleus.object.crop_labels_and_image(
            numpy.zeros((20, 20)), numpy.zeros((10, 20))
        )
        assert tuple(labels.shape) == (10, 20)
        assert tuple(image.shape) == (10, 20)

    def test_01_04_crop_both(self):
        labels, image = nucleus.object.crop_labels_and_image(
            numpy.zeros((10, 30)), numpy.zeros((20, 20))
        )
        assert tuple(labels.shape) == (10, 20)
        assert tuple(image.shape) == (10, 20)

    def test_relate_children_volume(self):
        parent_labels = numpy.zeros((30, 30, 30), dtype=numpy.uint8)

        k, i, j = numpy.mgrid[-15:15, -15:15, -15:15]
        parent_labels[k ** 2 + i ** 2 + j ** 2 <= 196] = 1

        parent_object = nucleus.object.Objects()

        parent_object.segmented = parent_labels

        labels = numpy.zeros((30, 30, 30), dtype=numpy.uint8)

        k, i, j = numpy.mgrid[-15:15, -15:15, -7:23]
        labels[k ** 2 + i ** 2 + j ** 2 <= 25] = 1

        k, i, j = numpy.mgrid[-15:15, -15:15, -22:8]
        labels[k ** 2 + i ** 2 + j ** 2 <= 16] = 2

        labels[
            0, 10:20, 10:20
        ] = 3  # not touching a parent, should not be counted as a child

        object = nucleus.object.Objects()

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

        overlay_pixel_data = nucleus.object.overlay_labels(
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

        overlay_pixel_data = nucleus.object.overlay_labels(
            pixel_data=data, labels=labels
        )

        overlay_region_1 = overlay_pixel_data[:3, :3, :3]
        assert numpy.all(overlay_region_1 == overlay_region_1[0, 0, 0])

        overlay_region_3 = overlay_pixel_data[-3:, -3:, -3:]
        assert numpy.all(overlay_region_3 == overlay_region_3[0, 0, 0])

        assert not numpy.all(overlay_region_1[0, 0, 0] == overlay_region_3[0, 0, 0])


class TestSizeSimilarly:
    def test_01_01_size_same(self):
        secondary, mask = nucleus.object.size_similarly(
            numpy.zeros((10, 20)), numpy.zeros((10, 20))
        )
        assert tuple(secondary.shape) == (10, 20)
        assert numpy.all(mask)

    def test_01_02_larger_secondary(self):
        secondary, mask = nucleus.object.size_similarly(
            numpy.zeros((10, 20)), numpy.zeros((10, 30))
        )
        assert tuple(secondary.shape) == (10, 20)
        assert numpy.all(mask)
        secondary, mask = nucleus.object.size_similarly(
            numpy.zeros((10, 20)), numpy.zeros((20, 20))
        )
        assert tuple(secondary.shape) == (10, 20)
        assert numpy.all(mask)

    def test_01_03_smaller_secondary(self):
        secondary, mask = nucleus.object.size_similarly(
            numpy.zeros((10, 20), int), numpy.zeros((10, 15), numpy.float32)
        )
        assert tuple(secondary.shape) == (10, 20)
        assert numpy.all(mask[:10, :15])
        assert numpy.all(~mask[:10, 15:])
        assert secondary.dtype == numpy.dtype(numpy.float32)

    def test_01_04_size_color(self):
        secondary, mask = nucleus.object.size_similarly(
            numpy.zeros((10, 20), int), numpy.zeros((10, 15, 3), numpy.float32)
        )
        assert tuple(secondary.shape) == (10, 20, 3)
        assert numpy.all(mask[:10, :15])
        assert numpy.all(~mask[:10, 15:])
        assert secondary.dtype == numpy.dtype(numpy.float32)
