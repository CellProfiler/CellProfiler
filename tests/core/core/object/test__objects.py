import numpy

from cellprofiler_core.image import Image
from cellprofiler_core.object import Objects


class TestObjects:
    def test_dimensions(self):
        x = numpy.zeros((100, 224, 224, 3), numpy.float32)

        parent_image = Image(x, dimensions=3)

        objects = Objects()

        objects.parent_image = parent_image

        assert objects.dimensions == 3

    def test_volumetric(self):
        x = numpy.zeros((100, 224, 224, 3), numpy.float32)

        parent_image = Image(x, dimensions=3)

        objects = Objects()

        objects.parent_image = parent_image

        assert objects.volumetric

    def test_masked(self):
        x = numpy.zeros((224, 224, 3), numpy.float32)

        mask = numpy.ones((224, 224), bool)

        parent_image = Image(x, mask=mask)

        objects = Objects()

        objects.segmented = mask

        objects.parent_image = parent_image

        numpy.testing.assert_array_equal(objects.masked, mask)

    def test_shape(self):
        objects = Objects()

        objects.segmented = numpy.ones((224, 224), numpy.uint8)

        assert objects.shape == (224, 224)

    def test_segmented(self):
        segmented = numpy.ones((224, 224), bool)

        objects = Objects()

        objects.segmented = segmented

        numpy.testing.assert_array_equal(objects.segmented, segmented)

    def test_indices(self):
        pass

    def test_count(self):
        pass

    def test_areas(self):
        pass

    def test_set_ijv(self):
        pass

    def test_get_ijv(self):
        pass

    def test_get_labels(self):
        pass

    def test_has_unedited_segmented(self):
        pass

    def test_unedited_segmented(self):
        pass

    def test_has_small_removed_segmented(self):
        pass

    def test_small_removed_segmented(self):
        pass

    def test_parent_image(self):
        pass

    def test_has_parent_image(self):
        pass

    def test_crop_image_similarly(self):
        pass

    def test_make_ijv_outlines(self):
        pass

    def test_relate_children(self):
        pass

    def test_relate_labels(self):
        pass

    def test_relate_histogram(self):
        pass

    def test_histogram_from_labels(self):
        pass

    def test_histogram_from_ijv(self):
        pass

    def test_fn_of_label_and_index(self):
        pass

    def test_fn_of_ones_label_and_index(self):
        pass

    def test_center_of_mass(self):
        pass

    def test_overlapping(self):
        pass
