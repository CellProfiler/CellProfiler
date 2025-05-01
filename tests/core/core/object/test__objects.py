import numpy
import pytest
from cellprofiler_core.image import Image
from cellprofiler_core.object import Objects
from cellprofiler_core.object import ObjectSet
import cellprofiler_core


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

    def test_relate_children_monotonically_increasing_parent_ids(self, create_lil_shaped_objects):
        # Create an image like "|:|" or "lil" where there is one long vertical object, with two small objects on the right followed by a large vertical object
        parent, child = create_lil_shaped_objects
        children_per_parent, parents_of_children = parent.relate_children(child)
        assert sum(children_per_parent) == len(children_per_parent)
        assert sum([i == j for i, j in zip(parents_of_children, range(1, len(children_per_parent) + 1))]) == len(children_per_parent)
        

    def test_relate_children_monotonically_increasing_parent_ids_transposed(self, create_lil_shaped_objects):
        # Create an image like "|:|" or "lil" where there is one long vertical object, with two small objects on the right followed by a large vertical object
        parent, child = create_lil_shaped_objects
        parent.segmented = numpy.transpose(parent.segmented)
        child.segmented = numpy.transpose(child.segmented)
        children_per_parent, parents_of_children = parent.relate_children(child)
        assert sum(children_per_parent) == len(children_per_parent)
        assert sum([i == j for i, j in zip(parents_of_children, range(1, len(children_per_parent) + 1))]) == len(children_per_parent)


    def test_relate_children_monotonically_increasing_parent_ids_one_object_removed(self, create_lil_shaped_objects):
        # Create an image like "|:|" or "lil" where there is one long vertical object, with two small objects on the right followed by a large vertical object
        parent, child = create_lil_shaped_objects
        obj_num = 2
        child.segmented[parent.segmented == obj_num] = 0
        children_per_parent, parents_of_children = parent.relate_children(child)
        assert sum(children_per_parent) == len(children_per_parent) -1 
        assert sum([i == j for i, j in zip(parents_of_children, range(1, len(children_per_parent) + 1))]) != len(children_per_parent)

    @pytest.mark.xfail
    def test_relate_children_monotonically_increasing_parent_ids_label_swapped(self, create_lil_shaped_objects):
        # Create an image like "|:|" or "lil" where there is one long vertical object, with two small objects on the right followed by a large vertical object
        parent, child = create_lil_shaped_objects
        obj_num1 = 2
        obj_num2 = 3
        # swap the labels of the two small objects
        # this means that the nucleus of object 2 is now the nucleus of object 3 but the cell label is still object 2
        child.segmented[child.segmented == obj_num1] = -1
        child.segmented[child.segmented == obj_num2] = obj_num1
        child.segmented[child.segmented == -1] = obj_num2
        children_per_parent, parents_of_children = parent.relate_children(child)
        # TODO: I'm unsure of the expected behavior here
        assert sum(children_per_parent) == len(children_per_parent) 
        assert sum([i == j for i, j in zip(parents_of_children, range(1, len(children_per_parent) + 1))]) == len(children_per_parent)

    @pytest.mark.xfail
    def test_relate_children_monotonically_increasing_nucleus_overflow_equal_parts(self, create_object_primary_overflows_equal_parts_into_other_object):
        # Create an image where we have two cells with one nucleus each but one of the nucleus overflows into parts of the other cell
        parent, child = create_object_primary_overflows_equal_parts_into_other_object
        children_per_parent, parents_of_children = parent.relate_children(child)
        # TODO: I'm unsure of the expected behavior here
        assert sum(children_per_parent) == len(children_per_parent) 
        assert sum([i == j for i, j in zip(parents_of_children, range(1, len(children_per_parent) + 1))]) == len(children_per_parent)

    @pytest.mark.xfail
    def test_relate_children_monotonically_increasing_nucleus_overflow_unequal_parts(self, create_object_primary_overflows_unequal_parts_into_other_object):
        # Create an image where we have two cells with one nucleus each but one of the nucleus overflows into parts of the other cell
        parent, child = create_object_primary_overflows_unequal_parts_into_other_object
        children_per_parent, parents_of_children = parent.relate_children(child)
        # TODO: I'm unsure of the expected behavior here
        assert sum(children_per_parent) == len(children_per_parent) 
        assert sum([i == j for i, j in zip(parents_of_children, range(1, len(children_per_parent) + 1))]) == len(children_per_parent)


@pytest.fixture(
    scope="function", 
    params= [
        (1, 2, 3, 4),
        (4, 3, 2, 1),
        (1, 3, 2, 4),
        (4, 2, 3, 1),
        (1, 4, 2, 3),
        (3, 1, 4, 2),
        (2, 4, 1, 3),
        (3, 2, 4, 1),
    ]
    )
def create_lil_shaped_objects(request):
    # parent is always bigger than child
    # Create an image like "|:|" or "lil" where there is one long vertical object, with two small objects on the right followed by a large vertical object

    objects = Objects()
    object_labels = numpy.zeros((10, 30))
    object_labels[1:9, 2:8] = request.param[0]
    object_labels[1:4, 12:18] = request.param[1]
    object_labels[6:9, 12:18] = request.param[2]
    object_labels[1:9, 22:28] = request.param[3]
    objects.segmented = object_labels

    child = Objects()
    child_labels = numpy.zeros((10, 30))
    child_labels[3:7, 3:7] = request.param[0]
    child_labels[2:3, 13:17] = request.param[1]
    child_labels[7:8, 13:17] = request.param[2]
    child_labels[3:7, 23:27] = request.param[3]
    child.segmented = child_labels

    return objects, child

@pytest.fixture(scope="function")
def create_object_primary_overflows_equal_parts_into_other_object():
    # Create an image where we have two cells with one nucleus each but one of the nucleus overflows into parts of the other cell
    objects = Objects()
    object_labels = numpy.zeros((10, 30))
    object_labels[1:9, 2:8] = 1
    object_labels[1:9, 10: 15] = 2
    objects.segmented = object_labels

    child = Objects()
    child_labels = numpy.zeros((10, 30))
    child_labels[2:8, 3:7] = 1
    child_labels[3:6, 6: 12] = 2
    child.segmented = child_labels
    return objects, child 

@pytest.fixture(scope="function")
def create_object_primary_overflows_unequal_parts_into_other_object():
    # Create an image where we have two cells with one nucleus each but one of the nucleus overflows into parts of the other cell
    objects = Objects()
    object_labels = numpy.zeros((10, 30))
    object_labels[1:9, 2:8] = 1
    object_labels[1:9, 10: 15] = 2
    objects.segmented = object_labels

    child = Objects()
    child_labels = numpy.zeros((10, 30))
    child_labels[2:8, 3:7] = 1
    child_labels[3:6, 6: 12] = 2
    child.segmented = child_labels
    return objects, child 

IMAGE1_NAME = "image1"
OBJECT1_NAME = "object1"


