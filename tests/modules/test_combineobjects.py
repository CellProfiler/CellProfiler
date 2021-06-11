import numpy
import pytest


import cellprofiler.modules.combineobjects
import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace


@pytest.fixture
def image():
    data = numpy.zeros((10, 10))

    return cellprofiler_core.image.Image(data)

@pytest.fixture
def image_volume():
    data = numpy.zeros((10, 10, 10))

    return cellprofiler_core.image.Image(data)

@pytest.fixture
def images():
    return cellprofiler_core.image.ImageSet(0, {"number": 0}, {})


@pytest.fixture
def objects_x():
    segmented = cellprofiler_core.object.Objects()

    segmented.segmented = numpy.zeros((10, 10))

    return segmented

@pytest.fixture
def objects_x_volume():
    segmented = cellprofiler_core.object.Objects()

    segmented.segmented = numpy.zeros((10, 10, 10))

    return segmented

@pytest.fixture
def objects_y():
    segmented = cellprofiler_core.object.Objects()

    segmented.segmented = numpy.zeros((10, 10))

    return segmented

@pytest.fixture
def objects_y_volume():
    segmented = cellprofiler_core.object.Objects()

    segmented.segmented = numpy.zeros((10, 10, 10))

    return segmented

@pytest.fixture
def measurements():
    return cellprofiler_core.measurement.Measurements()


@pytest.fixture
def module():
    return cellprofiler.modules.combineobjects.CombineObjects()


@pytest.fixture
def objects():
    return cellprofiler_core.object.ObjectSet()


@pytest.fixture
def pipeline():
    return cellprofiler_core.pipeline.Pipeline()


@pytest.fixture
def merge_methods():
    return ["Merge", "Preserve", "Discard", "Segment"]


@pytest.fixture
def workspace(images, objects_x, objects_y, measurements, module, objects, pipeline):
    images.add("example", image)

    objects.add_objects(objects_x, "m")
    objects.add_objects(objects_y, "n")

    module.objects_x.value = "m"
    module.objects_y.value = "n"
    module.output_object.value = "merged"

    return cellprofiler_core.workspace.Workspace(
        pipeline, module, images, objects, measurements, None
    )

@pytest.fixture
def workspace_volume(images, objects_x_volume, objects_y_volume, measurements, module, objects, pipeline):
    images.add("example", image_volume)

    objects.add_objects(objects_x_volume, "m")
    objects.add_objects(objects_y_volume, "n")

    module.objects_x.value = "m"
    module.objects_y.value = "n"
    module.output_object.value = "merged"

    return cellprofiler_core.workspace.Workspace(
        pipeline, module, images, objects, measurements, None
    )

class TestCombineObjects:
    def test_display(self):
        pass

    class TestRun:
        def test_zero_objects(self, module, workspace, merge_methods):
            # Test merge methods with blank arrays
            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace)
                output_objects = workspace.object_set.get_objects(method)
                assert len(output_objects.segmented[output_objects.segmented > 0]) == 0

        def test_one_object_first_image(
            self, objects_x, module, workspace, merge_methods
        ):
            # Test merge methods with one object in initial set
            segment = numpy.zeros((10, 10))
            segment[2:4,2:4] = 1

            objects_x.segmented = segment

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace)
                numpy.testing.assert_array_equal(workspace.get_objects(method).segmented, segment)

        def test_one_object_first_image_volume(
            self, objects_x_volume, module, workspace_volume, merge_methods
        ):
            # Test merge methods with one object in initial set for volumes
            segment = numpy.zeros((10, 10, 10))
            segment[2:4,2:4,2:4] = 1

            objects_x_volume.segmented = segment

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace_volume)
                numpy.testing.assert_array_equal(workspace_volume.get_objects(method).segmented, segment)

        def test_one_object_second_image(
            self, objects_y, module, workspace, merge_methods
        ):
            # Test merge methods with one object in target set
            segment = numpy.zeros((10, 10))
            segment[2:4, 2:4] = 1

            objects_y.segmented = segment

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace)
                numpy.testing.assert_array_equal(workspace.get_objects(method).segmented, segment)

        def test_one_object_second_image_volume(
                self, objects_y_volume, module, workspace_volume, merge_methods
        ):
            # Test merge methods with one object in target set for volumes
            segment = numpy.zeros((10, 10, 10))
            segment[2:4, 2:4, 2:4] = 1

            objects_y_volume.segmented = segment

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace_volume)
                numpy.testing.assert_array_equal(workspace_volume.get_objects(method).segmented, segment)

        def test_duplicate_object(self, objects_x, objects_y, module, workspace, merge_methods):
            # Test merge methods with same object in both sets
            segment = numpy.zeros((10, 10))
            segment[2:4, 2:4] = 1

            objects_x.segmented = segment
            objects_y.segmented = segment

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace)
                numpy.testing.assert_array_equal(workspace.get_objects(method).segmented, segment)

        def test_duplicate_object_volume(self, objects_x_volume, module, workspace_volume, merge_methods):
            # Test merge methods with same object in both sets for volumes
            segment = numpy.zeros((10, 10, 10))
            segment[2:4, 2:4, 2:4] = 1

            objects_x_volume.segmented = segment
            objects_y_volume.segmented = segment

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace_volume)
                numpy.testing.assert_array_equal(workspace_volume.get_objects(method).segmented, segment)

        def test_not_touching(
            self, objects_x, objects_y, module, workspace, merge_methods
        ):
            # Test merge methods with two distinct objects
            segment_x = numpy.zeros((10, 10))
            segment_x[2:4, 2:4] = 1
            objects_x.segmented = segment_x

            segment_y = numpy.zeros((10, 10))
            segment_y[8:10, 8:10] = 1
            objects_y.segmented = segment_y

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace)
                combined = workspace.get_objects(method)
                assert len(combined.indices) == 2
                numpy.testing.assert_array_equal(combined.segmented, segment_x + 2 * segment_y)

        def test_not_touching_volume(
            self, objects_x_volume, objects_y_volume, module, workspace_volume, merge_methods
        ):
            # Test merge methods with two distinct objects for volumes
            segment_x = numpy.zeros((10, 10, 10))
            segment_x[2:4, 2:4, 2:4] = 1
            objects_x_volume.segmented = segment_x

            segment_y = numpy.zeros((10, 10, 10))
            segment_y[8:10, 8:10, 8:10] = 1
            objects_y_volume.segmented = segment_y

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace_volume)
                combined = workspace_volume.get_objects(method)
                assert len(combined.indices) == 2
                numpy.testing.assert_array_equal(combined.segmented, segment_x + 2 * segment_y)

        def test_for_inappropriate_merge(
            self, objects_x, objects_y, module, workspace, merge_methods
        ):
            # Test that adjacent objects in the source set aren't merged inappropriately.
            segmentation_x = numpy.zeros((10, 10))
            segmentation_x[2, 2:4] = 1
            segmentation_x[3, 2:4] = 2

            segmentation_y = numpy.zeros((10, 10))
            segmentation_y[6,6] = 3
            objects_x.segmented = segmentation_x
            objects_y.segmented = segmentation_y

            segmentation_expected = segmentation_x + segmentation_y

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace)
                merged = workspace.get_objects(method)
                assert len(merged.indices) == 3
                numpy.testing.assert_array_equal(merged.segmented, segmentation_expected)

        def test_for_inappropriate_merge_volume(
            self, objects_x_volume, objects_y_volume, module, workspace_volume, merge_methods
        ):
            # Test that adjacent objects in the source set aren't merged inappropriately for volumes.
            # This creates two adjacent planes in the source set
            segmentation_x = numpy.zeros((10, 10, 10))
            segmentation_x[2, 2:4, 2:4] = 1
            segmentation_x[3, 2:4, 2:4] = 2

            segmentation_y = numpy.zeros((10, 10, 10))
            # single point in the second set
            segmentation_y[6,6,6] = 3
            objects_x_volume.segmented = segmentation_x
            objects_y_volume.segmented = segmentation_y

            segmentation_expected = segmentation_x + segmentation_y

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace_volume)
                merged = workspace_volume.get_objects(method)
                assert len(merged.indices) == 3
                numpy.testing.assert_array_equal(merged.segmented, segmentation_expected)

        def test_overlap_discard(self, objects_x, objects_y, module, workspace):
            # Test handling of overlapping objects in 'discard' mode
            segment_x = numpy.zeros((10, 10))
            segment_x[2:4, 2:4] = 1
            objects_x.segmented = segment_x

            segment_y = numpy.zeros((10, 10))
            segment_y[3, 3:5] = 1
            objects_y.segmented = segment_y

            module.merge_method.value = "Discard"
            module.run(workspace)

            merged = workspace.get_objects("merged")
            assert len(merged.indices) == 1
            expected_segment = segment_x
            numpy.testing.assert_array_equal(merged.segmented, expected_segment)

        def test_overlap_discard_volume(self, objects_x_volume, objects_y_volume, module, workspace_volume):
            # Test handling of overlapping objects in 'discard' mode for volumes
            segment_x = numpy.zeros((10, 10, 10))
            segment_x[2:4, 2:4, 2:4] = 1
            objects_x_volume.segmented = segment_x

            segment_y = numpy.zeros((10, 10, 10))
            segment_y[3, 3:5, 3:5] = 1
            objects_y_volume.segmented = segment_y

            module.merge_method.value = "Discard"
            module.run(workspace_volume)

            merged = workspace_volume.get_objects("merged")
            assert len(merged.indices) == 1
            expected_segment = segment_x
            numpy.testing.assert_array_equal(merged.segmented, expected_segment)

        def test_overlap_preserve(self, objects_x, objects_y, module, workspace):
            # Test handling of overlapping objects in 'preserve' mode
            segment_x = numpy.zeros((10, 10))
            segment_x[2:4, 2:4] = 1
            objects_x.segmented = segment_x

            segment_y = numpy.zeros((10, 10))
            segment_y[3, 3:5] = 1
            objects_y.segmented = segment_y

            module.merge_method.value = "Preserve"
            module.run(workspace)

            merged = workspace.get_objects("merged")
            assert len(merged.indices) == 2
            expected_segment = numpy.zeros_like(segment_x)
            expected_segment[segment_y > 0] = 2
            expected_segment[segment_x > 0] = 1
            numpy.testing.assert_array_equal(merged.segmented, expected_segment)

        def test_overlap_preserve_volume(self, objects_x_volume, objects_y_volume, module, workspace_volume):
            # Test handling of overlapping objects in 'preserve' mode for volumes
            segment_x = numpy.zeros((10, 10, 10))
            segment_x[2:4, 2:4, 2:4] = 1
            objects_x_volume.segmented = segment_x

            segment_y = numpy.zeros((10, 10, 10))
            segment_y[3, 3:5, 3:5] = 1
            objects_y_volume.segmented = segment_y

            module.merge_method.value = "Preserve"
            module.run(workspace_volume)

            merged = workspace_volume.get_objects("merged")
            assert len(merged.indices) == 2
            expected_segment = numpy.zeros_like(segment_x)
            expected_segment[segment_y > 0] = 2
            expected_segment[segment_x > 0] = 1
            numpy.testing.assert_array_equal(merged.segmented, expected_segment)

        def test_overlap_merge(self, objects_x, objects_y, module, workspace):
            # Test handling of overlapping objects in 'merge' mode
            segment_x = numpy.zeros((10, 10))
            segment_x[2:4, 2:4] = 1
            objects_x.segmented = segment_x

            segment_y = numpy.zeros((10, 10))
            segment_y[3:5, 3:5] = 1
            objects_y.segmented = segment_y

            module.merge_method.value = "Merge"
            module.run(workspace)
            merged = workspace.get_objects("merged")

            assert len(merged.indices) == 1
            expected_segment = numpy.zeros_like(segment_x)
            expected_segment[segment_y > 0] = 1
            expected_segment[segment_x > 0] = 1
            numpy.testing.assert_array_equal(merged.segmented, expected_segment)

        def test_overlap_merge_volume(self, objects_x_volume, objects_y_volume, module, workspace_volume):
            # Test handling of overlapping objects in 'merge' mode for volumes
            segment_x = numpy.zeros((10, 10, 10))
            segment_x[2:4, 2:4, 2:4] = 1
            objects_x_volume.segmented = segment_x

            segment_y = numpy.zeros((10, 10, 10))
            segment_y[3:5, 3:5, 3:5] = 1
            objects_y_volume.segmented = segment_y

            module.merge_method.value = "Merge"
            module.run(workspace_volume)
            merged = workspace_volume.get_objects("merged")

            assert len(merged.indices) == 1
            # the expected segmentation includes segment_x and segment_y as a single object
            expected_segment = numpy.zeros_like(segment_x)
            expected_segment[segment_y > 0] = 1
            expected_segment[segment_x > 0] = 1
            numpy.testing.assert_array_equal(merged.segmented, expected_segment)

        def test_overlap_segment(self, objects_x, objects_y, module, workspace):
            # Test handling of overlapping objects in 'segment' mode
            segment_x = numpy.zeros((10, 10))
            segment_x[1:5, 1:6] = 1
            objects_x.segmented = segment_x

            segment_y = numpy.zeros((10, 10))
            segment_y[1:6, 4:9] = 1
            objects_y.segmented = segment_y

            module.merge_method.value = "Segment"
            module.run(workspace)
            merged = workspace.get_objects("merged")

            assert len(merged.indices) == 2
            expected_segment = numpy.zeros_like(segment_x)
            expected_segment[1:6, 4:9] = 2
            expected_segment[1:5, 1:5] = 1
            numpy.testing.assert_array_equal(merged.segmented, expected_segment)

        def test_overlap_segment_volume(self, objects_x_volume, objects_y_volume, module, workspace_volume):
            # Test handling of overlapping objects in 'segment' mode for volumes
            segment_x = numpy.zeros((10, 10, 10))
            segment_x[1:5, 1:6, 1:6] = 1
            objects_x_volume.segmented = segment_x

            segment_y = numpy.zeros((10, 10, 10))
            segment_y[1:9, 4:9, 4:9] = 1
            objects_y_volume.segmented = segment_y

            module.merge_method.value = "Segment"
            module.run(workspace_volume)
            merged = workspace_volume.get_objects("merged")

            assert len(merged.indices) == 2
            expected_segment = numpy.zeros_like(segment_x)
            # in 3D, the objects get segmented on a diagonal
            expected_segment[1:5, 1:6, 1:5] = 1
            expected_segment[1:5, 1:5, 5:6] = 1
            expected_segment[1:5, 6:9, 4:5] = 2
            expected_segment[1:5, 5:9, 5:6] = 2
            expected_segment[1:5, 4:9, 6:9] = 2
            expected_segment[5:9, 4:9, 4:9] = 2
            numpy.testing.assert_array_equal(merged.segmented, expected_segment)
