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
def images():
    return cellprofiler_core.image.ImageSet(0, {"number": 0}, {})


@pytest.fixture
def objects_x():
    segmented = cellprofiler_core.object.Objects()

    segmented.segmented = numpy.zeros((10, 10))

    return segmented


@pytest.fixture
def objects_y():
    segmented = cellprofiler_core.object.Objects()

    segmented.segmented = numpy.zeros((10, 10))

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
            segment[2][2] = 1
            segment[2][3] = 1
            segment[3][2] = 1
            segment[3][3] = 1

            objects_x.segmented = segment

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace)
                assert (workspace.get_objects(method).segmented == segment).all()

        def test_one_object_second_image(
            self, objects_y, module, workspace, merge_methods
        ):
            # Test merge methods with one object in target set
            segment = numpy.zeros((10, 10))
            segment[2][2] = 1
            segment[2][3] = 1
            segment[3][2] = 1
            segment[3][3] = 1

            objects_y.segmented = segment

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace)
                assert (workspace.get_objects(method).segmented == segment).all()

        def test_duplicate_object(self, objects_x, module, workspace, merge_methods):
            # Test merge methods with same object in both sets
            segment = numpy.zeros((10, 10))
            segment[2][2] = 1
            segment[2][3] = 1
            segment[3][2] = 1
            segment[3][3] = 1

            objects_x.segmented = segment
            objects_y.segmented = segment

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace)
                assert (workspace.get_objects(method).segmented == segment).all()

        def test_not_touching(
            self, objects_x, objects_y, module, workspace, merge_methods
        ):
            # Test merge methods with two distinct objects
            segment_x = numpy.zeros((10, 10))
            segment_x[2][2] = 1
            segment_x[2][3] = 1
            segment_x[3][2] = 1
            segment_x[3][3] = 1
            objects_x.segmented = segment_x

            segment_y = numpy.zeros((10, 10))
            segment_y[8][8] = 1
            segment_y[8][9] = 1
            segment_y[9][8] = 1
            segment_y[9][9] = 1
            objects_y.segmented = segment_y

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace)
                combined = workspace.get_objects(method)
                assert len(combined.indices) == 2
                assert (combined.segmented == segment_x + 2 * segment_y).all()

        def test_for_inappropriate_merge(
            self, objects_x, objects_y, module, workspace, merge_methods
        ):
            # Test that adjacent objects in the source set aren't merged inappropriately.
            segmentation_x = numpy.zeros((10, 10))
            segmentation_x[2][2] = 1
            segmentation_x[2][3] = 1
            segmentation_x[3][2] = 2
            segmentation_x[3][3] = 2

            segmentation_y = numpy.zeros((10, 10))
            segmentation_y[6][6] = 3
            objects_x.segmented = segmentation_x
            objects_y.segmented = segmentation_y

            segmentation_expected = segmentation_x + segmentation_y

            for method in merge_methods:
                module.merge_method.value = method
                module.output_object.value = method
                module.run(workspace)
                merged = workspace.get_objects(method)
                assert len(merged.indices) == 3
                assert (merged.segmented == segmentation_expected).all()

        def test_overlap_discard(self, objects_x, objects_y, module, workspace):
            # Test handling of overlapping objects in 'discard' mode
            segment_x = numpy.zeros((10, 10))
            segment_x[2][2] = 1
            segment_x[2][3] = 1
            segment_x[3][2] = 1
            segment_x[3][3] = 1
            objects_x.segmented = segment_x

            segment_y = numpy.zeros((10, 10))
            segment_y[3][3] = 1
            segment_y[3][4] = 1
            objects_y.segmented = segment_y

            module.merge_method.value = "Discard"
            module.run(workspace)

            merged = workspace.get_objects("merged")
            assert len(merged.indices) == 1
            expected_segment = segment_x
            assert (merged.segmented == expected_segment).all()

        def test_overlap_preserve(self, objects_x, objects_y, module, workspace):
            # Test handling of overlapping objects in 'preserve' mode
            segment_x = numpy.zeros((10, 10))
            segment_x[2][2] = 1
            segment_x[2][3] = 1
            segment_x[3][2] = 1
            segment_x[3][3] = 1
            objects_x.segmented = segment_x

            segment_y = numpy.zeros((10, 10))
            segment_y[3][3] = 1
            segment_y[3][4] = 1
            objects_y.segmented = segment_y

            module.merge_method.value = "Preserve"
            module.run(workspace)

            merged = workspace.get_objects("merged")
            assert len(merged.indices) == 2
            expected_segment = numpy.zeros_like(segment_x)
            expected_segment[segment_y > 0] = 2
            expected_segment[segment_x > 0] = 1
            assert (merged.segmented == expected_segment).all()

        def test_overlap_merge(self, objects_x, objects_y, module, workspace):
            # Test handling of overlapping objects in 'merge' mode
            segment_x = numpy.zeros((10, 10))
            segment_x[2][2] = 1
            segment_x[2][3] = 1
            segment_x[3][2] = 1
            segment_x[3][3] = 1
            objects_x.segmented = segment_x

            segment_y = numpy.zeros((10, 10))
            segment_y[3][3] = 1
            segment_y[3][4] = 1
            segment_y[4][3] = 1
            segment_y[4][4] = 1
            objects_y.segmented = segment_y

            module.merge_method.value = "Merge"
            module.run(workspace)
            merged = workspace.get_objects("merged")

            assert len(merged.indices) == 1
            expected_segment = numpy.zeros_like(segment_x)
            expected_segment[segment_y > 0] = 1
            expected_segment[segment_x > 0] = 1
            assert (merged.segmented == expected_segment).all()

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
            assert (merged.segmented == expected_segment).all()
