import numpy
import pytest
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.mergeobjects
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.workspace


@pytest.fixture
def image():
    data = numpy.zeros((10, 10))

    return cellprofiler.image.Image(data)


@pytest.fixture
def images():
    return cellprofiler.image.ImageSet(0, {"number": 0}, {})


@pytest.fixture
def input_object_a():
    segmented = cellprofiler.object.Objects()

    segmented.segmented = numpy.zeros((10, 10))

    return segmented


@pytest.fixture
def input_object_b():
    segmented = cellprofiler.object.Objects()

    segmented.segmented = numpy.zeros((10, 10))

    return segmented


@pytest.fixture
def measurements():
    return cellprofiler.measurement.Measurements()


@pytest.fixture
def module():
    return cellprofiler.modules.mergeobjects.MergeObjects()


@pytest.fixture
def objects():
    return cellprofiler.object.ObjectSet()


@pytest.fixture
def pipeline():
    return cellprofiler.pipeline.Pipeline()


@pytest.fixture
def workspace(images, input_object_a, input_object_b, measurements, module, objects, pipeline):
    images.add("example", image)

    objects.add_objects(input_object_a, "m")
    objects.add_objects(input_object_b, "n")

    module.input_object_a.value = "m"
    module.input_object_b.value = "n"
    module.output_object.value = "merged"

    return cellprofiler.workspace.Workspace(pipeline, module, images, objects, measurements, None)


class TestMergeObjects:
    def test_display(self):
        pass

    class TestRun:
        def test_zero_objects(self, module, workspace):
            module.run(workspace)

            # TEST:
            # Assert the object set includes the merged objects
            object_set = workspace.object_set
            assert len(object_set.object_names) == 3

            # Assert the object set is properly named
            assert "merged" in object_set.object_names

            # Assert zero objects (no segments)
            output_objects = object_set.get_objects("merged")
            assert len(output_objects.segmented[output_objects.segmented > 0]) == 0

        def test_one_object_first_image(self, input_object_a, module, workspace):
            segment = numpy.zeros((10, 10))
            segment[2][2] = 1
            segment[2][3] = 1
            segment[3][2] = 1
            segment[3][3] = 1

            input_object_a.segmented = segment

            module.run(workspace)

            assert (workspace.get_objects("merged").segmented == segment).all()

        def test_one_object_second_image(self, input_object_b, module, workspace):
            segment = numpy.zeros((10, 10))
            segment[2][2] = 1
            segment[2][3] = 1
            segment[3][2] = 1
            segment[3][3] = 1

            input_object_b.segmented = segment

            module.run(workspace)

            assert (workspace.get_objects("merged").segmented == segment).all()

        def test_disjoint(self, input_object_a, input_object_b, module, workspace):
            # Create a segment for the first input object
            segment_a = numpy.zeros((10, 10))
            segment_a[2][2] = 1
            segment_a[2][3] = 1
            segment_a[3][2] = 1
            segment_a[3][3] = 1

            input_object_a.segmented = segment_a

            # Create a segment for the second input object
            segment_b = numpy.zeros((10, 10))
            segment_b[8][8] = 1
            segment_b[8][9] = 1
            segment_b[9][8] = 1
            segment_b[9][9] = 1

            input_object_b.segmented = segment_b

            module.run(workspace)

            # TEST:
            # Assert two segments exist
            merged = workspace.get_objects("merged")
            assert len(merged.indices) == 2

            # Verify segment positions
            assert (merged.segmented == segment_a + 2*segment_b).all()

        # Alternative approach: keep segments separated with the dividing line cross their
        # intersection (see: MaskObjects). Only merge when one segment is completely enclosed
        # in another. Is that easily determined?
        def test_overlap(self, input_object_a, input_object_b, module, workspace):
            # Create a segment for the first input object
            segment_a = numpy.zeros((10, 10))
            segment_a[2][2] = 1
            segment_a[2][3] = 1
            segment_a[3][2] = 1
            segment_a[3][3] = 1

            input_object_a.segmented = segment_a

            # Create a segment for the second input object
            segment_b = numpy.zeros((10, 10))
            segment_b[3][3] = 1
            segment_b[3][4] = 1

            input_object_b.segmented = segment_b

            module.run(workspace)

            # TEST:
            # Assert one segment exists
            merged = workspace.get_objects("merged")
            assert len(merged.indices) == 1

            # Verify segment position
            expected_segment = segment_a + segment_b
            expected_segment[expected_segment > 0] = 1
            assert (merged.segmented == expected_segment).all()

        def test_adjacent_segments_same_object(self, input_object_a, module, workspace):
            # Create a segment for the first input object
            segmentation = numpy.zeros((10, 10))
            segmentation[2][2] = 1
            segmentation[2][3] = 1
            segmentation[3][2] = 2
            segmentation[3][3] = 2

            input_object_a.segmented = segmentation

            module.run(workspace)

            # TEST:
            # Assert two segments exist
            merged = workspace.get_objects("merged")
            assert len(merged.indices) == 2

            # Verify segment position
            assert (merged.segmented == segmentation).all()

        def test_adjacent_segments_different_objects(self, input_object_a, input_object_b, module, workspace):
            # Create a segment for the first input object
            segment_a = numpy.zeros((10, 10))
            segment_a[2][2] = 1
            segment_a[2][3] = 1

            input_object_a.segmented = segment_a

            # Create a segment for the second input object
            segment_b = numpy.zeros((10, 10))
            segment_b[3][2] = 1
            segment_b[3][3] = 1

            input_object_b.segmented = segment_b

            module.run(workspace)

            # TEST:
            # Assert two segments exist
            merged = workspace.get_objects("merged")
            assert len(merged.indices) == 2

            # Verify segment position
            assert (merged.segmented == segment_a + 2*segment_b).all()
