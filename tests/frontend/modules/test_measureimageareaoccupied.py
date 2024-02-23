import numpy
import six

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT


import cellprofiler.modules.measureimageareaoccupied
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

OBJECTS_NAME = "MyObjects"


def make_workspace(labels, parent_image=None):
    object_set = cellprofiler_core.object.ObjectSet()
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    objects.parent_image = parent_image
    object_set.add_objects(objects, OBJECTS_NAME)

    pipeline = cellprofiler_core.pipeline.Pipeline()
    module = cellprofiler.modules.measureimageareaoccupied.MeasureImageAreaOccupied()
    module.set_module_num(1)
    module.operand_choice.value = (
        cellprofiler.modules.measureimageareaoccupied.O_OBJECTS
    )
    module.objects_list.value = OBJECTS_NAME
    pipeline.add_module(module)
    image_set_list = cellprofiler_core.image.ImageSetList()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set_list.get_image_set(0),
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return workspace


def test_zeros():
    workspace = make_workspace(numpy.zeros((10, 10), int))
    module = workspace.module
    module.operand_choice.value = "Objects"
    module.run(workspace)
    m = workspace.measurements

    def mn(x):
        return "AreaOccupied_%s_%s" % (x, module.objects_list.value[0])

    assert m.get_current_measurement("Image", mn("AreaOccupied")) == 0.0
    assert m.get_current_measurement("Image", mn("TotalArea")) == 100

    columns = module.get_measurement_columns(workspace.pipeline)
    features = m.get_feature_names("Image")
    assert len(columns) == len(features)
    for column in columns:
        assert column[1] in features


def test_one_object():
    labels = numpy.zeros((10, 10), int)
    labels[2:7, 3:8] = 1
    area_occupied = numpy.sum(labels)
    workspace = make_workspace(labels)
    module = workspace.module
    module.operand_choice.value = "Objects"
    module.run(workspace)
    m = workspace.measurements

    def mn(x):
        return "AreaOccupied_%s_%s" % (x, module.objects_list.value[0])

    assert m.get_current_measurement("Image", mn("AreaOccupied")) == area_occupied
    assert m.get_current_measurement("Image", mn("TotalArea")) == 100


def test_object_with_cropping():
    labels = numpy.zeros((10, 10), int)
    labels[0:7, 3:8] = 1
    mask = numpy.zeros((10, 10), bool)
    mask[1:9, 1:9] = True
    image = cellprofiler_core.image.Image(numpy.zeros((10, 10)), mask=mask)
    area_occupied = [30]
    perimeter = [18]
    total_area = 64
    workspace = make_workspace(labels, image)
    module = workspace.module
    module.operand_choice.value = "Objects"
    module.run(workspace)
    m = workspace.measurements

    def mn(x):
        return "AreaOccupied_%s_%s" % (x, module.objects_list.value[0])

    assert m.get_current_measurement("Image", mn("AreaOccupied")) == area_occupied
    assert m.get_current_measurement("Image", mn("Perimeter")) == perimeter
    assert m.get_current_measurement("Image", mn("TotalArea")) == total_area


def test_get_measurement_columns():
    module = cellprofiler.modules.measureimageareaoccupied.MeasureImageAreaOccupied()
    module.objects_list.value = OBJECTS_NAME
    module.operand_choice.value = "Objects"
    columns = module.get_measurement_columns(cellprofiler_core.pipeline.Pipeline())
    expected = (
        (
            "Image",
            "AreaOccupied_AreaOccupied_%s" % OBJECTS_NAME,
            COLTYPE_FLOAT,
        ),
        (
            "Image",
            "AreaOccupied_Perimeter_%s" % OBJECTS_NAME,
            COLTYPE_FLOAT,
        ),
        (
            "Image",
            "AreaOccupied_TotalArea_%s" % OBJECTS_NAME,
            COLTYPE_FLOAT,
        ),
    )
    assert len(columns) == len(expected)
    for column in columns:
        assert any([all([cf == ef for cf, ef in zip(column, ex)]) for ex in expected])


def test_objects_volume():
    labels = numpy.zeros((5, 10, 10), dtype=numpy.uint8)
    labels[:2, :2, :2] = 1
    labels[3:, 8:, 8:] = 2

    expected_area = 16
    expected_perimeter = 16
    expected_total_area = 500

    workspace = make_workspace(labels)
    workspace.pipeline.set_volumetric(True)

    module = workspace.module
    module.operand_choice.value = "Objects"

    module.run(workspace)

    def mn(x):
        return "AreaOccupied_%s_%s" % (x, module.objects_list.value[0])

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement("Image", mn("VolumeOccupied")),
        expected_area,
    )

    numpy.testing.assert_array_almost_equal(
        workspace.measurements.get_current_measurement("Image", mn("SurfaceArea")),
        expected_perimeter,
        decimal=0,
    )

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement("Image", mn("TotalVolume")),
        expected_total_area,
    )


def test_image_volume():
    pixel_data = numpy.zeros((5, 10, 10), dtype=bool)
    pixel_data[:2, :2, :2] = True
    pixel_data[3:, 8:, 8:] = True

    image = cellprofiler_core.image.Image(pixel_data, dimensions=3)

    expected_area = [16]
    expected_perimeter = [16]
    expected_total_area = 500

    workspace = make_workspace(numpy.zeros_like(pixel_data), parent_image=image)
    workspace.pipeline.set_volumetric(True)
    workspace.image_set.add("MyBinaryImage", image)

    module = workspace.module
    module.operand_choice.value = "Binary Image"
    module.images_list.value = "MyBinaryImage"

    module.run(workspace)

    def mn(x):
        return "AreaOccupied_%s_%s" % (x, module.images_list.value[0])

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement("Image", mn("VolumeOccupied")),
        expected_area,
    )

    numpy.testing.assert_array_almost_equal(
        workspace.measurements.get_current_measurement("Image", mn("SurfaceArea")),
        expected_perimeter,
        decimal=0,
    )

    numpy.testing.assert_array_equal(
        workspace.measurements.get_current_measurement("Image", mn("TotalVolume")),
        expected_total_area,
    )


def test_load_v3():
    file = tests.frontend.modules.get_test_resources_directory(
        "measureimageareaoccupied/v3.pipeline"
    )
    with open(file, "r") as fd:
        data = fd.read()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(callback)
    pipeline.load(six.StringIO(data))

    module = pipeline.modules()[0]

    assert module.operand_choice.value == "Both"
    assert module.images_list.value_text == "DNA"
    assert len(module.objects_list.value) == 2
    assert set(module.objects_list.value) == {"Nuclei", "Cells"}
