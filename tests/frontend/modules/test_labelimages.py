import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import M_SITE, M_ROW, M_COLUMN, M_PLATE, M_WELL


import cellprofiler.modules.labelimages
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules


def test_load_v1():
    file = tests.frontend.modules.get_test_resources_directory("labelimages/v1.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 2
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.labelimages.LabelImages)
    assert module.site_count == 3
    assert module.row_count == 32
    assert module.column_count == 48
    assert module.order == cellprofiler.modules.labelimages.O_COLUMN

    module = pipeline.modules()[1]
    assert isinstance(module, cellprofiler.modules.labelimages.LabelImages)
    assert module.site_count == 1
    assert module.row_count == 8
    assert module.column_count == 12
    assert module.order == cellprofiler.modules.labelimages.O_ROW


def make_workspace(image_set_count):
    image_set_list = cellprofiler_core.image.ImageSetList()
    for i in range(image_set_count):
        image_set = image_set_list.get_image_set(i)
    module = cellprofiler.modules.labelimages.LabelImages()
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    module.set_module_num(1)
    pipeline.add_module(module)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set_list.get_image_set(0),
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return workspace, module


def test_label_plate_by_row():
    """Label one complete plate"""
    nsites = 6
    nimagesets = 96 * nsites
    workspace, module = make_workspace(nimagesets)
    measurements = workspace.measurements
    assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
    assert isinstance(module, cellprofiler.modules.labelimages.LabelImages)
    module.row_count.value = 8
    module.column_count.value = 12
    module.order.value = cellprofiler.modules.labelimages.O_ROW
    module.site_count.value = nsites
    for i in range(nimagesets):
        if i != 0:
            measurements.next_image_set()
        module.run(workspace)
    sites = measurements.get_all_measurements(
        "Image", M_SITE
    )
    rows = measurements.get_all_measurements(
        "Image", M_ROW
    )
    columns = measurements.get_all_measurements(
        "Image", M_COLUMN
    )
    plates = measurements.get_all_measurements(
        "Image", M_PLATE
    )
    wells = measurements.get_all_measurements(
        "Image", M_WELL
    )
    for i in range(nimagesets):
        assert sites[i] == (i % 6) + 1
        this_row = "ABCDEFGH"[int(i / 6 / 12)]
        this_column = (int(i / 6) % 12) + 1
        assert rows[i] == this_row
        assert columns[i] == this_column
        assert wells[i] == "%s%02d" % (this_row, this_column)
        assert plates[i] == 1


def test_label_plate_by_column():
    """Label one complete plate"""
    nsites = 6
    nimagesets = 96 * nsites
    workspace, module = make_workspace(nimagesets)
    measurements = workspace.measurements
    assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
    assert isinstance(module, cellprofiler.modules.labelimages.LabelImages)
    module.row_count.value = 8
    module.column_count.value = 12
    module.order.value = cellprofiler.modules.labelimages.O_COLUMN
    module.site_count.value = nsites
    for i in range(nimagesets):
        if i != 0:
            measurements.next_image_set()
        module.run(workspace)
    sites = measurements.get_all_measurements(
        "Image", M_SITE
    )
    rows = measurements.get_all_measurements(
        "Image", M_ROW
    )
    columns = measurements.get_all_measurements(
        "Image", M_COLUMN
    )
    plates = measurements.get_all_measurements(
        "Image", M_PLATE
    )
    wells = measurements.get_all_measurements(
        "Image", M_WELL
    )
    for i in range(nimagesets):
        assert sites[i] == (i % 6) + 1
        this_row = "ABCDEFGH"[int(i / 6) % 8]
        this_column = int(i / 6 / 8) + 1
        assert rows[i] == this_row
        assert columns[i] == this_column
        assert wells[i] == "%s%02d" % (this_row, this_column)
        assert plates[i] == 1


def test_label_many_plates():
    nsites = 1
    nplates = 6
    nimagesets = 96 * nsites * nplates
    workspace, module = make_workspace(nimagesets)
    measurements = workspace.measurements
    assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
    assert isinstance(module, cellprofiler.modules.labelimages.LabelImages)
    module.row_count.value = 8
    module.column_count.value = 12
    module.order.value = cellprofiler.modules.labelimages.O_ROW
    module.site_count.value = nsites
    for i in range(nimagesets):
        if i != 0:
            measurements.next_image_set()
        module.run(workspace)
    sites = measurements.get_all_measurements(
        "Image", M_SITE
    )
    rows = measurements.get_all_measurements(
        "Image", M_ROW
    )
    columns = measurements.get_all_measurements(
        "Image", M_COLUMN
    )
    plates = measurements.get_all_measurements(
        "Image", M_PLATE
    )
    wells = measurements.get_all_measurements(
        "Image", M_WELL
    )
    for i in range(nimagesets):
        assert sites[i] == 1
        this_row = "ABCDEFGH"[int(i / 12) % 8]
        this_column = (i % 12) + 1
        assert rows[i] == this_row
        assert columns[i] == this_column
        assert wells[i] == "%s%02d" % (this_row, this_column)
        assert plates[i] == int(i / 8 / 12) + 1


def test_multichar_row_names():
    nimagesets = 1000
    workspace, module = make_workspace(nimagesets)
    measurements = workspace.measurements
    assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
    assert isinstance(module, cellprofiler.modules.labelimages.LabelImages)
    module.row_count.value = 1000
    module.column_count.value = 1
    module.order.value = cellprofiler.modules.labelimages.O_ROW
    module.site_count.value = 1
    for i in range(nimagesets):
        if i != 0:
            measurements.next_image_set()
        module.run(workspace)
    sites = measurements.get_all_measurements(
        "Image", M_SITE
    )
    rows = measurements.get_all_measurements(
        "Image", M_ROW
    )
    columns = measurements.get_all_measurements(
        "Image", M_COLUMN
    )
    plates = measurements.get_all_measurements(
        "Image", M_PLATE
    )
    wells = measurements.get_all_measurements(
        "Image", M_WELL
    )
    abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(nimagesets):
        assert sites[i] == 1
        this_row = abc[int(i / 26 / 26)] + abc[int(i / 26) % 26] + abc[i % 26]
        assert rows[i] == this_row
