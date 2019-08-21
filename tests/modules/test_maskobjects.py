import numpy as np
from six.moves import StringIO

import cellprofiler.measurement
from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.maskobjects as M

INPUT_OBJECTS = "inputobjects"
OUTPUT_OBJECTS = "outputobjects"
MASKING_OBJECTS = "maskingobjects"
MASKING_IMAGE = "maskingobjects"
OUTPUT_OUTLINES = "outputoutlines"


def test_load_v1():
    data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9193

MaskObjects:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
Objects to be masked\x3A:Nuclei
Name the masked objects\x3A:MaskedNuclei
Mask using other objects or binary image?:Objects
Masking objects\x3A:Wells
Masking image\x3A:None
How do you want to handle objects that are partially masked?:Keep overlapping region
How much of the object must overlap?:0.5
Retain original numbering or renumber objects?:Renumber
Save outlines for the resulting objects?:No
Outlines name\x3A:MaskedOutlines

MaskObjects:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
Objects to be masked\x3A:Cells
Name the masked objects\x3A:MaskedCells
Mask using other objects or binary image?:Image
Masking objects\x3A:None
Masking image\x3A:WellBoundary
How do you want to handle objects that are partially masked?:Keep
How much of the object must overlap?:0.5
Retain original numbering or renumber objects?:Retain
Save outlines for the resulting objects?:Yes
Outlines name\x3A:MaskedCellOutlines

MaskObjects:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
Objects to be masked\x3A:Cytoplasm
Name the masked objects\x3A:MaskedCytoplasm
Mask using other objects or binary image?:Objects
Masking objects\x3A:Cells
Masking image\x3A:None
How do you want to handle objects that are partially masked?:Remove
How much of the object must overlap?:0.5
Retain original numbering or renumber objects?:Renumber
Save outlines for the resulting objects?:No
Outlines name\x3A:MaskedOutlines

MaskObjects:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
Objects to be masked\x3A:Speckles
Name the masked objects\x3A:MaskedSpeckles
Mask using other objects or binary image?:Objects
Masking objects\x3A:Cells
Masking image\x3A:None
How do you want to handle objects that are partially masked?:Remove depending on overlap
How much of the object must overlap?:0.3
Retain original numbering or renumber objects?:Renumber
Save outlines for the resulting objects?:No
Outlines name\x3A:MaskedOutlines
"""
    pipeline = cpp.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cpp.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(StringIO(data))
    assert len(pipeline.modules()) == 4

    module = pipeline.modules()[0]
    assert isinstance(module, M.MaskObjects)
    assert module.object_name.value == "Nuclei"
    assert module.mask_choice.value == M.MC_OBJECTS
    assert module.masking_objects.value == "Wells"
    assert module.remaining_objects.value == "MaskedNuclei"
    assert module.retain_or_renumber.value == M.R_RENUMBER
    assert module.overlap_choice.value == M.P_MASK
    assert not module.wants_inverted_mask

    module = pipeline.modules()[1]
    assert isinstance(module, M.MaskObjects)
    assert module.object_name.value == "Cells"
    assert module.mask_choice.value == M.MC_IMAGE
    assert module.masking_image.value == "WellBoundary"
    assert module.remaining_objects.value == "MaskedCells"
    assert module.retain_or_renumber.value == M.R_RETAIN
    assert module.overlap_choice.value == M.P_KEEP
    assert not module.wants_inverted_mask

    module = pipeline.modules()[2]
    assert isinstance(module, M.MaskObjects)
    assert module.object_name.value == "Cytoplasm"
    assert module.mask_choice.value == M.MC_OBJECTS
    assert module.masking_objects.value == "Cells"
    assert module.remaining_objects.value == "MaskedCytoplasm"
    assert module.retain_or_renumber.value == M.R_RENUMBER
    assert module.overlap_choice.value == M.P_REMOVE
    assert not module.wants_inverted_mask

    module = pipeline.modules()[3]
    assert isinstance(module, M.MaskObjects)
    assert module.object_name.value == "Speckles"
    assert module.mask_choice.value == M.MC_OBJECTS
    assert module.masking_objects.value == "Cells"
    assert module.remaining_objects.value == "MaskedSpeckles"
    assert module.retain_or_renumber.value == M.R_RENUMBER
    assert module.overlap_choice.value == M.P_REMOVE_PERCENTAGE
    assert round(abs(module.overlap_fraction.value - 0.3), 7) == 0
    assert not module.wants_inverted_mask


def test_load_v2():
    data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9193

MaskObjects:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
Objects to be masked\x3A:Nuclei
Name the masked objects\x3A:MaskedNuclei
Mask using other objects or binary image?:Objects
Masking objects\x3A:Wells
Masking image\x3A:None
How do you want to handle objects that are partially masked?:Keep overlapping region
How much of the object must overlap?:0.5
Retain original numbering or renumber objects?:Renumber
Save outlines for the resulting objects?:No
Outlines name\x3A:MaskedOutlines
Invert the mask?:Yes
"""
    pipeline = cpp.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cpp.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(StringIO(data))
    assert len(pipeline.modules()) == 1

    module = pipeline.modules()[0]
    assert isinstance(module, M.MaskObjects)
    assert module.object_name.value == "Nuclei"
    assert module.mask_choice.value == M.MC_OBJECTS
    assert module.masking_objects.value == "Wells"
    assert module.remaining_objects.value == "MaskedNuclei"
    assert module.retain_or_renumber.value == M.R_RENUMBER
    assert module.overlap_choice.value == M.P_MASK
    assert module.wants_inverted_mask


def test_load_v3():
    data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20150319195827
GitHash:d8289bf
ModuleCount:1
HasImagePlaneDetails:False

MaskObjects:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
Select objects to be masked:IdentifyPrimaryObjects
Name the masked objects:MaskObjects
Mask using a region defined by other objects or by binary image?:Objects
Select the masking object:FilterObjects
Select the masking image:None
Handling of objects that are partially masked:Keep overlapping region
Fraction of object that must overlap:0.5
Numbering of resulting objects:Renumber
Invert the mask?:No
"""
    pipeline = cpp.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cpp.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.loadtxt(StringIO(data))
    module = pipeline.modules()[0]

    assert module.object_name.value == "IdentifyPrimaryObjects"
    assert module.remaining_objects.value == "MaskObjects"
    assert module.mask_choice.value == "Objects"
    assert module.masking_objects.value == "FilterObjects"
    assert module.masking_image.value == "None"
    assert module.overlap_choice.value == "Keep overlapping region"
    assert module.overlap_fraction.value == 0.5
    assert module.retain_or_renumber.value == "Renumber"
    assert not module.wants_inverted_mask.value


def make_workspace(
    labels, overlap_choice, masking_objects=None, masking_image=None, renumber=True
):
    module = M.MaskObjects()
    module.set_module_num(1)
    module.object_name.value = INPUT_OBJECTS
    module.remaining_objects.value = OUTPUT_OBJECTS
    module.mask_choice.value = (
        M.MC_OBJECTS if masking_objects is not None else M.MC_IMAGE
    )
    module.masking_image.value = MASKING_IMAGE
    module.masking_objects.value = MASKING_OBJECTS
    module.retain_or_renumber.value = M.R_RENUMBER if renumber else M.R_RETAIN
    module.overlap_choice.value = overlap_choice

    pipeline = cpp.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cpp.RunExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.add_module(module)

    object_set = cpo.ObjectSet()
    io = cpo.Objects()
    io.segmented = labels
    object_set.add_objects(io, INPUT_OBJECTS)

    if masking_objects is not None:
        oo = cpo.Objects()
        oo.segmented = masking_objects
        object_set.add_objects(oo, MASKING_OBJECTS)

    image_set_list = cpi.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    if masking_image is not None:
        mi = cpi.Image(masking_image)
        image_set.add(MASKING_IMAGE, mi)

    workspace = cpw.Workspace(
        pipeline, module, image_set, object_set, cpmeas.Measurements(), image_set_list
    )
    return workspace, module


def test_measurement_columns():
    """Test get_measurement_columns"""
    workspace, module = make_workspace(
        np.zeros((20, 10), int), M.P_MASK, np.zeros((20, 10), int)
    )
    columns = module.get_measurement_columns(workspace.pipeline)
    assert len(columns) == 6
    for expected in (
        (
            cpmeas.IMAGE,
            cellprofiler.measurement.FF_COUNT % OUTPUT_OBJECTS,
            cpmeas.COLTYPE_INTEGER,
        ),
        (
            OUTPUT_OBJECTS,
            cellprofiler.measurement.M_LOCATION_CENTER_X,
            cpmeas.COLTYPE_FLOAT,
        ),
        (
            OUTPUT_OBJECTS,
            cellprofiler.measurement.M_LOCATION_CENTER_Y,
            cpmeas.COLTYPE_FLOAT,
        ),
        (
            OUTPUT_OBJECTS,
            cellprofiler.measurement.FF_PARENT % INPUT_OBJECTS,
            cpmeas.COLTYPE_INTEGER,
        ),
        (
            OUTPUT_OBJECTS,
            cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
            cpmeas.COLTYPE_INTEGER,
        ),
        (
            INPUT_OBJECTS,
            cellprofiler.measurement.FF_CHILDREN_COUNT % OUTPUT_OBJECTS,
            cpmeas.COLTYPE_INTEGER,
        ),
    ):
        assert any(
            [all([c in e for c, e in zip(column, expected)]) for column in columns]
        )


def test_measurement_categories():
    workspace, module = make_workspace(
        np.zeros((20, 10), int), M.MC_OBJECTS, np.zeros((20, 10), int)
    )
    categories = module.get_categories(workspace.pipeline, "Foo")
    assert len(categories) == 0

    categories = module.get_categories(workspace.pipeline, cpmeas.IMAGE)
    assert len(categories) == 1
    assert categories[0] == cellprofiler.measurement.C_COUNT

    categories = module.get_categories(workspace.pipeline, OUTPUT_OBJECTS)
    assert len(categories) == 3
    for category, expected in zip(
        sorted(categories),
        (
            cellprofiler.measurement.C_LOCATION,
            cellprofiler.measurement.C_NUMBER,
            cellprofiler.measurement.C_PARENT,
        ),
    ):
        assert category == expected

    categories = module.get_categories(workspace.pipeline, INPUT_OBJECTS)
    assert len(categories) == 1
    assert categories[0] == cellprofiler.measurement.C_CHILDREN


def test_measurements():
    workspace, module = make_workspace(
        np.zeros((20, 10), int), M.P_MASK, np.zeros((20, 10), int)
    )
    ftr_count = (cellprofiler.measurement.FF_CHILDREN_COUNT % OUTPUT_OBJECTS).split(
        "_", 1
    )[1]
    d = {
        "Foo": {},
        cpmeas.IMAGE: {"Foo": [], cellprofiler.measurement.C_COUNT: [OUTPUT_OBJECTS]},
        OUTPUT_OBJECTS: {
            "Foo": [],
            cellprofiler.measurement.C_LOCATION: [
                cellprofiler.measurement.FTR_CENTER_X,
                cellprofiler.measurement.FTR_CENTER_Y,
            ],
            cellprofiler.measurement.C_PARENT: [INPUT_OBJECTS],
            cellprofiler.measurement.C_NUMBER: [
                cellprofiler.measurement.FTR_OBJECT_NUMBER
            ],
        },
        INPUT_OBJECTS: {"Foo": [], cellprofiler.measurement.C_CHILDREN: [ftr_count]},
    }
    for object_name in list(d.keys()):
        od = d[object_name]
        for category in list(od.keys()):
            features = module.get_measurements(
                workspace.pipeline, object_name, category
            )
            expected = od[category]
            assert len(features) == len(expected)
            for feature, e in zip(sorted(features), sorted(expected)):
                assert feature == e


def test_mask_nothing():
    workspace, module = make_workspace(
        np.zeros((20, 10), int), M.MC_OBJECTS, np.zeros((20, 10), int)
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m, cpmeas.Measurements)
    value = m.get_current_image_measurement(
        cellprofiler.measurement.FF_COUNT % OUTPUT_OBJECTS
    )
    assert value == 0
    for object_name, feature in (
        (OUTPUT_OBJECTS, cellprofiler.measurement.M_LOCATION_CENTER_X),
        (OUTPUT_OBJECTS, cellprofiler.measurement.M_LOCATION_CENTER_Y),
        (OUTPUT_OBJECTS, cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER),
        (OUTPUT_OBJECTS, cellprofiler.measurement.FF_PARENT % INPUT_OBJECTS),
        (INPUT_OBJECTS, cellprofiler.measurement.FF_CHILDREN_COUNT % OUTPUT_OBJECTS),
    ):
        data = m.get_current_measurement(object_name, feature)
        assert len(data) == 0
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert np.all(objects.segmented == 0)


def test_mask_with_objects():
    labels = np.zeros((20, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 2
    mask = np.zeros((20, 10), int)
    mask[3:17, 2:6] = 1
    expected = labels.copy()
    expected[mask == 0] = 0
    workspace, module = make_workspace(labels, M.P_MASK, mask)
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert np.all(objects.segmented == expected)

    expected_x = np.array([4, 4])
    expected_y = np.array([5, 14])
    m = workspace.measurements
    assert isinstance(m, cpmeas.Measurements)
    value = m.get_current_image_measurement(
        cellprofiler.measurement.FF_COUNT % OUTPUT_OBJECTS
    )
    assert value == 2

    for object_name, feature, expected in (
        (OUTPUT_OBJECTS, cellprofiler.measurement.M_LOCATION_CENTER_X, expected_x),
        (OUTPUT_OBJECTS, cellprofiler.measurement.M_LOCATION_CENTER_Y, expected_y),
        (
            OUTPUT_OBJECTS,
            cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
            np.array([1, 2]),
        ),
        (
            OUTPUT_OBJECTS,
            cellprofiler.measurement.FF_PARENT % INPUT_OBJECTS,
            np.array([1, 2]),
        ),
        (
            INPUT_OBJECTS,
            cellprofiler.measurement.FF_CHILDREN_COUNT % OUTPUT_OBJECTS,
            np.array([1, 1]),
        ),
    ):
        data = m.get_current_measurement(object_name, feature)
        assert len(data) == len(expected)
        for value, e in zip(data, expected):
            assert value == e


def test_mask_with_image():
    labels = np.zeros((20, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 2
    mask = np.zeros((20, 10), bool)
    mask[3:17, 2:6] = True
    expected = labels.copy()
    expected[~mask] = 0
    workspace, module = make_workspace(labels, M.P_MASK, masking_image=mask)
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert np.all(objects.segmented == expected)

    expected_x = np.array([4, 4])
    expected_y = np.array([5, 14])
    m = workspace.measurements
    assert isinstance(m, cpmeas.Measurements)
    value = m.get_current_image_measurement(
        cellprofiler.measurement.FF_COUNT % OUTPUT_OBJECTS
    )
    assert value == 2

    for object_name, feature, expected in (
        (OUTPUT_OBJECTS, cellprofiler.measurement.M_LOCATION_CENTER_X, expected_x),
        (OUTPUT_OBJECTS, cellprofiler.measurement.M_LOCATION_CENTER_Y, expected_y),
        (
            OUTPUT_OBJECTS,
            cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
            np.array([1, 2]),
        ),
        (
            OUTPUT_OBJECTS,
            cellprofiler.measurement.FF_PARENT % INPUT_OBJECTS,
            np.array([1, 2]),
        ),
        (
            INPUT_OBJECTS,
            cellprofiler.measurement.FF_CHILDREN_COUNT % OUTPUT_OBJECTS,
            np.array([1, 1]),
        ),
    ):
        data = m.get_current_measurement(object_name, feature)
        assert len(data) == len(expected)
        for value, e in zip(data, expected):
            assert value == e


def test_mask_renumber():
    labels = np.zeros((30, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 3
    labels[22:28, 3:7] = 2
    mask = np.zeros((30, 10), bool)
    mask[3:17, 2:6] = True
    expected = labels.copy()
    expected[~mask] = 0
    expected[expected == 3] = 2
    workspace, module = make_workspace(labels, M.P_MASK, masking_image=mask)
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert np.all(objects.segmented == expected)

    expected_x = np.array([4, 4])
    expected_y = np.array([5, 14])
    m = workspace.measurements
    assert isinstance(m, cpmeas.Measurements)
    value = m.get_current_image_measurement(
        cellprofiler.measurement.FF_COUNT % OUTPUT_OBJECTS
    )
    assert value == 2

    for object_name, feature, expected in (
        (OUTPUT_OBJECTS, cellprofiler.measurement.M_LOCATION_CENTER_X, expected_x),
        (OUTPUT_OBJECTS, cellprofiler.measurement.M_LOCATION_CENTER_Y, expected_y),
        (
            OUTPUT_OBJECTS,
            cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
            np.array([1, 2]),
        ),
        (
            OUTPUT_OBJECTS,
            cellprofiler.measurement.FF_PARENT % INPUT_OBJECTS,
            np.array([1, 3]),
        ),
        (
            INPUT_OBJECTS,
            cellprofiler.measurement.FF_CHILDREN_COUNT % OUTPUT_OBJECTS,
            np.array([1, 0, 1]),
        ),
    ):
        data = m.get_current_measurement(object_name, feature)
        assert len(data) == len(expected)
        for value, e in zip(data, expected):
            assert value == e


def test_mask_retain():
    labels = np.zeros((30, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 3
    labels[22:28, 3:7] = 2
    mask = np.zeros((30, 10), bool)
    mask[3:17, 2:6] = True
    expected = labels.copy()
    expected[~mask] = 0
    workspace, module = make_workspace(
        labels, M.P_MASK, masking_image=mask, renumber=False
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert np.all(objects.segmented == expected)

    expected_x = np.array([4, None, 4])
    expected_y = np.array([5, None, 14])
    m = workspace.measurements
    assert isinstance(m, cpmeas.Measurements)
    value = m.get_current_image_measurement(
        cellprofiler.measurement.FF_COUNT % OUTPUT_OBJECTS
    )
    assert value == 3

    for object_name, feature, expected in (
        (OUTPUT_OBJECTS, cellprofiler.measurement.M_LOCATION_CENTER_X, expected_x),
        (OUTPUT_OBJECTS, cellprofiler.measurement.M_LOCATION_CENTER_Y, expected_y),
        (
            OUTPUT_OBJECTS,
            cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
            np.array([1, 2, 3]),
        ),
        (
            OUTPUT_OBJECTS,
            cellprofiler.measurement.FF_PARENT % INPUT_OBJECTS,
            np.array([1, 2, 3]),
        ),
        (
            INPUT_OBJECTS,
            cellprofiler.measurement.FF_CHILDREN_COUNT % OUTPUT_OBJECTS,
            np.array([1, 0, 1]),
        ),
    ):
        data = m.get_current_measurement(object_name, feature)
        assert len(data) == len(expected)
        for value, e in zip(data, expected):
            if e is None:
                assert np.isnan(value)
            else:
                assert value == e


def test_mask_invert():
    labels = np.zeros((20, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 2
    #
    # Make a mask that covers only object # 1 and that is missing
    # one pixel of that object. Invert it in anticipation of invert op
    #
    mask = labels == 1
    mask[2, 3] = False
    mask = ~mask
    expected = labels
    expected[labels != 1] = 0
    expected[2, 3] = 0
    workspace, module = make_workspace(labels, M.P_MASK, masking_image=mask)
    assert isinstance(module, M.MaskObjects)
    module.wants_inverted_mask.value = True
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert np.all(objects.segmented == expected)


def test_keep():
    labels = np.zeros((30, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 3
    labels[22:28, 3:7] = 2
    mask = np.zeros((30, 10), bool)
    mask[3:17, 2:6] = True
    expected = labels.copy()
    expected[expected == 2] = 0
    expected[expected == 3] = 2
    workspace, module = make_workspace(labels, M.P_KEEP, masking_image=mask)
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert np.all(objects.segmented == expected)


def test_remove():
    labels = np.zeros((20, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 2
    mask = np.zeros((20, 10), bool)
    mask[2:17, 2:7] = True
    expected = labels.copy()
    expected[labels == 2] = 0
    workspace, module = make_workspace(labels, M.P_REMOVE, masking_image=mask)
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert np.all(objects.segmented == expected)


def test_remove_percent():
    labels = np.zeros((20, 10), int)
    labels[2:8, 3:6] = 1
    labels[12:18, 3:7] = 2
    mask = np.zeros((20, 10), bool)
    mask[3:17, 2:6] = True
    # loses 3 of 18 from object 1 = .1666
    # loses 9 of 24 from object 2 = .375
    expected = labels.copy()
    expected[labels == 2] = 0
    workspace, module = make_workspace(
        labels, M.P_REMOVE_PERCENTAGE, masking_image=mask
    )
    module.overlap_fraction.value = 0.75
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert np.all(objects.segmented == expected)


def test_different_object_sizes():
    labels = np.zeros((30, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 2
    mask = np.zeros((20, 20), int)
    mask[3:17, 2:6] = 1
    expected = labels.copy()
    expected[:20, :][mask[:, :10] == 0] = 0
    workspace, module = make_workspace(labels, M.P_MASK, mask)
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert np.all(objects.segmented == expected)


def test_different_image_sizes():
    labels = np.zeros((30, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 2
    mask = np.zeros((20, 20), bool)
    mask[3:17, 2:6] = 1
    expected = labels.copy()
    expected[:20, :][mask[:, :10] == 0] = 0
    workspace, module = make_workspace(labels, M.P_MASK, masking_image=mask)
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert np.all(objects.segmented == expected)
