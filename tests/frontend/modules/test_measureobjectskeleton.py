import os
import tempfile
import traceback

import numpy
import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import GROUP_NUMBER, GROUP_INDEX, COLTYPE_FLOAT, COLTYPE_INTEGER

import cellprofiler.modules.measureobjectskeleton
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler_core.setting
import cellprofiler_core.workspace
import tests.frontend.modules


IMAGE_NAME = "MyImage"
INTENSITY_IMAGE_NAME = "MyIntensityImage"
OBJECT_NAME = "MyObject"
EDGE_FILE = "my_edges.csv"
VERTEX_FILE = "my_vertices.csv"


temp_dir = tempfile.mkdtemp()


def tearDown():
    if hasattr("temp_dir"):
        for file_name in (EDGE_FILE, VERTEX_FILE):
            p = os.path.join(temp_dir, file_name)
            try:
                if os.path.exists(p):
                    os.remove(p)
            except:
                print(("Failed to remove %s" % p))
                traceback.print_exc()
        os.rmdir(temp_dir)


def test_load_v1():
    file = tests.frontend.modules.get_test_resources_directory("measureobjectskeleton/v1.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[-1]
    assert isinstance(
        module, cellprofiler.modules.measureobjectskeleton.MeasureObjectSkeleton
    )
    assert module.image_name == "DNA"
    assert module.seed_objects_name == "Nucs"
    assert module.wants_branchpoint_image
    assert module.branchpoint_image_name == "BPImg"


def make_workspace(labels, image, mask=None, intensity_image=None, wants_graph=False):
    m = cellprofiler_core.measurement.Measurements()
    image_set_list = cellprofiler_core.image.ImageSetList()
    m.add_measurement(
        "Image",
        GROUP_NUMBER,
        1,
    )
    m.add_measurement(
        "Image",
        GROUP_INDEX,
        1,
    )
    image_set = m
    img = cellprofiler_core.image.Image(image, mask)
    image_set.add(IMAGE_NAME, img)

    object_set = cellprofiler_core.object.ObjectSet()
    o = cellprofiler_core.object.Objects()
    o.segmented = labels
    object_set.add_objects(o, OBJECT_NAME)

    module = cellprofiler.modules.measureobjectskeleton.MeasureObjectSkeleton()
    module.image_name.value = IMAGE_NAME
    module.seed_objects_name.value = OBJECT_NAME
    if intensity_image is not None:
        img = cellprofiler_core.image.Image(intensity_image)
        image_set.add(INTENSITY_IMAGE_NAME, img)
        module.intensity_image_name.value = INTENSITY_IMAGE_NAME
    if wants_graph:
        module.wants_objskeleton_graph.value = True
        module.directory.dir_choice = cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = temp_dir
        module.edge_file_name.value = EDGE_FILE
        module.vertex_file_name.value = VERTEX_FILE
    module.set_module_num(1)

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, module, image_set, object_set, m, image_set_list
    )
    return workspace, module


def test_empty():
    workspace, module = make_workspace(
        numpy.zeros((20, 10), int), numpy.zeros((20, 10), bool)
    )
    #
    # Make sure module tells us about the measurements
    #
    columns = module.get_measurement_columns(None)
    features = [c[1] for c in columns]
    features.sort()
    expected = cellprofiler.modules.measureobjectskeleton.F_ALL
    expected.sort()
    coltypes = {}
    for feature, expected in zip(features, expected):
        expected_feature = "_".join(
            (
                cellprofiler.modules.measureobjectskeleton.C_OBJSKELETON,
                expected,
                IMAGE_NAME,
            )
        )
        assert feature == expected_feature
        coltypes[expected_feature] = (
            COLTYPE_FLOAT
            if expected
            == cellprofiler.modules.measureobjectskeleton.F_TOTAL_OBJSKELETON_LENGTH
            else COLTYPE_INTEGER
        )
    assert all([c[0] == OBJECT_NAME for c in columns])
    assert all([c[2] == coltypes[c[1]] for c in columns])

    categories = module.get_categories(None, OBJECT_NAME)
    assert len(categories) == 1
    assert categories[0] == cellprofiler.modules.measureobjectskeleton.C_OBJSKELETON
    assert len(module.get_categories(None, "Foo")) == 0

    measurements = module.get_measurements(
        None, OBJECT_NAME, cellprofiler.modules.measureobjectskeleton.C_OBJSKELETON
    )
    assert len(measurements) == len(cellprofiler.modules.measureobjectskeleton.F_ALL)
    assert measurements[0] != measurements[1]
    assert all(
        [m in cellprofiler.modules.measureobjectskeleton.F_ALL for m in measurements]
    )

    assert (
        len(
            module.get_measurements(
                None, "Foo", cellprofiler.modules.measureobjectskeleton.C_OBJSKELETON
            )
        )
        == 0
    )
    assert len(module.get_measurements(None, OBJECT_NAME, "Foo")) == 0

    for feature in cellprofiler.modules.measureobjectskeleton.F_ALL:
        images = module.get_measurement_images(
            None,
            OBJECT_NAME,
            cellprofiler.modules.measureobjectskeleton.C_OBJSKELETON,
            feature,
        )
        assert len(images) == 1
        assert images[0] == IMAGE_NAME

    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    for feature in cellprofiler.modules.measureobjectskeleton.F_ALL:
        mname = "_".join(
            (
                cellprofiler.modules.measureobjectskeleton.C_OBJSKELETON,
                expected,
                IMAGE_NAME,
            )
        )
        data = m.get_current_measurement(OBJECT_NAME, mname)
        assert len(data) == 0


def test_trunk():
    """Create an image with one soma with one neurite"""
    image = numpy.zeros((20, 15), bool)
    image[9, 5:] = True
    labels = numpy.zeros((20, 15), int)
    labels[6:12, 2:8] = 1
    workspace, module = make_workspace(labels, image)
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    for feature, expected in (
        (cellprofiler.modules.measureobjectskeleton.F_NUMBER_NON_TRUNK_BRANCHES, 0),
        (cellprofiler.modules.measureobjectskeleton.F_NUMBER_TRUNKS, 1),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureobjectskeleton.C_OBJSKELETON,
                feature,
                IMAGE_NAME,
            )
        )
        data = m.get_current_measurement(OBJECT_NAME, mname)
        assert len(data) == 1
        assert data[0] == expected


def test_trunks():
    """Create an image with two soma and a neurite that goes through both"""
    image = numpy.zeros((30, 15), bool)
    image[1:25, 7] = True
    labels = numpy.zeros((30, 15), int)
    labels[6:13, 3:10] = 1
    labels[18:26, 3:10] = 2
    workspace, module = make_workspace(labels, image)
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    for feature, expected in (
        (
            cellprofiler.modules.measureobjectskeleton.F_NUMBER_NON_TRUNK_BRANCHES,
            [0, 0],
        ),
        (cellprofiler.modules.measureobjectskeleton.F_NUMBER_TRUNKS, [2, 1]),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureobjectskeleton.C_OBJSKELETON,
                feature,
                IMAGE_NAME,
            )
        )
        data = m.get_current_measurement(OBJECT_NAME, mname)
        assert len(data) == 2
        for i in range(2):
            assert data[i] == expected[i]


def test_branch():
    """Create an image with one soma and a neurite with a branch"""
    image = numpy.zeros((30, 15), bool)
    image[6:15, 7] = True
    image[15 + numpy.arange(3), 7 + numpy.arange(3)] = True
    image[15 + numpy.arange(3), 7 - numpy.arange(3)] = True
    labels = numpy.zeros((30, 15), int)
    labels[1:8, 3:10] = 1
    workspace, module = make_workspace(labels, image)
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    for feature, expected in (
        (cellprofiler.modules.measureobjectskeleton.F_NUMBER_NON_TRUNK_BRANCHES, 1),
        (cellprofiler.modules.measureobjectskeleton.F_NUMBER_TRUNKS, 1),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureobjectskeleton.C_OBJSKELETON,
                feature,
                IMAGE_NAME,
            )
        )
        data = m.get_current_measurement(OBJECT_NAME, mname)
        assert len(data) == 1
        assert data[0] == expected


def test_img_667():
    """Create an image with a one-pixel soma and a neurite with a branch

    Regression test of IMG-667
    """
    image = numpy.zeros((30, 15), bool)
    image[6:15, 7] = True
    image[15 + numpy.arange(3), 7 + numpy.arange(3)] = True
    image[15 + numpy.arange(3), 7 - numpy.arange(3)] = True
    labels = numpy.zeros((30, 15), int)
    labels[10, 7] = 1
    workspace, module = make_workspace(labels, image)
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    for feature, expected in (
        (cellprofiler.modules.measureobjectskeleton.F_NUMBER_NON_TRUNK_BRANCHES, 1),
        (cellprofiler.modules.measureobjectskeleton.F_NUMBER_TRUNKS, 2),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureobjectskeleton.C_OBJSKELETON,
                feature,
                IMAGE_NAME,
            )
        )
        data = m.get_current_measurement(OBJECT_NAME, mname)
        assert len(data) == 1
        assert data[0] == expected, "%s: expected %d, got %d" % (
            feature,
            expected,
            data[0],
        )


def test_quadrabranch():
    """An odd example that I noticed and thought was worthy of a test

    You get this pattern:
            x
            I
        I   I
        I   I
        I   I
            I
        x   x

        And there should be 3 trunks (or possibly two trunks and a branch)
    """
    image = numpy.zeros((30, 15), bool)
    image[6:15, 7] = True
    image[15 + numpy.arange(3), 7 + numpy.arange(3)] = True
    image[15 + numpy.arange(3), 7 - numpy.arange(3)] = True
    labels = numpy.zeros((30, 15), int)
    labels[13, 7] = 1
    workspace, module = make_workspace(labels, image)
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    for feature, expected in (
        (cellprofiler.modules.measureobjectskeleton.F_NUMBER_NON_TRUNK_BRANCHES, 0),
        (cellprofiler.modules.measureobjectskeleton.F_NUMBER_TRUNKS, 3),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureobjectskeleton.C_OBJSKELETON,
                feature,
                IMAGE_NAME,
            )
        )
        data = m.get_current_measurement(OBJECT_NAME, mname)
        assert len(data) == 1
        assert data[0] == expected, "%s: expected %d, got %d" % (
            feature,
            expected,
            data[0],
        )


def test_wrong_size():
    """Regression of img-961, image and labels size differ

    Assume that image is primary, labels outside of image are ignored
    and image outside of labels is unlabeled.
    """
    image = numpy.zeros((40, 15), bool)
    image[1:25, 7] = True
    labels = numpy.zeros((30, 20), int)
    labels[6:13, 3:10] = 1
    labels[18:26, 3:10] = 2
    workspace, module = make_workspace(labels, image)
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    for feature, expected in (
        (
            cellprofiler.modules.measureobjectskeleton.F_NUMBER_NON_TRUNK_BRANCHES,
            [0, 0],
        ),
        (cellprofiler.modules.measureobjectskeleton.F_NUMBER_TRUNKS, [2, 1]),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureobjectskeleton.C_OBJSKELETON,
                feature,
                IMAGE_NAME,
            )
        )
        data = m.get_current_measurement(OBJECT_NAME, mname)
        assert len(data) == 2
        for i in range(2):
            assert data[i] == expected[i]


def test_skeleton_length():
    #
    # Soma ends at x=8, neurite ends at x=15. Length should be 7
    #
    image = numpy.zeros((20, 20), bool)
    image[9, 5:15] = True
    labels = numpy.zeros((20, 20), int)
    labels[6:12, 2:8] = 1
    workspace, module = make_workspace(labels, image)
    module.run(workspace)
    m = workspace.measurements
    ftr = "_".join(
        (
            cellprofiler.modules.measureobjectskeleton.C_OBJSKELETON,
            cellprofiler.modules.measureobjectskeleton.F_TOTAL_OBJSKELETON_LENGTH,
            IMAGE_NAME,
        )
    )
    result = m[OBJECT_NAME, ftr]
    assert len(result) == 1
    assert abs(result[0] - 5) < numpy.sqrt(numpy.finfo(numpy.float32).eps)


def read_graph_file(file_name):
    type_dict = dict(
        image_number="i4",
        v1="i4",
        v2="i4",
        length="i4",
        total_intensity="f8",
        i="i4",
        j="i4",
        vertex_number="i4",
        labels="i4",
        kind="S1",
    )

    path = os.path.join(temp_dir, file_name)
    fd = open(path, "r")
    fields = fd.readline().strip().split(",")
    dt = numpy.dtype(dict(names=fields, formats=[type_dict[x] for x in fields]))
    pos = fd.tell()
    if len(fd.readline()) == 0:
        return numpy.recarray(0, dt)
    fd.seek(pos)
    return numpy.loadtxt(fd, dt, delimiter=",")


def test_graph():
    """Does graph neurons work on an empty image?"""
    workspace, module = make_workspace(
        numpy.zeros((20, 10), int),
        numpy.zeros((20, 10), bool),
        intensity_image=numpy.zeros((20, 10)),
        wants_graph=True,
    )
    module.prepare_run(workspace)
    module.run(workspace)
    edge_graph = read_graph_file(EDGE_FILE)
    vertex_graph = read_graph_file(VERTEX_FILE)
    assert len(edge_graph) == 0
    assert len(vertex_graph) == 0


def test_graph():
    """Make a simple graph"""
    #
    # The skeleton looks something like this:
    #
    #   .   .
    #    . .
    #     .
    #     .
    i, j = numpy.mgrid[-10:11, -10:11]
    skel = (i < 0) & (numpy.abs(i) == numpy.abs(j))
    skel[(i >= 0) & (j == 0)] = True
    #
    # Put a single label at the bottom
    #
    labels = numpy.zeros(skel.shape, int)
    labels[(i > 8) & (numpy.abs(j) < 2)] = 1
    numpy.random.seed(31)
    intensity = numpy.random.uniform(size=skel.shape)
    workspace, module = make_workspace(
        labels, skel, intensity_image=intensity, wants_graph=True
    )
    module.prepare_run(workspace)
    module.run(workspace)
    edge_graph = read_graph_file(EDGE_FILE)
    vertex_graph = read_graph_file(VERTEX_FILE)
    vidx = numpy.lexsort((vertex_graph["j"], vertex_graph["i"]))
    #
    # There should be two vertices at the bottom of the array - these
    # are bogus artifacts of the object hitting the edge of the image
    #
    for vidxx in vidx[-2:]:
        assert vertex_graph["i"][vidxx] == 20
    vidx = vidx[:-2]

    expected_vertices = ((0, 0), (0, 20), (10, 10), (17, 10))
    assert len(vidx) == len(expected_vertices)
    for idx, v in enumerate(expected_vertices):
        vv = vertex_graph[vidx[idx]]
        assert vv["i"] == v[0]
        assert vv["j"] == v[1]

    #
    # Get rid of edges to the bogus vertices
    #
    for v in ("v1", "v2"):
        edge_graph = edge_graph[vertex_graph["i"][edge_graph[v] - 1] != 20]

    eidx = numpy.lexsort(
        (
            vertex_graph["j"][edge_graph["v1"] - 1],
            vertex_graph["i"][edge_graph["v1"] - 1],
            vertex_graph["j"][edge_graph["v2"] - 1],
            vertex_graph["i"][edge_graph["v2"] - 1],
        )
    )
    expected_edges = (
        ((0, 0), (10, 10), 11, numpy.sum(intensity[(i <= 0) & (j <= 0) & skel])),
        ((0, 20), (10, 10), 11, numpy.sum(intensity[(i <= 0) & (j >= 0) & skel])),
        ((10, 10), (17, 10), 8, numpy.sum(intensity[(i >= 0) & (i <= 7) & skel])),
    )
    for i, (v1, v2, length, total_intensity) in enumerate(expected_edges):
        ee = edge_graph[eidx[i]]
        for ve, v in ((v1, ee["v1"]), (v2, ee["v2"])):
            assert ve[0] == vertex_graph["i"][v - 1]
            assert ve[1] == vertex_graph["j"][v - 1]
        assert length == ee["length"]
        assert round(abs(total_intensity - ee["total_intensity"]), 4) == 0
