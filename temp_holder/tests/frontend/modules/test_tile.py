import numpy
import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler.modules.tile
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

INPUT_IMAGE_NAME = "inputimage"
OUTPUT_IMAGE_NAME = "outputimage"


def input_image_name(index):
    return INPUT_IMAGE_NAME + str(index + 1)


def test_load_v1():
    file = tests.frontend.modules.get_test_resources_directory("tile/v1.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.tile.Tile)
    assert module.input_image == "ResizedColorImage"
    assert module.output_image == "TiledImage"
    assert module.tile_method == cellprofiler.modules.tile.T_ACROSS_CYCLES
    assert module.rows == 2
    assert module.columns == 12
    assert module.wants_automatic_rows
    assert not module.wants_automatic_columns
    assert module.place_first == cellprofiler.modules.tile.P_TOP_LEFT
    assert module.tile_style == cellprofiler.modules.tile.S_ROW
    assert not module.meander
    assert len(module.additional_images) == 3
    for g, expected in zip(
        module.additional_images, ("Cytoplasm", "ColorImage", "DNA")
    ):
        assert g.input_image_name == expected


def make_tile_workspace(images):
    module = cellprofiler.modules.tile.Tile()
    module.set_module_num(1)
    module.tile_method.value = cellprofiler.modules.tile.T_ACROSS_CYCLES
    module.input_image.value = INPUT_IMAGE_NAME
    module.output_image.value = OUTPUT_IMAGE_NAME

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    image_set_list = cellprofiler_core.image.ImageSetList()
    for i, image in enumerate(images):
        image_set = image_set_list.get_image_set(i)
        image_set.add(INPUT_IMAGE_NAME, cellprofiler_core.image.Image(image))

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set_list.get_image_set(0),
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return workspace, module


def test_manual_rows_and_columns():
    numpy.random.seed(0)
    images = [
        numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)
    ]
    workspace, module = make_tile_workspace(images)
    assert isinstance(module, cellprofiler.modules.tile.Tile)
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    module.wants_automatic_columns.value = False
    module.wants_automatic_rows.value = False
    module.rows.value = 6
    module.columns.value = 16
    module.tile_style.value = cellprofiler.modules.tile.S_ROW

    module.prepare_group(workspace, (), numpy.arange(1, 97))

    for i in range(96):
        workspace.set_image_set_for_testing_only(i)
        module.run(workspace)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape[0] == 6 * 20
    assert pixel_data.shape[1] == 16 * 10
    for i, image in enumerate(images):
        ii = int(i / 16)
        jj = i % 16
        iii = ii * 20
        jjj = jj * 10
        assert numpy.all(pixel_data[iii : (iii + 20), jjj : (jjj + 10)] == image)


def test_automatic_rows():
    numpy.random.seed(1)
    images = [
        numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)
    ]
    workspace, module = make_tile_workspace(images)
    assert isinstance(module, cellprofiler.modules.tile.Tile)
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    module.wants_automatic_columns.value = False
    module.wants_automatic_rows.value = True
    module.rows.value = 8
    module.columns.value = 16
    module.tile_style.value = cellprofiler.modules.tile.S_ROW

    module.prepare_group(workspace, (), numpy.arange(1, 97))

    for i in range(96):
        workspace.set_image_set_for_testing_only(i)
        module.run(workspace)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape[0] == 6 * 20
    assert pixel_data.shape[1] == 16 * 10
    for i, image in enumerate(images):
        ii = int(i / 16)
        jj = i % 16
        iii = ii * 20
        jjj = jj * 10
        assert numpy.all(pixel_data[iii : (iii + 20), jjj : (jjj + 10)] == image)


def test_automatic_columns():
    numpy.random.seed(2)
    images = [
        numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)
    ]
    workspace, module = make_tile_workspace(images)
    assert isinstance(module, cellprofiler.modules.tile.Tile)
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    module.wants_automatic_columns.value = True
    module.wants_automatic_rows.value = False
    module.rows.value = 6
    module.columns.value = 365
    module.tile_style.value = cellprofiler.modules.tile.S_ROW

    module.prepare_group(workspace, (), numpy.arange(1, 97))

    for i in range(96):
        workspace.set_image_set_for_testing_only(i)
        module.run(workspace)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape[0] == 6 * 20
    assert pixel_data.shape[1] == 16 * 10
    for i, image in enumerate(images):
        ii = int(i / 16)
        jj = i % 16
        iii = ii * 20
        jjj = jj * 10
        assert numpy.all(pixel_data[iii : (iii + 20), jjj : (jjj + 10)] == image)


def test_automatic_rows_and_columns():
    numpy.random.seed(3)
    images = [
        numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)
    ]
    workspace, module = make_tile_workspace(images)
    assert isinstance(module, cellprofiler.modules.tile.Tile)
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    module.wants_automatic_columns.value = True
    module.wants_automatic_rows.value = True
    module.rows.value = 365
    module.columns.value = 24
    module.tile_style.value = cellprofiler.modules.tile.S_ROW

    module.prepare_group(workspace, (), numpy.arange(1, 97))
    for i in range(96):
        workspace.set_image_set_for_testing_only(i)
        module.run(workspace)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape[0] == 9 * 20
    assert pixel_data.shape[1] == 11 * 10
    for i, image in enumerate(images):
        ii = int(i / 11)
        jj = i % 11
        iii = ii * 20
        jjj = jj * 10
        assert numpy.all(pixel_data[iii : (iii + 20), jjj : (jjj + 10)] == image)


def test_color():
    numpy.random.seed(4)
    images = [
        numpy.random.uniform(size=(20, 10, 3)).astype(numpy.float32) for i in range(96)
    ]
    workspace, module = make_tile_workspace(images)
    assert isinstance(module, cellprofiler.modules.tile.Tile)
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    module.wants_automatic_columns.value = False
    module.wants_automatic_rows.value = False
    module.rows.value = 6
    module.columns.value = 16
    module.tile_style.value = cellprofiler.modules.tile.S_ROW

    module.prepare_group(workspace, (), numpy.arange(1, 97))

    for i in range(96):
        workspace.set_image_set_for_testing_only(i)
        module.run(workspace)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape[0] == 6 * 20
    assert pixel_data.shape[1] == 16 * 10
    for i, image in enumerate(images):
        ii = int(i / 16)
        jj = i % 16
        iii = ii * 20
        jjj = jj * 10
        assert numpy.all(pixel_data[iii : (iii + 20), jjj : (jjj + 10), :] == image)


def test_columns_first():
    numpy.random.seed(5)
    images = [
        numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)
    ]
    workspace, module = make_tile_workspace(images)
    assert isinstance(module, cellprofiler.modules.tile.Tile)
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    module.wants_automatic_columns.value = False
    module.wants_automatic_rows.value = False
    module.rows.value = 6
    module.columns.value = 16
    module.tile_style.value = cellprofiler.modules.tile.S_COL

    module.prepare_group(workspace, (), numpy.arange(1, 97))

    for i in range(96):
        workspace.set_image_set_for_testing_only(i)
        module.run(workspace)
    module.post_group(workspace, None)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape[0] == 6 * 20
    assert pixel_data.shape[1] == 16 * 10
    for i, image in enumerate(images):
        ii = i % 6
        jj = int(i / 6)
        iii = ii * 20
        jjj = jj * 10
        assert numpy.all(pixel_data[iii : (iii + 20), jjj : (jjj + 10)] == image)


def test_top_right():
    numpy.random.seed(0)
    images = [
        numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)
    ]
    workspace, module = make_tile_workspace(images)
    assert isinstance(module, cellprofiler.modules.tile.Tile)
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    module.wants_automatic_columns.value = False
    module.wants_automatic_rows.value = False
    module.rows.value = 6
    module.columns.value = 16
    module.tile_style.value = cellprofiler.modules.tile.S_ROW
    module.place_first.value = cellprofiler.modules.tile.P_TOP_RIGHT

    module.prepare_group(workspace, (), numpy.arange(1, 97))

    for i in range(96):
        workspace.set_image_set_for_testing_only(i)
        module.run(workspace)
    module.post_group(workspace, None)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape[0] == 6 * 20
    assert pixel_data.shape[1] == 16 * 10
    for i, image in enumerate(images):
        ii = int(i / 16)
        jj = 15 - (i % 16)
        iii = ii * 20
        jjj = jj * 10
        assert numpy.all(pixel_data[iii : (iii + 20), jjj : (jjj + 10)] == image)


def test_bottom_left():
    numpy.random.seed(8)
    images = [
        numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)
    ]
    workspace, module = make_tile_workspace(images)
    assert isinstance(module, cellprofiler.modules.tile.Tile)
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    module.wants_automatic_columns.value = False
    module.wants_automatic_rows.value = False
    module.rows.value = 6
    module.columns.value = 16
    module.tile_style.value = cellprofiler.modules.tile.S_ROW
    module.place_first.value = cellprofiler.modules.tile.P_BOTTOM_LEFT

    module.prepare_group(workspace, (), numpy.arange(1, 97))

    for i in range(96):
        workspace.set_image_set_for_testing_only(i)
        module.run(workspace)
    module.post_group(workspace, None)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape[0] == 6 * 20
    assert pixel_data.shape[1] == 16 * 10
    for i, image in enumerate(images):
        ii = 5 - int(i / 16)
        jj = i % 16
        iii = ii * 20
        jjj = jj * 10
        assert numpy.all(pixel_data[iii : (iii + 20), jjj : (jjj + 10)] == image)


def test_bottom_right():
    numpy.random.seed(9)
    images = [
        numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)
    ]
    workspace, module = make_tile_workspace(images)
    assert isinstance(module, cellprofiler.modules.tile.Tile)
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    module.wants_automatic_columns.value = False
    module.wants_automatic_rows.value = False
    module.rows.value = 6
    module.columns.value = 16
    module.tile_style.value = cellprofiler.modules.tile.S_ROW
    module.place_first.value = cellprofiler.modules.tile.P_BOTTOM_RIGHT

    module.prepare_group(workspace, (), numpy.arange(1, 97))

    for i in range(96):
        workspace.set_image_set_for_testing_only(i)
        module.run(workspace)
    module.post_group(workspace, None)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape[0] == 6 * 20
    assert pixel_data.shape[1] == 16 * 10
    for i, image in enumerate(images):
        ii = 5 - int(i / 16)
        jj = 15 - (i % 16)
        iii = ii * 20
        jjj = jj * 10
        assert numpy.all(pixel_data[iii : (iii + 20), jjj : (jjj + 10)] == image)


def test_different_sizes():
    numpy.random.seed(10)
    images = [
        numpy.random.uniform(size=(20, 10)).astype(numpy.float32),
        numpy.random.uniform(size=(10, 20)).astype(numpy.float32),
        numpy.random.uniform(size=(40, 5)).astype(numpy.float32),
        numpy.random.uniform(size=(40, 20)).astype(numpy.float32),
    ]
    workspace, module = make_tile_workspace(images)
    assert isinstance(module, cellprofiler.modules.tile.Tile)
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    module.wants_automatic_columns.value = False
    module.wants_automatic_rows.value = False
    module.rows.value = 1
    module.columns.value = 4
    module.tile_style.value = cellprofiler.modules.tile.S_ROW
    module.prepare_group(workspace, (), numpy.arange(1, 4))

    for i in range(4):
        workspace.set_image_set_for_testing_only(i)
        module.run(workspace)
    module.post_group(workspace, None)
    pixel_data = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    assert pixel_data.shape[0] == 20
    assert pixel_data.shape[1] == 40
    assert numpy.all(pixel_data[:, :10] == images[0])
    assert numpy.all(pixel_data[:10, 10:20] == images[1][:, :10])
    assert numpy.all(pixel_data[10:, 10:20] == 0)
    assert numpy.all(pixel_data[:, 20:25] == images[2][:20, :])
    assert numpy.all(pixel_data[:, 25:30] == 0)
    assert numpy.all(pixel_data[:, 30:] == images[3][:20, :10])


def test_filtered():
    numpy.random.seed(9)
    images = [
        numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)
    ]
    workspace, module = make_tile_workspace(images)
    assert isinstance(module, cellprofiler.modules.tile.Tile)
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    module.wants_automatic_columns.value = False
    module.wants_automatic_rows.value = False
    module.rows.value = 6
    module.columns.value = 16
    module.tile_style.value = cellprofiler.modules.tile.S_ROW
    module.place_first.value = cellprofiler.modules.tile.P_BOTTOM_RIGHT

    module.prepare_group(workspace, (), numpy.arange(1, 97))

    for i in range(95):
        workspace.set_image_set_for_testing_only(i)
        module.run(workspace)
    workspace.set_image_set_for_testing_only(95)
    module.post_group(workspace, None)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape[0] == 6 * 20
    assert pixel_data.shape[1] == 16 * 10
    for i, image in enumerate(images[:-1]):
        ii = 5 - int(i / 16)
        jj = 15 - (i % 16)
        iii = ii * 20
        jjj = jj * 10
        assert numpy.all(pixel_data[iii : (iii + 20), jjj : (jjj + 10)] == image)


def make_place_workspace(images):
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    module = cellprofiler.modules.tile.Tile()
    module.set_module_num(1)
    module.tile_method.value = cellprofiler.modules.tile.T_WITHIN_CYCLES
    module.output_image.value = OUTPUT_IMAGE_NAME
    module.wants_automatic_rows.value = False
    module.wants_automatic_columns.value = True
    module.rows.value = 1
    for i, image in enumerate(images):
        image_name = input_image_name(i)
        if i == 0:
            module.input_image.value = image_name
        else:
            if len(module.additional_images) <= i:
                module.add_image()
            module.additional_images[i - 1].input_image_name.value = image_name
        image_set.add(image_name, cellprofiler_core.image.Image(image))

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    pipeline.add_module(module)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return workspace, module


def test_some_images():
    numpy.random.seed(31)
    for i in range(1, 5):
        images = [
            numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for ii in range(i)
        ]
        workspace, module = make_place_workspace(images)
        assert isinstance(module, cellprofiler.modules.tile.Tile)
        assert isinstance(workspace, cellprofiler_core.workspace.Workspace)

        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data
        for j, p in enumerate(images):
            jj = 10 * j
            assert numpy.all(pixel_data[:, jj : (jj + 10)] == p)


def test_mix_color_bw():
    numpy.random.seed(32)
    for color in range(3):
        images = [
            numpy.random.uniform(size=(20, 10, 3) if i == color else (20, 10)).astype(
                numpy.float32
            )
            for i in range(3)
        ]
        workspace, module = make_place_workspace(images)
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data

        for j, p in enumerate(images):
            jj = 10 * j
            if j == color:
                assert numpy.all(pixel_data[:, jj : (jj + 10), :] == p)
            else:
                for k in range(3):
                    assert numpy.all(pixel_data[:, jj : (jj + 10), k] == p)


def test_different_sizes():
    numpy.random.seed(33)
    images = [
        numpy.random.uniform(size=(20, 10)).astype(numpy.float32),
        numpy.random.uniform(size=(10, 20)).astype(numpy.float32),
        numpy.random.uniform(size=(40, 5)).astype(numpy.float32),
        numpy.random.uniform(size=(40, 20)).astype(numpy.float32),
    ]
    workspace, module = make_place_workspace(images)
    module.run(workspace)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape[0] == 40
    assert pixel_data.shape[1] == 80
    mask = numpy.ones(pixel_data.shape, bool)
    assert numpy.all(pixel_data[:20, :10] == images[0])
    mask[:20, :10] = False
    assert numpy.all(pixel_data[:10, 20:40] == images[1])
    mask[:10, 20:40] = False
    assert numpy.all(pixel_data[:, 40:45] == images[2])
    mask[:, 40:45] = False
    assert numpy.all(pixel_data[:, 60:] == images[3])
    mask[:, 60:] = False
    assert numpy.all(pixel_data[mask] == 0)
