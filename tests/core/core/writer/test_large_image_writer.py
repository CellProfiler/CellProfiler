import os
import io
import tests.core
import cellprofiler_core.pipeline
import cellprofiler_core.modules.namesandtypes
import cellprofiler_core.utilities.pathname
import cellprofiler_core.utilities.measurement
import cellprofiler_core.measurement
import cellprofiler_core.workspace
import cellprofiler_core.object
import cellprofiler_core.writers.tiled_writer
import dask.array


def get_data_directory():
    folder = os.path.dirname(tests.core.__file__)
    return os.path.abspath(os.path.join(folder, "data/"))

def get_pipeline():
    pipeline_file = os.path.join(
        get_data_directory(), "tiled/tiled.cppipe"
    )
    with open(pipeline_file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))

    return pipeline

def make_ipd(url, metadata, series=0, index=0, channel=None):
    plane = ImagePlane(ImageFile(url), series=series, index=index, channel=channel)
    for key, value in metadata.items():
        plane.set_metadata(key, value)
    return plane

def run_pipeline():
    pipeline = get_pipeline()
    url = cellprofiler_core.utilities.pathname.pathname2url(os.path.join(get_data_directory(), "tiled/largeimg.ome.tiff"))
    pipeline.add_urls([url])
    for module in pipeline.modules():
        module.show_window = False
    measurements = pipeline.run(image_set_end=1)
    return pipeline, measurements

def test_01_load_pipeline():
    pipeline = get_pipeline()

    assert len(pipeline.modules()) == 4
    module = pipeline.modules()[2]
    assert isinstance(module, cellprofiler_core.modules.namesandtypes.NamesAndTypes)

    assert (
        module.assignment_method == cellprofiler_core.modules.namesandtypes.ASSIGN_ALL
    )
    assert (
        module.single_load_as_choice
            == cellprofiler_core.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE
    )
    assert module.single_image_provider.value == "DNA"

    assert module.assignments_count.value == 1
    assert (
        module.single_rescale_method.value
            == cellprofiler_core.modules.namesandtypes.INTENSITY_RESCALING_BY_DATATYPE
    )

    assert module.process_as_tiled.value # "Yes"
    assert not module.process_as_3d.value # "No"

def test_02_ome_tiff_read():
    data = dask.array.random.random((500, 1000), chunks=(500, 500))
    fd, filepath = cellprofiler_core.utilities.measurement.make_temporary_file(prefix="largeimg", suffix=".ome.zarr")
    os.close(fd)

    pipeline, _ = run_pipeline()

    for plane in pipeline.image_plane_list:
        file = cellprofiler_core.pipeline.ImageFile(filepath)
        writer = cellprofiler_core.writers.tiled_writer.TiledImageWriter(file)
        # provider -> get data
        writer.write_tiled()
