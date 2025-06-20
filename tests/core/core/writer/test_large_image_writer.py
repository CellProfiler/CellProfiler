import os
import io
import tests.core
from cellprofiler_core.utilities.measurement import make_temporary_file
from cellprofiler_core.utilities.pathname import pathname2url
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.pipeline.event import LoadException
from cellprofiler_core.pipeline import ImageFile, ImagePlane
from cellprofiler_core.image import Image
from cellprofiler_core.writers.tiled_writer import TiledImageWriter
from cellprofiler_core.modules.namesandtypes import NamesAndTypes, LOAD_AS_GRAYSCALE_IMAGE, INTENSITY_RESCALING_BY_DATATYPE, ASSIGN_ALL
import dask.array


def get_data_directory():
    folder = os.path.dirname(tests.core.__file__)
    return os.path.abspath(os.path.join(folder, "data/"))

def get_pipeline(name):
    pipeline_file = os.path.join(
        get_data_directory(), f"tiled/{name}.cppipe"
    )
    with open(pipeline_file, "r") as fd:
        data = fd.read()

    pipeline = Pipeline()

    def callback(caller, event):
        assert not isinstance(event, LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))

    return pipeline

def make_ipd(url, metadata, series=0, index=0, channel=None):
    plane = ImagePlane(ImageFile(url), series=series, index=index, channel=channel)
    for key, value in metadata.items():
        plane.set_metadata(key, value)
    return plane

def run_pipeline(name):
    pipeline = get_pipeline(name)
    url = pathname2url(os.path.join(get_data_directory(), "tiled/largeimg.ome.tiff"))
    pipeline.add_urls([url])
    for module in pipeline.modules():
        module.show_window = False
    measurements = pipeline.run(image_set_end=1)
    return pipeline, measurements

def test_01_load_pipeline():
    pipeline = get_pipeline("tiled")

    assert len(pipeline.modules()) == 4
    module = pipeline.modules()[2]
    assert isinstance(module, NamesAndTypes)

    assert (module.assignment_method == ASSIGN_ALL)
    assert (module.single_load_as_choice == LOAD_AS_GRAYSCALE_IMAGE)
    assert module.single_image_provider.value == "DNA"

    assert module.assignments_count.value == 1
    assert (module.single_rescale_method.value == INTENSITY_RESCALING_BY_DATATYPE)

    assert module.process_as_tiled.value # "Yes"
    assert not module.process_as_3d.value # "No"

def test_02_ome_tiff_read():
    data = dask.array.random.random((500, 1000), chunks=(500, 500))
    fd, filepath = make_temporary_file(prefix="largeimg", suffix=".ome.zarr")
    os.close(fd)

    pipeline, measurements = run_pipeline("tiled_gaussian")
    image_plane_list: list[ImagePlane] = pipeline.image_plane_list

    for plane in image_plane_list:
        file = ImageFile(filepath)
        img_name: list[str] = measurements.get_names()[0] # e.g. 'DNA'
        img: Image = measurements.get_image(img_name)
        writer = TiledImageWriter(file)
        # provider -> get data
        writer.write_tiled()
