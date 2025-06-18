import os
import io
import tests.core
from cellprofiler_core.utilities.pathname import pathname2url
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.pipeline.event import LoadException
from cellprofiler_core.pipeline import ImageFile, ImagePlane
from cellprofiler_core.modules.namesandtypes import NamesAndTypes, LOAD_AS_GRAYSCALE_IMAGE, INTENSITY_RESCALING_BY_DATATYPE, ASSIGN_ALL


def get_data_directory():
    folder = os.path.dirname(tests.core.__file__)
    return os.path.abspath(os.path.join(folder, "data/"))

def get_pipeline():
    pipeline_file = os.path.join(
        get_data_directory(), "tiled/tiled.cppipe"
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

def test_01_load_pipeline():
    pipeline = get_pipeline()

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
    pipeline = get_pipeline()
    url = pathname2url(os.path.join(get_data_directory(), "tiled/largeimg.ome.tiff"))
    pipeline.add_urls([url])
    for module in pipeline.modules():
        module.show_window = False
    m = pipeline.run(image_set_end=1)
    assert len(pipeline.image_plane_list) == 4

    sc = [(s,c) for s in range(2) for c in range(2)]
    for i, p in enumerate(pipeline.image_plane_list):
        assert p.reader_name == "TiledImage"
        assert p.series == sc[i][0]
        assert p.channel == sc[i][1]
        assert p.color_format == "monochrome"

