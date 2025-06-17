import os
from pathlib import Path
import io
import tempfile
import tests.core
from cellprofiler_core.utilities.pathname import pathname2url
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.pipeline.event import LoadException
from cellprofiler_core.pipeline import ImageFile, ImagePlane
from cellprofiler_core.preferences import get_temporary_directory
from cellprofiler_core.image import Image
from cellprofiler_core.writers.tiled_writer import TiledImageWriter
from cellprofiler_core.modules.namesandtypes import NamesAndTypes, LOAD_AS_GRAYSCALE_IMAGE, INTENSITY_RESCALING_BY_DATATYPE, ASSIGN_ALL
from cellprofiler_core.constants.image import MD_SIZE_S, MD_SIZE_C, MD_SIZE_Z, MD_SIZE_T, MD_SIZE_Y, MD_SIZE_X
import dask.array
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import numpy as np


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

def run_pipeline(name):
    pipeline = get_pipeline(name)
    url = pathname2url(os.path.join(get_data_directory(), "tiled/largeimg.ome.tiff"))
    pipeline.add_urls([url])
    for module in pipeline.modules():
        module.show_window = False
    measurements = pipeline.run()
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
    #data = dask.array.random.random((500, 1000), chunks=(500, 500))
    ...

def test_03_ome_zarr_write():
    # run the pipeline with the 4 standard modules + GaussianFilter
    # GaussianFilter writes a temp file
    # results of this are what's under test
    run_pipeline("tiled_gaussian")
    # run the pipeline with just the 4 standard modules
    # this generates the planes, and sets other state
    # results of this are used to setup expectations
    pipeline, measurements = run_pipeline("tiled")
    image_plane_list: list[ImagePlane] = pipeline.image_plane_list

    # source filepath -> TiledImageWriter instance
    writers: dict[str, TiledImageWriter] = {}

    for i, plane in enumerate(image_plane_list):
        # one writer for all planes from same source file
        if plane.path not in writers:
            shapes = list(zip(
                plane.file.metadata[MD_SIZE_T],
                plane.file.metadata[MD_SIZE_C],
                plane.file.metadata[MD_SIZE_Z],
                plane.file.metadata[MD_SIZE_Y],
                plane.file.metadata[MD_SIZE_X]
            ))
            writers[plane.path] = TiledImageWriter(None, img_shapes=shapes)
        writer = writers[plane.path]

        # setup measurement state to be on current plane
        image_set_num = i+1
        measurements = pipeline.run(image_set_start=image_set_num, image_set_end=image_set_num)

        img_name: list[str] = measurements.get_names()[0] # e.g. 'DNA'
        img: Image = measurements.get_image(img_name)

        writer.write_tiled(
            img.pixel_data,
            series=plane.series,
            c=plane.channel,
            z=plane.z,
            t=plane.t,
            xywh=None,
            channel_names=None)

    # only one source file was used
    assert len(writers) == 1
    source_file_path = list(writers.keys())[0]
    writer = writers[source_file_path]
    out_file_path = writer.file_path
    assert out_file_path.endswith('.ome.zarr')

    out_location = parse_url(out_file_path, mode="r")
    out_reader = Reader(out_location)
    # nodes may include images, labels, etc
    out_nodes = list(out_reader())
    assert len(out_nodes) >= 1

    # first node should be image pixel data
    out_img_node = out_nodes[0]
    out_dask_data = out_img_node.data

    # 2 resolution levels
    assert len(out_dask_data) == 2

    # higher resolution
    assert out_dask_data[0].shape == (1, 2, 1, 512, 512)
    assert out_dask_data[0].dtype == np.float32
    assert out_dask_data[0].chunksize == (1, 1, 1, 256, 256)
    # lower resolution
    assert out_dask_data[1].shape == (1, 2, 1, 256, 256)
    assert out_dask_data[1].dtype == np.float32
    assert out_dask_data[1].chunksize == (1, 1, 1, 256, 256)

    # cleanup
    for w in writers.values():
        w.delete()
