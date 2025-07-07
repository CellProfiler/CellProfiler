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

def test_02_ome_zarr_write_manual():
    """
    This tests the ability of the simplest pipeline to run in large image mode,
    setup a daskfiied image for modules to consume, such that we can manually
    write it out to a temporary file (.ome.zarr).

    The simple pipeline contains only the 4 standard modules.

    The input image is an .ome.tiff.
    The large image reader should set it up such that it is not
    read directly, but instead wrapped in a Dask array.

    The input image has 2 channels, and 2 pyramid levels (series).
    The z and t dimensions are always size 1.
    The low resolution level has height and width of 256x256, which
    is the same as the chunk size, and therefore each of the 2 channels is a single chunk.
    The high resolution level has height and width of 512x512, which is twice
    the chunk size in each dimension, and therefore each of the 2 channels has four chunks.

    In total there are 10 chunks.

    We run the pipeline to "daskify" the input image, and then manually (within the test itself)
    write it out as an .ome.zarr.

    This is important to test, because a less simple pipeline (one with more than just the 4 standard input modules)
    must also write out .ome.zarr files automatically. So we need a reference
    for what that should look like.

    If we can't set things up to manually write a correct .ome.zarr here, then we can expect
    less simple pipelines to also fail to do so.
    """
    # run the pipeline with just the 4 standard modules
    # this generates the planes, and sets other state
    # results of this are used to get the dask image and write it out to .ome.zarr
    pipeline, measurements = run_pipeline("tiled")
    image_plane_list: list[ImagePlane] = pipeline.image_plane_list

    # source filepath -> TiledImageWriter instance
    writers: dict[str, TiledImageWriter] = {}

    # for each image plane (single series, single channel)
    # write out that plane's tiles
    for i, plane in enumerate(image_plane_list):
        # should only trigger on the first plane
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

        # get the writer associated with the source image the plane comes from
        writer = writers[plane.path]

        # setup measurement state to be on current plane
        image_set_num = i+1
        measurements = pipeline.run(image_set_start=image_set_num, image_set_end=image_set_num)

        # the test pipeline should only have a single image name, 'DNA'
        img_name: list[str] = measurements.get_names()[0]
        img: Image = measurements.get_image(img_name)

        # directly write out the pixel data of the image plane to the .ome.zarr
        # the writer should handle the job of reading one chunk into memory,
        # then writing it out to the .ome.zarr, one at a time
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

    # get the source file path
    source_file_path = list(writers.keys())[0]

    # get the associated writer for the source file
    writer = writers[source_file_path]

    # get the path to the temporary file that was output
    tmp_file_file_path = writer.file_path

    # ensure the temp file is an .ome.zarr
    assert tmp_file_file_path.endswith('.ome.zarr')

    # setup a zarr reader for the temp file
    tmp_file_location = parse_url(tmp_file_file_path, mode="r")
    tmp_file_reader = Reader(tmp_file_location)

    # nodes generally may include images, labels, etc
    # we just have pixel data
    tmp_file_nodes = list(tmp_file_reader())
    assert len(tmp_file_nodes) >= 1

    # first node should be image pixel data
    tmp_file_img_node = tmp_file_nodes[0]
    tmp_file_dask_data = tmp_file_img_node.data

    # ensure there are 2 resolution levels
    assert len(tmp_file_dask_data) == 2

    # higher resolution
    assert tmp_file_dask_data[0].shape == (1, 2, 1, 512, 512)
    assert tmp_file_dask_data[0].dtype == np.float32
    assert tmp_file_dask_data[0].chunksize == (1, 1, 1, 256, 256)
    # lower resolution
    assert tmp_file_dask_data[1].shape == (1, 2, 1, 256, 256)
    assert tmp_file_dask_data[1].dtype == np.float32
    assert tmp_file_dask_data[1].chunksize == (1, 1, 1, 256, 256)

    # cleanup
    for w in writers.values():
        w.delete()

def test_03_ome_zarr_write_module():
    """
    This tests the ability of a simple pipeline to run in large image mode,
    and write out a temporary file (.ome.zarr) for subsequent usage.

    The simple pipeline contains the 4 standard modules, and GaussianFilter.
    The GaussianFilter module should run against the input Dask array,
    wrap it in a compute graph of performing the gaussian filtering,
    and write out the results, on disk, to a temporary .ome.zarr file.

    The input image is described above in test_02.

    The gaussian filter operation should run one tile at a time
    (i.e. only a single chunk of the input image is ever in memory).

    The temporary .ome.zarr file should contain the results of having
    performed gaussian filter on all chunks and pyramid levels.

    The output .ome.zarr should have 2 series (pyramid levels), 2 channels.
    The first level should be 256x256, the other should be 512x512.
    That should be identical in dimensionality and chunksize to the input image.
    """
    # run the pipeline with the 4 standard modules + GaussianFilter
    # GaussianFilter writes a temp file
    # results of this are what's under test
    ref_pipeline, ref_measurements = run_pipeline("tiled_gaussian")
