import io

import centrosome.cpmorphology
import centrosome.filter
import numpy
import pytest
import scipy.ndimage

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler.modules.morph
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.setting
import cellprofiler_core.workspace
import tests.frontend.modules


def test_load_v2():
    file = tests.frontend.modules.get_test_resources_directory("morph/v2.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    ops = [
        cellprofiler.modules.morph.F_BRANCHPOINTS,
        cellprofiler.modules.morph.F_BRIDGE,
        cellprofiler.modules.morph.F_CLEAN,
        cellprofiler.modules.morph.F_CONVEX_HULL,
        cellprofiler.modules.morph.F_DIAG,
        cellprofiler.modules.morph.F_DISTANCE,
        cellprofiler.modules.morph.F_ENDPOINTS,
        cellprofiler.modules.morph.F_FILL,
        cellprofiler.modules.morph.F_HBREAK,
        cellprofiler.modules.morph.F_MAJORITY,
        cellprofiler.modules.morph.F_REMOVE,
        cellprofiler.modules.morph.F_SHRINK,
        cellprofiler.modules.morph.F_SKELPE,
        cellprofiler.modules.morph.F_SPUR,
        cellprofiler.modules.morph.F_THICKEN,
        cellprofiler.modules.morph.F_THIN,
        cellprofiler.modules.morph.F_VBREAK,
    ]
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.morph.Morph)
    assert module.image_name == "InputImage"
    assert module.output_image_name == "MorphImage"
    assert len(module.functions) == len(ops)


# https://github.com/CellProfiler/CellProfiler/issues/3349
def test_load_with_extracted_operations():
    file = tests.frontend.modules.get_test_resources_directory("morph/extracted_operations.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))

    module = pipeline.modules()[-1]
    with pytest.raises(cellprofiler_core.setting.ValidationError):
        module.test_valid(pipeline)


def execute(image, function, mask=None, custom_repeats=None, scale=None, module=None):
    """Run the morph module on an input and return the resulting image"""
    INPUT_IMAGE_NAME = "input"
    OUTPUT_IMAGE_NAME = "output"
    if module is None:
        module = cellprofiler.modules.morph.Morph()
    module.functions[0].function.value = function
    module.image_name.value = INPUT_IMAGE_NAME
    module.output_image_name.value = OUTPUT_IMAGE_NAME
    if custom_repeats is None:
        module.functions[0].repeats_choice.value = cellprofiler.modules.morph.R_ONCE
    elif custom_repeats == -1:
        module.functions[0].repeats_choice.value = cellprofiler.modules.morph.R_FOREVER
    else:
        module.functions[0].repeats_choice.value = cellprofiler.modules.morph.R_CUSTOM
        module.functions[0].custom_repeats.value = custom_repeats
    if scale is not None:
        module.functions[0].scale.value = scale
    pipeline = cellprofiler_core.pipeline.Pipeline()
    object_set = cellprofiler_core.object.ObjectSet()
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    image_set.add(INPUT_IMAGE_NAME, cellprofiler_core.image.Image(image, mask=mask))
    module.run(workspace)
    output = image_set.get_image(OUTPUT_IMAGE_NAME)
    return output.pixel_data


def binary_tteesstt(
    function_name, function, gray_out=False, scale=None, custom_repeats=None
):
    numpy.random.seed(list(map(ord, function_name)))
    input = numpy.random.uniform(size=(20, 20)) > 0.7
    output = execute(input, function_name, scale=scale, custom_repeats=custom_repeats)
    if scale is None:
        expected = function(input)
    else:
        footprint = centrosome.cpmorphology.strel_disk(float(scale) / 2.0)
        expected = function(input, footprint=footprint)
    if not gray_out:
        expected = expected > 0
        assert numpy.all(output == expected)
    else:
        assert numpy.all(numpy.abs(output - expected) < numpy.finfo(numpy.float32).eps)


def test_binary_branchpoints():
    binary_tteesstt("branchpoints", centrosome.cpmorphology.branchpoints)


def test_binary_bridge():
    binary_tteesstt("bridge", centrosome.cpmorphology.bridge)


def test_binary_clean():
    binary_tteesstt("clean", centrosome.cpmorphology.clean)


def test_binary_diag():
    binary_tteesstt("diag", centrosome.cpmorphology.diag)


def test_binary_endpoints():
    binary_tteesstt("endpoints", centrosome.cpmorphology.endpoints)


def test_binary_fill():
    binary_tteesstt("fill", centrosome.cpmorphology.fill)


def test_binary_hbreak():
    binary_tteesstt("hbreak", centrosome.cpmorphology.hbreak)


def test_binary_majority():
    binary_tteesstt("majority", centrosome.cpmorphology.majority)


def test_binary_remove():
    binary_tteesstt("remove", centrosome.cpmorphology.remove)


def test_binary_shrink():
    binary_tteesstt("shrink", lambda x: centrosome.cpmorphology.binary_shrink(x, 1))


def test_binary_spur():
    binary_tteesstt("spur", centrosome.cpmorphology.spur)


def test_binary_thicken():
    binary_tteesstt("thicken", centrosome.cpmorphology.thicken)


def test_binary_thin():
    binary_tteesstt("thin", centrosome.cpmorphology.thin)


def test_binary_vbreak():
    binary_tteesstt("vbreak", centrosome.cpmorphology.vbreak)


def test_binary_distance():
    def distance(x):
        y = scipy.ndimage.distance_transform_edt(x)
        if numpy.max(y) == 0:
            return y
        else:
            return y / numpy.max(y)

    binary_tteesstt("distance", distance, True)


def test_binary_convex_hull():
    #
    # Set the four points of a square to True
    #
    image = numpy.zeros((20, 15), bool)
    image[2, 3] = True
    image[17, 3] = True
    image[2, 12] = True
    image[17, 12] = True
    expected = numpy.zeros((20, 15), bool)
    expected[2:18, 3:13] = True
    result = execute(image, "convex hull")
    assert numpy.all(result == expected)


def test_binary_skelpe():
    def fn(x):
        d = scipy.ndimage.distance_transform_edt(x)
        pe = centrosome.filter.poisson_equation(x)
        return centrosome.cpmorphology.skeletonize(x, ordering=pe * d)

    binary_tteesstt("skelpe", fn)
