import io

import centrosome.cpmorphology as cpmorph
import centrosome.filter as cpfilter
import numpy as np
import pytest
import scipy.ndimage as scind

import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.modules.morph as morph
import cellprofiler.object as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import cellprofiler.workspace as cpw


def test_load_v2():
    with open("./tests/resources/modules/align/load_v2.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cpp.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cpp.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    ops = [
        morph.F_BRANCHPOINTS,
        morph.F_BRIDGE,
        morph.F_CLEAN,
        morph.F_CONVEX_HULL,
        morph.F_DIAG,
        morph.F_DISTANCE,
        morph.F_ENDPOINTS,
        morph.F_FILL,
        morph.F_HBREAK,
        morph.F_MAJORITY,
        morph.F_REMOVE,
        morph.F_SHRINK,
        morph.F_SKELPE,
        morph.F_SPUR,
        morph.F_THICKEN,
        morph.F_THIN,
        morph.F_VBREAK,
    ]
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, morph.Morph)
    assert module.image_name == "InputImage"
    assert module.output_image_name == "MorphImage"
    assert len(module.functions) == len(ops)


# https://github.com/CellProfiler/CellProfiler/issues/3349
def test_load_with_extracted_operations():
    with open("./tests/resources/modules/align/load_v2.pipeline", "r") as fd:
        data = fd.read()

    def callback(caller, event):
        assert not isinstance(event, cpp.LoadExceptionEvent)

    pipeline = cpp.Pipeline()
    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))

    module = pipeline.modules()[-1]
    with pytest.raises(cps.ValidationError):
        module.test_valid(pipeline)


def execute(image, function, mask=None, custom_repeats=None, scale=None, module=None):
    """Run the morph module on an input and return the resulting image"""
    INPUT_IMAGE_NAME = "input"
    OUTPUT_IMAGE_NAME = "output"
    if module is None:
        module = morph.Morph()
    module.functions[0].function.value = function
    module.image_name.value = INPUT_IMAGE_NAME
    module.output_image_name.value = OUTPUT_IMAGE_NAME
    if custom_repeats is None:
        module.functions[0].repeats_choice.value = morph.R_ONCE
    elif custom_repeats == -1:
        module.functions[0].repeats_choice.value = morph.R_FOREVER
    else:
        module.functions[0].repeats_choice.value = morph.R_CUSTOM
        module.functions[0].custom_repeats.value = custom_repeats
    if scale is not None:
        module.functions[0].scale.value = scale
    pipeline = cpp.Pipeline()
    object_set = cpo.ObjectSet()
    image_set_list = cpi.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    workspace = cpw.Workspace(
        pipeline, module, image_set, object_set, cpmeas.Measurements(), image_set_list
    )
    image_set.add(INPUT_IMAGE_NAME, cpi.Image(image, mask=mask))
    module.run(workspace)
    output = image_set.get_image(OUTPUT_IMAGE_NAME)
    return output.pixel_data


def binary_tteesstt(
    function_name, function, gray_out=False, scale=None, custom_repeats=None
):
    np.random.seed(list(map(ord, function_name)))
    input = np.random.uniform(size=(20, 20)) > 0.7
    output = execute(input, function_name, scale=scale, custom_repeats=custom_repeats)
    if scale is None:
        expected = function(input)
    else:
        footprint = cpmorph.strel_disk(float(scale) / 2.0)
        expected = function(input, footprint=footprint)
    if not gray_out:
        expected = expected > 0
        assert np.all(output == expected)
    else:
        assert np.all(np.abs(output - expected) < np.finfo(np.float32).eps)


def test_binary_branchpoints():
    binary_tteesstt("branchpoints", cpmorph.branchpoints)


def test_binary_bridge():
    binary_tteesstt("bridge", cpmorph.bridge)


def test_binary_clean():
    binary_tteesstt("clean", cpmorph.clean)


def test_binary_diag():
    binary_tteesstt("diag", cpmorph.diag)


def test_binary_endpoints():
    binary_tteesstt("endpoints", cpmorph.endpoints)


def test_binary_fill():
    binary_tteesstt("fill", cpmorph.fill)


def test_binary_hbreak():
    binary_tteesstt("hbreak", cpmorph.hbreak)


def test_binary_majority():
    binary_tteesstt("majority", cpmorph.majority)


def test_binary_remove():
    binary_tteesstt("remove", cpmorph.remove)


def test_binary_shrink():
    binary_tteesstt("shrink", lambda x: cpmorph.binary_shrink(x, 1))


def test_binary_spur():
    binary_tteesstt("spur", cpmorph.spur)


def test_binary_thicken():
    binary_tteesstt("thicken", cpmorph.thicken)


def test_binary_thin():
    binary_tteesstt("thin", cpmorph.thin)


def test_binary_vbreak():
    binary_tteesstt("vbreak", cpmorph.vbreak)


def test_binary_distance():
    def distance(x):
        y = scind.distance_transform_edt(x)
        if np.max(y) == 0:
            return y
        else:
            return y / np.max(y)

    binary_tteesstt("distance", distance, True)


def test_binary_convex_hull():
    #
    # Set the four points of a square to True
    #
    image = np.zeros((20, 15), bool)
    image[2, 3] = True
    image[17, 3] = True
    image[2, 12] = True
    image[17, 12] = True
    expected = np.zeros((20, 15), bool)
    expected[2:18, 3:13] = True
    result = execute(image, "convex hull")
    assert np.all(result == expected)


def test_binary_skelpe():
    def fn(x):
        d = scind.distance_transform_edt(x)
        pe = cpfilter.poisson_equation(x)
        return cpmorph.skeletonize(x, ordering=pe * d)

    binary_tteesstt("skelpe", fn)
