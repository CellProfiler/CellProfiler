import numpy
import pytest

import cellprofiler_core.image
import cellprofiler_core.measurement


import cellprofiler.modules.correctilluminationapply
import cellprofiler_core.modules.injectimage
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace


def error_callback(calller, event):
    if isinstance(event, cellprofiler_core.pipeline.event.RunException):
        pytest.fail(event.error.message)


def test_divide():
    """Test correction by division"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
    illum = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
    expected = image / illum
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    input_module = cellprofiler_core.modules.injectimage.InjectImage(
        "InputImage", image
    )
    input_module.set_module_num(1)
    pipeline.add_module(input_module)
    illum_module = cellprofiler_core.modules.injectimage.InjectImage(
        "IllumImage", illum
    )
    illum_module.set_module_num(2)
    pipeline.add_module(illum_module)
    module = cellprofiler.modules.correctilluminationapply.CorrectIlluminationApply()
    module.set_module_num(3)
    pipeline.add_module(module)
    image = module.images[0]
    image.image_name.value = "InputImage"
    image.illum_correct_function_image_name.value = "IllumImage"
    image.corrected_image_name.value = "OutputImage"
    image.divide_or_subtract.value = (
        cellprofiler.modules.correctilluminationapply.DOS_DIVIDE
    )
    image.rescale_option = cellprofiler.modules.correctilluminationapply.RE_NONE
    image_set_list = cellprofiler_core.image.ImageSetList()
    measurements = cellprofiler_core.measurement.Measurements()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, None, None, None, measurements, image_set_list
    )
    pipeline.prepare_run(workspace)
    input_module.prepare_group(workspace, {}, [1])
    illum_module.prepare_group(workspace, {}, [1])
    module.prepare_group(workspace, {}, [1])
    module.truncate_high.value = False
    module.truncate_low.value = False
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, input_module, image_set, object_set, measurements, image_set_list
    )
    input_module.run(workspace)
    illum_module.run(workspace)
    module.run(workspace)
    output_image = workspace.image_set.get_image("OutputImage")
    assert numpy.all(output_image.pixel_data == expected)


def test_divide_with_clip():
    """Test correction by division"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
    illum = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
    expected = image / illum
    expected = numpy.where(expected < 0, 0, expected)
    expected = numpy.where(expected > 1, 1, expected)
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    input_module = cellprofiler_core.modules.injectimage.InjectImage(
        "InputImage", image
    )
    input_module.set_module_num(1)
    pipeline.add_module(input_module)
    illum_module = cellprofiler_core.modules.injectimage.InjectImage(
        "IllumImage", illum
    )
    illum_module.set_module_num(2)
    pipeline.add_module(illum_module)
    module = cellprofiler.modules.correctilluminationapply.CorrectIlluminationApply()
    module.set_module_num(3)
    pipeline.add_module(module)
    image = module.images[0]
    image.image_name.value = "InputImage"
    image.illum_correct_function_image_name.value = "IllumImage"
    image.corrected_image_name.value = "OutputImage"
    image.divide_or_subtract.value = (
        cellprofiler.modules.correctilluminationapply.DOS_DIVIDE
    )
    image.rescale_option = cellprofiler.modules.correctilluminationapply.RE_NONE
    image_set_list = cellprofiler_core.image.ImageSetList()
    measurements = cellprofiler_core.measurement.Measurements()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, None, None, None, measurements, image_set_list
    )
    pipeline.prepare_run(workspace)
    input_module.prepare_group(workspace, {}, [1])
    illum_module.prepare_group(workspace, {}, [1])
    module.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, input_module, image_set, object_set, measurements, image_set_list
    )
    input_module.run(workspace)
    illum_module.run(workspace)
    module.run(workspace)
    output_image = workspace.image_set.get_image("OutputImage")
    assert numpy.all(output_image.pixel_data == expected)

def test_subtract():
    """Test correction by subtraction"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
    illum = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
    expected = image - illum
    expected[expected < 0] = 0
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    input_module = cellprofiler_core.modules.injectimage.InjectImage(
        "InputImage", image
    )
    input_module.set_module_num(1)
    pipeline.add_module(input_module)
    illum_module = cellprofiler_core.modules.injectimage.InjectImage(
        "IllumImage", illum
    )
    illum_module.set_module_num(2)
    pipeline.add_module(illum_module)
    module = cellprofiler.modules.correctilluminationapply.CorrectIlluminationApply()
    module.set_module_num(3)
    pipeline.add_module(module)
    image = module.images[0]
    image.image_name.value = "InputImage"
    image.illum_correct_function_image_name.value = "IllumImage"
    image.corrected_image_name.value = "OutputImage"
    image.divide_or_subtract.value = (
        cellprofiler.modules.correctilluminationapply.DOS_SUBTRACT
    )
    image.rescale_option = cellprofiler.modules.correctilluminationapply.RE_NONE
    measurements = cellprofiler_core.measurement.Measurements()
    image_set_list = cellprofiler_core.image.ImageSetList()
    measurements = cellprofiler_core.measurement.Measurements()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, None, None, None, measurements, image_set_list
    )
    pipeline.prepare_run(workspace)
    input_module.prepare_group(workspace, {}, [1])
    illum_module.prepare_group(workspace, {}, [1])
    module.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, input_module, image_set, object_set, measurements, image_set_list
    )
    input_module.run(workspace)
    illum_module.run(workspace)
    module.run(workspace)
    output_image = workspace.image_set.get_image("OutputImage")
    assert numpy.all(output_image.pixel_data == expected)


def test_color_by_bw():
    """Correct a color image with a black & white illumination fn"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(10, 10, 3)).astype(numpy.float32)
    illum = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
    expected = image - illum[:, :, numpy.newaxis]
    expected[expected < 0] = 0
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    input_module = cellprofiler_core.modules.injectimage.InjectImage(
        "InputImage", image
    )
    input_module.set_module_num(1)
    pipeline.add_module(input_module)
    illum_module = cellprofiler_core.modules.injectimage.InjectImage(
        "IllumImage", illum
    )
    illum_module.set_module_num(2)
    pipeline.add_module(illum_module)
    module = cellprofiler.modules.correctilluminationapply.CorrectIlluminationApply()
    module.set_module_num(3)
    pipeline.add_module(module)
    image = module.images[0]
    image.image_name.value = "InputImage"
    image.illum_correct_function_image_name.value = "IllumImage"
    image.corrected_image_name.value = "OutputImage"
    image.divide_or_subtract.value = (
        cellprofiler.modules.correctilluminationapply.DOS_SUBTRACT
    )
    image.rescale_option = cellprofiler.modules.correctilluminationapply.RE_NONE
    measurements = cellprofiler_core.measurement.Measurements()
    image_set_list = cellprofiler_core.image.ImageSetList()
    measurements = cellprofiler_core.measurement.Measurements()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, None, None, None, measurements, image_set_list
    )
    pipeline.prepare_run(workspace)
    input_module.prepare_group(workspace, {}, [1])
    illum_module.prepare_group(workspace, {}, [1])
    module.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, input_module, image_set, object_set, measurements, image_set_list
    )
    input_module.run(workspace)
    illum_module.run(workspace)
    module.run(workspace)
    output_image = workspace.image_set.get_image("OutputImage")
    assert numpy.all(output_image.pixel_data == expected)


def test_color_by_color():
    """Correct a color image with a black & white illumination fn"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(10, 10, 3)).astype(numpy.float32)
    illum = numpy.random.uniform(size=(10, 10, 3)).astype(numpy.float32)
    expected = image - illum
    expected[expected < 0] = 0
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    input_module = cellprofiler_core.modules.injectimage.InjectImage(
        "InputImage", image
    )
    input_module.set_module_num(1)
    pipeline.add_module(input_module)
    illum_module = cellprofiler_core.modules.injectimage.InjectImage(
        "IllumImage", illum
    )
    illum_module.set_module_num(2)
    pipeline.add_module(illum_module)
    module = cellprofiler.modules.correctilluminationapply.CorrectIlluminationApply()
    module.set_module_num(3)
    pipeline.add_module(module)
    image = module.images[0]
    image.image_name.value = "InputImage"
    image.illum_correct_function_image_name.value = "IllumImage"
    image.corrected_image_name.value = "OutputImage"
    image.divide_or_subtract.value = (
        cellprofiler.modules.correctilluminationapply.DOS_SUBTRACT
    )
    image.rescale_option = cellprofiler.modules.correctilluminationapply.RE_NONE
    measurements = cellprofiler_core.measurement.Measurements()
    image_set_list = cellprofiler_core.image.ImageSetList()
    measurements = cellprofiler_core.measurement.Measurements()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, None, None, None, measurements, image_set_list
    )
    pipeline.prepare_run(workspace)
    input_module.prepare_group(workspace, {}, [1])
    illum_module.prepare_group(workspace, {}, [1])
    module.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, input_module, image_set, object_set, measurements, image_set_list
    )
    input_module.run(workspace)
    illum_module.run(workspace)
    module.run(workspace)
    output_image = workspace.image_set.get_image("OutputImage")
    assert numpy.all(output_image.pixel_data == expected)
