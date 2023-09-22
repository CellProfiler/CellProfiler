import numpy
import pytest
from centrosome.bg_compensate import MODE_AUTO, MODE_BRIGHT, MODE_DARK, MODE_GRAY
from six.moves import StringIO

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler.modules.correctilluminationcalculate
import cellprofiler_core.modules.injectimage
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.setting
import cellprofiler_core.workspace

import tests.frontend.modules

INPUT_IMAGE_NAME = "MyImage"
OUTPUT_IMAGE_NAME = "MyResult"
AVERAGE_IMAGE_NAME = "Ave"
DILATED_IMAGE_NAME = "Dilate"


def error_callback(calller, event):
    if isinstance(event, cellprofiler_core.pipeline.event.RunException):
        pytest.fail(event.error.message)


def make_workspaces(images_and_masks):
    """Make a workspace for each image set provided

    images_and_masks - a collection of two-tuples: image+mask

    returns a list of workspaces + the module
    """
    image_set_list = cellprofiler_core.image.ImageSetList()
    workspaces = []
    module = (
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
    )
    module.set_module_num(1)
    module.image_name.value = INPUT_IMAGE_NAME
    module.illumination_image_name.value = OUTPUT_IMAGE_NAME
    module.average_image_name.value = AVERAGE_IMAGE_NAME
    module.dilated_image_name.value = DILATED_IMAGE_NAME
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    measurements = cellprofiler_core.measurement.Measurements()

    for i, (image, mask) in enumerate(images_and_masks):
        image_set = image_set_list.get_image_set(i)
        if mask is None:
            image = cellprofiler_core.image.Image(image)
        else:
            image = cellprofiler_core.image.Image(image, mask)
        image_set.add(INPUT_IMAGE_NAME, image)
        workspace = cellprofiler_core.workspace.Workspace(
            pipeline,
            module,
            image_set,
            cellprofiler_core.object.ObjectSet(),
            measurements,
            image_set_list,
        )
        workspaces.append(workspace)
    return workspaces, module


def test_zeros():
    """Test all combinations of options with an image of all zeros"""
    for image in (numpy.zeros((10, 10)), numpy.zeros((10, 10, 3))):
        pipeline = cellprofiler_core.pipeline.Pipeline()
        pipeline.add_listener(error_callback)
        inj_module = cellprofiler_core.modules.injectimage.InjectImage("MyImage", image)
        inj_module.set_module_num(1)
        pipeline.add_module(inj_module)
        module = (
            cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
        )
        module.set_module_num(2)
        pipeline.add_module(module)
        module.image_name.value = "MyImage"
        module.illumination_image_name.value = "OutputImage"
        module.save_average_image.value = True
        module.save_dilated_image.value = True

        for ea in (
            cellprofiler.modules.correctilluminationcalculate.EA_EACH,
            cellprofiler.modules.correctilluminationcalculate.EA_ALL_ACROSS,
            cellprofiler.modules.correctilluminationcalculate.EA_ALL_FIRST,
        ):
            module.each_or_all.value = ea
            for intensity_choice in (
                cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND,
                cellprofiler.modules.correctilluminationcalculate.IC_REGULAR,
            ):
                module.intensity_choice.value = intensity_choice
                for dilate_objects in (True, False):
                    module.dilate_objects.value = dilate_objects
                    for rescale_option in (
                        "Yes",
                        "No",
                        cellprofiler.modules.correctilluminationcalculate.RE_MEDIAN,
                    ):
                        module.rescale_option.value = rescale_option
                        for smoothing_method in (
                            cellprofiler.modules.correctilluminationcalculate.SM_NONE,
                            cellprofiler.modules.correctilluminationcalculate.SM_FIT_POLYNOMIAL,
                            cellprofiler.modules.correctilluminationcalculate.SM_GAUSSIAN_FILTER,
                            cellprofiler.modules.correctilluminationcalculate.SM_MEDIAN_FILTER,
                            cellprofiler.modules.correctilluminationcalculate.SM_TO_AVERAGE,
                            cellprofiler.modules.correctilluminationcalculate.SM_SPLINES,
                            cellprofiler.modules.correctilluminationcalculate.SM_CONVEX_HULL,
                        ):
                            module.smoothing_method.value = smoothing_method
                            for ow in (
                                cellprofiler.modules.correctilluminationcalculate.FI_AUTOMATIC,
                                cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY,
                                cellprofiler.modules.correctilluminationcalculate.FI_OBJECT_SIZE,
                            ):
                                module.automatic_object_width.value = ow
                                measurements = (
                                    cellprofiler_core.measurement.Measurements()
                                )
                                image_set_list = cellprofiler_core.image.ImageSetList()
                                workspace = cellprofiler_core.workspace.Workspace(
                                    pipeline,
                                    None,
                                    None,
                                    None,
                                    measurements,
                                    image_set_list,
                                )
                                pipeline.prepare_run(workspace)
                                inj_module.prepare_group(workspace, {}, [1])
                                module.prepare_group(workspace, {}, [1])
                                image_set = image_set_list.get_image_set(0)
                                object_set = cellprofiler_core.object.ObjectSet()
                                workspace = cellprofiler_core.workspace.Workspace(
                                    pipeline,
                                    inj_module,
                                    image_set,
                                    object_set,
                                    measurements,
                                    image_set_list,
                                )
                                inj_module.run(workspace)
                                module.run(workspace)
                                image = image_set.get_image("OutputImage")
                                assert image is not None
                                assert numpy.all(image.pixel_data == 0), (
                                    """Failure case:
                            intensity_choice = %(intensity_choice)s
                            dilate_objects = %(dilate_objects)s
                            rescale_option = %(rescale_option)s
                            smoothing_method = %(smoothing_method)s
                            automatic_object_width = %(ow)s"""
                                    % locals()
                                )


def test_ones_image():
    """The illumination correction of an image of all ones should be uniform

    """
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    for image in (numpy.ones((10, 10)), numpy.ones((10, 10, 3))):
        inj_module = cellprofiler_core.modules.injectimage.InjectImage("MyImage", image)
        inj_module.set_module_num(1)
        pipeline.add_module(inj_module)
        module = (
            cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
        )
        module.set_module_num(2)
        pipeline.add_module(module)
        module.image_name.value = "MyImage"
        module.illumination_image_name.value = "OutputImage"
        module.rescale_option.value = "Yes"

        for ea in (
            cellprofiler.modules.correctilluminationcalculate.EA_EACH,
            cellprofiler.modules.correctilluminationcalculate.EA_ALL_ACROSS,
            cellprofiler.modules.correctilluminationcalculate.EA_ALL_FIRST,
        ):
            module.each_or_all.value = ea
            for intensity_choice in (
                cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND,
                cellprofiler.modules.correctilluminationcalculate.IC_REGULAR,
            ):
                module.intensity_choice.value = intensity_choice
                for dilate_objects in (True, False):
                    module.dilate_objects.value = dilate_objects
                    for smoothing_method in (
                        cellprofiler.modules.correctilluminationcalculate.SM_NONE,
                        cellprofiler.modules.correctilluminationcalculate.SM_FIT_POLYNOMIAL,
                        cellprofiler.modules.correctilluminationcalculate.SM_GAUSSIAN_FILTER,
                        cellprofiler.modules.correctilluminationcalculate.SM_MEDIAN_FILTER,
                        cellprofiler.modules.correctilluminationcalculate.SM_TO_AVERAGE,
                        cellprofiler.modules.correctilluminationcalculate.SM_SPLINES,
                        cellprofiler.modules.correctilluminationcalculate.SM_CONVEX_HULL,
                    ):
                        module.smoothing_method.value = smoothing_method
                        for ow in (
                            cellprofiler.modules.correctilluminationcalculate.FI_AUTOMATIC,
                            cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY,
                            cellprofiler.modules.correctilluminationcalculate.FI_OBJECT_SIZE,
                        ):
                            module.automatic_object_width.value = ow
                            measurements = cellprofiler_core.measurement.Measurements()
                            image_set_list = cellprofiler_core.image.ImageSetList()
                            workspace = cellprofiler_core.workspace.Workspace(
                                pipeline, None, None, None, measurements, image_set_list
                            )
                            pipeline.prepare_run(workspace)
                            inj_module.prepare_group(workspace, {}, [1])
                            module.prepare_group(workspace, {}, [1])
                            image_set = image_set_list.get_image_set(0)
                            object_set = cellprofiler_core.object.ObjectSet()
                            workspace = cellprofiler_core.workspace.Workspace(
                                pipeline,
                                inj_module,
                                image_set,
                                object_set,
                                measurements,
                                image_set_list,
                            )
                            inj_module.run(workspace)
                            module.run(workspace)
                            image = image_set.get_image("OutputImage")
                            assert image is not None
                            assert numpy.all(numpy.std(image.pixel_data) < 0.00001), (
                                """Failure case:
                        each_or_all            = %(ea)s
                        intensity_choice       = %(intensity_choice)s
                        dilate_objects         = %(dilate_objects)s
                        smoothing_method       = %(smoothing_method)s
                        automatic_object_width = %(ow)s"""
                                % locals()
                            )


def test_masked_image():
    """A masked image should be insensitive to points outside the mask"""
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    numpy.random.seed(12)
    for image in (
        numpy.random.uniform(size=(10, 10)),
        numpy.random.uniform(size=(10, 10, 3)),
    ):
        mask = numpy.zeros((10, 10), bool)
        mask[2:7, 3:8] = True
        image[mask] = 1
        inj_module = cellprofiler_core.modules.injectimage.InjectImage(
            "MyImage", image, mask
        )
        inj_module.set_module_num(1)
        pipeline.add_module(inj_module)
        module = (
            cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
        )
        module.set_module_num(2)
        pipeline.add_module(module)
        module.image_name.value = "MyImage"
        module.illumination_image_name.value = "OutputImage"
        module.rescale_option.value = "Yes"
        module.dilate_objects.value = False

        for ea in (
            cellprofiler.modules.correctilluminationcalculate.EA_EACH,
            cellprofiler.modules.correctilluminationcalculate.EA_ALL_ACROSS,
            cellprofiler.modules.correctilluminationcalculate.EA_ALL_FIRST,
        ):
            module.each_or_all.value = ea
            for intensity_choice in (
                cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND,
                cellprofiler.modules.correctilluminationcalculate.IC_REGULAR,
            ):
                module.intensity_choice.value = intensity_choice
                for smoothing_method in (
                    cellprofiler.modules.correctilluminationcalculate.SM_NONE,
                    cellprofiler.modules.correctilluminationcalculate.SM_FIT_POLYNOMIAL,
                    cellprofiler.modules.correctilluminationcalculate.SM_GAUSSIAN_FILTER,
                    cellprofiler.modules.correctilluminationcalculate.SM_MEDIAN_FILTER,
                    cellprofiler.modules.correctilluminationcalculate.SM_TO_AVERAGE,
                    cellprofiler.modules.correctilluminationcalculate.SM_CONVEX_HULL,
                ):
                    module.smoothing_method.value = smoothing_method
                    for ow in (
                        cellprofiler.modules.correctilluminationcalculate.FI_AUTOMATIC,
                        cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY,
                        cellprofiler.modules.correctilluminationcalculate.FI_OBJECT_SIZE,
                    ):
                        module.automatic_object_width.value = ow
                        measurements = cellprofiler_core.measurement.Measurements()
                        image_set_list = cellprofiler_core.image.ImageSetList()
                        workspace = cellprofiler_core.workspace.Workspace(
                            pipeline, None, None, None, measurements, image_set_list
                        )
                        pipeline.prepare_run(workspace)
                        inj_module.prepare_group(workspace, {}, [1])
                        module.prepare_group(workspace, {}, [1])
                        image_set = image_set_list.get_image_set(0)
                        object_set = cellprofiler_core.object.ObjectSet()
                        workspace = cellprofiler_core.workspace.Workspace(
                            pipeline,
                            inj_module,
                            image_set,
                            object_set,
                            measurements,
                            image_set_list,
                        )
                        inj_module.run(workspace)
                        module.run(workspace)
                        image = image_set.get_image("OutputImage")
                        assert image is not None
                        assert numpy.all(abs(image.pixel_data[mask] - 1 < 0.00001)), (
                            """Failure case:
                        each_or_all            = %(ea)s
                        intensity_choice       = %(intensity_choice)s
                        smoothing_method       = %(smoothing_method)s
                        automatic_object_width = %(ow)s"""
                            % locals()
                        )


def test_filtered():
    """Regression test of issue #310

    post_group should add the composite image to the image set
    if CorrectIllumination_Calculate didn't run because the image
    set was filtered.
    """
    r = numpy.random.RandomState()
    r.seed(13)
    i0 = r.uniform(size=(11, 13))
    i1 = r.uniform(size=(11, 13))
    i2 = r.uniform(size=(11, 13))
    workspaces, module = make_workspaces(((i0, None), (i1, None), (i2, None)))
    module.each_or_all.value = (
        cellprofiler.modules.correctilluminationcalculate.EA_ALL_ACROSS
    )
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_TO_AVERAGE
    )
    module.save_average_image.value = True
    module.save_dilated_image.value = True

    module.prepare_group(workspaces[0], None, [1, 2, 3])
    assert isinstance(
        module,
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate,
    )
    for workspace in workspaces[:-1]:
        assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
        module.run(workspace)
    image_set = workspaces[-1].image_set
    assert OUTPUT_IMAGE_NAME not in image_set.names
    assert DILATED_IMAGE_NAME not in image_set.names
    assert AVERAGE_IMAGE_NAME not in image_set.names
    module.post_group(workspaces[-1], None)
    assert OUTPUT_IMAGE_NAME in image_set.names
    assert DILATED_IMAGE_NAME in image_set.names
    assert AVERAGE_IMAGE_NAME in image_set.names


def test_not_filtered():
    """Regression test of issue #310, negative case

    post_group should not add the composite image to the image set
    if CorrectIllumination_Calculate did run.
    """
    r = numpy.random.RandomState()
    r.seed(13)
    i0 = r.uniform(size=(11, 13))
    i1 = r.uniform(size=(11, 13))
    i2 = r.uniform(size=(11, 13))
    workspaces, module = make_workspaces(((i0, None), (i1, None), (i2, None)))
    module.each_or_all.value = (
        cellprofiler.modules.correctilluminationcalculate.EA_ALL_ACROSS
    )
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_TO_AVERAGE
    )
    module.save_average_image.value = True
    module.save_dilated_image.value = True

    module.prepare_group(workspaces[0], None, [1, 2, 3])
    assert isinstance(
        module,
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate,
    )
    for workspace in workspaces:
        assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
        module.run(workspace)
    image_set = workspaces[-1].image_set
    assert OUTPUT_IMAGE_NAME in image_set.names
    assert DILATED_IMAGE_NAME in image_set.names
    assert AVERAGE_IMAGE_NAME in image_set.names
    module.post_group(workspaces[-1], None)
    #
    # Make sure it appears only once
    #
    for image_name in (OUTPUT_IMAGE_NAME, DILATED_IMAGE_NAME, AVERAGE_IMAGE_NAME):
        assert len([x for x in image_set.names if x == image_name]) == 1


def test_Background():
    """Test an image with four distinct backgrounds"""

    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    image = numpy.ones((40, 40))
    image[10, 10] = 0.25
    image[10, 30] = 0.5
    image[30, 10] = 0.75
    image[30, 30] = 0.9
    inj_module = cellprofiler_core.modules.injectimage.InjectImage("MyImage", image)
    inj_module.set_module_num(1)
    pipeline.add_module(inj_module)
    module = (
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
    )
    module.set_module_num(2)
    pipeline.add_module(module)
    module.image_name.value = "MyImage"
    module.illumination_image_name.value = "OutputImage"
    module.intensity_choice.value = (
        cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
    )
    module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
    module.block_size.value = 20
    module.rescale_option.value = "No"
    module.dilate_objects.value = False
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_NONE
    )
    measurements = cellprofiler_core.measurement.Measurements()
    image_set_list = cellprofiler_core.image.ImageSetList()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, None, None, None, measurements, image_set_list
    )
    pipeline.prepare_run(workspace)
    inj_module.prepare_group(workspace, {}, [1])
    module.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, inj_module, image_set, object_set, measurements, image_set_list
    )
    inj_module.run(workspace)
    module.run(workspace)
    image = image_set.get_image("OutputImage")
    assert numpy.all(image.pixel_data[:20, :20] == 0.25)
    assert numpy.all(image.pixel_data[:20, 20:] == 0.5)
    assert numpy.all(image.pixel_data[20:, :20] == 0.75)
    assert numpy.all(image.pixel_data[20:, 20:] == 0.9)


def test_no_smoothing():
    """Make sure that no smoothing takes place if smoothing is turned off"""
    input_image = numpy.random.uniform(size=(10, 10))
    image_name = "InputImage"
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    inj_module = cellprofiler_core.modules.injectimage.InjectImage(
        image_name, input_image
    )
    inj_module.set_module_num(1)
    pipeline.add_module(inj_module)
    module = (
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
    )
    module.set_module_num(2)
    pipeline.add_module(module)
    module.image_name.value = image_name
    module.illumination_image_name.value = "OutputImage"
    module.intensity_choice.value = (
        cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
    )
    module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_NONE
    )
    module.rescale_option.value = "No"
    module.dilate_objects.value = False
    measurements = cellprofiler_core.measurement.Measurements()
    image_set_list = cellprofiler_core.image.ImageSetList()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, None, None, None, measurements, image_set_list
    )
    pipeline.prepare_run(workspace)
    inj_module.prepare_group(workspace, {}, [1])
    module.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, inj_module, image_set, object_set, measurements, image_set_list
    )
    inj_module.run(workspace)
    module.run(workspace)
    image = image_set.get_image("OutputImage")
    assert numpy.all(numpy.abs(image.pixel_data - input_image) < 0.001), (
        "Failed to fit polynomial to %s" % image_name
    )


def test_FitPolynomial():
    """Test fitting a polynomial to different gradients"""

    y, x = (numpy.mgrid[0:20, 0:20]).astype(float) / 20.0
    image_x = x
    image_y = y
    image_x2 = x ** 2
    image_y2 = y ** 2
    image_xy = x * y
    for input_image, image_name in (
        (image_x, "XImage"),
        (image_y, "YImage"),
        (image_x2, "X2Image"),
        (image_y2, "Y2Image"),
        (image_xy, "XYImage"),
    ):
        pipeline = cellprofiler_core.pipeline.Pipeline()
        pipeline.add_listener(error_callback)
        inj_module = cellprofiler_core.modules.injectimage.InjectImage(
            image_name, input_image
        )
        inj_module.set_module_num(1)
        pipeline.add_module(inj_module)
        module = (
            cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
        )
        module.set_module_num(2)
        pipeline.add_module(module)
        module.image_name.value = image_name
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = (
            cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
        )
        module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
        module.smoothing_method.value = (
            cellprofiler.modules.correctilluminationcalculate.SM_FIT_POLYNOMIAL
        )
        module.rescale_option.value = "No"
        module.dilate_objects.value = False
        measurements = cellprofiler_core.measurement.Measurements()
        image_set_list = cellprofiler_core.image.ImageSetList()
        workspace = cellprofiler_core.workspace.Workspace(
            pipeline, None, None, None, measurements, image_set_list
        )
        pipeline.prepare_run(workspace)
        inj_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cellprofiler_core.object.ObjectSet()
        workspace = cellprofiler_core.workspace.Workspace(
            pipeline, inj_module, image_set, object_set, measurements, image_set_list
        )
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        assert numpy.all(numpy.abs(image.pixel_data - input_image) < 0.001), (
            "Failed to fit polynomial to %s" % image_name
        )


def test_gaussian_filter():
    """Test gaussian filtering a gaussian of a point"""
    input_image = numpy.zeros((101, 101))
    input_image[50, 50] = 1
    image_name = "InputImage"
    i, j = numpy.mgrid[-50:51, -50:51]
    expected_image = numpy.e ** (-(i ** 2 + j ** 2) / (2 * (10.0 / 2.35) ** 2))
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    inj_module = cellprofiler_core.modules.injectimage.InjectImage(
        image_name, input_image
    )
    inj_module.set_module_num(1)
    pipeline.add_module(inj_module)
    module = (
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
    )
    module.set_module_num(2)
    pipeline.add_module(module)
    module.image_name.value = image_name
    module.illumination_image_name.value = "OutputImage"
    module.intensity_choice.value = (
        cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
    )
    module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_GAUSSIAN_FILTER
    )
    module.automatic_object_width.value = (
        cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY
    )
    module.size_of_smoothing_filter.value = 10
    module.rescale_option.value = "No"
    module.dilate_objects.value = False
    measurements = cellprofiler_core.measurement.Measurements()
    image_set_list = cellprofiler_core.image.ImageSetList()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, None, None, None, measurements, image_set_list
    )
    pipeline.prepare_run(workspace)
    inj_module.prepare_group(workspace, {}, [1])
    module.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, inj_module, image_set, object_set, measurements, image_set_list
    )
    inj_module.run(workspace)
    module.run(workspace)
    image = image_set.get_image("OutputImage")
    ipd = image.pixel_data[40:61, 40:61]
    expected_image = expected_image[40:61, 40:61]
    assert numpy.all(
        numpy.abs(ipd / ipd.mean() - expected_image / expected_image.mean()) < 0.001
    )


def test_median_filter():
    """Test median filtering of a point"""
    input_image = numpy.zeros((101, 101))
    input_image[50, 50] = 1
    image_name = "InputImage"
    expected_image = numpy.zeros((101, 101))
    filter_distance = int(0.5 + 10 / 2.35)
    expected_image[
        -filter_distance : filter_distance + 1, -filter_distance : filter_distance + 1
    ] = 1
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    inj_module = cellprofiler_core.modules.injectimage.InjectImage(
        image_name, input_image
    )
    inj_module.set_module_num(1)
    pipeline.add_module(inj_module)
    module = (
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
    )
    module.set_module_num(2)
    pipeline.add_module(module)
    module.image_name.value = image_name
    module.illumination_image_name.value = "OutputImage"
    module.intensity_choice.value = (
        cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
    )
    module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_MEDIAN_FILTER
    )
    module.automatic_object_width.value = (
        cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY
    )
    module.size_of_smoothing_filter.value = 10
    module.rescale_option.value = "No"
    module.dilate_objects.value = False
    measurements = cellprofiler_core.measurement.Measurements()
    image_set_list = cellprofiler_core.image.ImageSetList()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, None, None, None, measurements, image_set_list
    )
    pipeline.prepare_run(workspace)
    inj_module.prepare_group(workspace, {}, [1])
    module.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, inj_module, image_set, object_set, measurements, image_set_list
    )
    inj_module.run(workspace)
    module.run(workspace)
    image = image_set.get_image("OutputImage")
    assert numpy.all(image.pixel_data == expected_image)


def test_smooth_to_average():
    """Test smoothing to an average value"""
    numpy.random.seed(0)
    input_image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
    image_name = "InputImage"
    expected_image = numpy.ones((10, 10)) * input_image.mean()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    inj_module = cellprofiler_core.modules.injectimage.InjectImage(
        image_name, input_image
    )
    inj_module.set_module_num(1)
    pipeline.add_module(inj_module)
    module = (
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
    )
    module.set_module_num(2)
    pipeline.add_module(module)
    module.image_name.value = image_name
    module.illumination_image_name.value = "OutputImage"
    module.intensity_choice.value = (
        cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
    )
    module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_TO_AVERAGE
    )
    module.automatic_object_width.value = (
        cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY
    )
    module.size_of_smoothing_filter.value = 10
    module.rescale_option.value = "No"
    module.dilate_objects.value = False
    measurements = cellprofiler_core.measurement.Measurements()
    image_set_list = cellprofiler_core.image.ImageSetList()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, None, None, None, measurements, image_set_list
    )
    pipeline.prepare_run(workspace)
    inj_module.prepare_group(workspace, {}, [1])
    module.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, inj_module, image_set, object_set, measurements, image_set_list
    )
    inj_module.run(workspace)
    module.run(workspace)
    image = image_set.get_image("OutputImage")
    numpy.testing.assert_almost_equal(image.pixel_data, expected_image)


def test_splines():
    for (
        automatic,
        bg_mode,
        spline_points,
        threshold,
        convergence,
        offset,
        hi,
        lo,
        succeed,
    ) in (
        (
            True,
            MODE_AUTO,
            5,
            2,
            0.001,
            0,
            True,
            False,
            True,
        ),
        (
            True,
            MODE_AUTO,
            5,
            2,
            0.001,
            0.7,
            False,
            True,
            True,
        ),
        (
            True,
            MODE_AUTO,
            5,
            2,
            0.001,
            0.5,
            True,
            True,
            True,
        ),
        (
            False,
            MODE_AUTO,
            5,
            2,
            0.001,
            0,
            True,
            False,
            True,
        ),
        (
            False,
            MODE_AUTO,
            5,
            2,
            0.001,
            0.7,
            False,
            True,
            True,
        ),
        (
            False,
            MODE_AUTO,
            5,
            2,
            0.001,
            0.5,
            True,
            True,
            True,
        ),
        (
            False,
            MODE_BRIGHT,
            5,
            2,
            0.001,
            0.7,
            False,
            True,
            True,
        ),
        (
            False,
            MODE_DARK,
            5,
            2,
            0.001,
            0,
            True,
            False,
            True,
        ),
        (
            False,
            MODE_GRAY,
            5,
            2,
            0.001,
            0.5,
            True,
            True,
            True,
        ),
        (
            False,
            MODE_AUTO,
            7,
            2,
            0.001,
            0,
            True,
            False,
            True,
        ),
        (
            False,
            MODE_AUTO,
            4,
            2,
            0.001,
            0,
            True,
            False,
            True,
        ),
        (
            False,
            MODE_DARK,
            5,
            2,
            0.001,
            0.7,
            False,
            True,
            False,
        ),
        (
            False,
            MODE_BRIGHT,
            5,
            2,
            0.001,
            0,
            True,
            False,
            False,
        ),
    ):

        #
        # Make an image with a random background
        #
        numpy.random.seed(35)
        image = numpy.random.uniform(size=(21, 31)) * 0.05 + offset
        if hi:
            #
            # Add some "foreground" pixels
            #
            fg = numpy.random.permutation(400)[:100]
            image[fg % image.shape[0], (fg / image.shape[0]).astype(int)] *= 10
        if lo:
            #
            # Add some "background" pixels
            #
            bg = numpy.random.permutation(400)[:100]
            image[bg % image.shape[0], (bg / image.shape[0]).astype(int)] -= offset

        #
        # Make a background function
        #
        ii, jj = numpy.mgrid[-10:11, -15:16]
        bg = ((ii.astype(float) / 10) ** 2) * ((jj.astype(float) / 15) ** 2)
        bg *= 0.2
        image += bg

        workspaces, module = make_workspaces(((image, None),))
        assert isinstance(
            module,
            cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate,
        )
        module.intensity_choice.value = (
            cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
        )
        module.each_or_all.value = (
            cellprofiler.modules.correctilluminationcalculate.EA_EACH
        )
        module.rescale_option.value = "No"
        module.smoothing_method.value = (
            cellprofiler.modules.correctilluminationcalculate.SM_SPLINES
        )
        module.automatic_splines.value = automatic
        module.spline_bg_mode.value = bg_mode
        module.spline_convergence.value = convergence
        module.spline_threshold.value = threshold
        module.spline_points.value = spline_points
        module.spline_rescale.value = 1
        module.prepare_group(workspaces[0], {}, [1])
        module.run(workspaces[0])
        img = workspaces[0].image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = img.pixel_data
        diff = pixel_data - numpy.min(pixel_data) - bg
        if succeed:
            assert numpy.all(diff < 0.05)
        else:
            assert not numpy.all(diff < 0.05)


def test_splines_scaled():
    #
    # Make an image with a random background
    #
    numpy.random.seed(36)
    image = numpy.random.uniform(size=(101, 131)) * 0.05
    #
    # Add some "foreground" pixels
    #
    fg = numpy.random.permutation(numpy.prod(image.shape))[:200]
    image[fg % image.shape[0], (fg / image.shape[0]).astype(int)] *= 15
    #
    # Make a background function
    #
    ii, jj = numpy.mgrid[-50:51, -65:66]
    bg = ((ii.astype(float) / 10) ** 2) * ((jj.astype(float) / 15) ** 2)
    bg *= 0.2
    image += bg

    workspaces, module = make_workspaces(((image, None),))
    assert isinstance(
        module,
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate,
    )
    module.intensity_choice.value = (
        cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
    )
    module.each_or_all.value = cellprofiler.modules.correctilluminationcalculate.EA_EACH
    module.rescale_option.value = "No"
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_SPLINES
    )
    module.automatic_splines.value = False
    module.spline_rescale.value = 2
    module.prepare_group(workspaces[0], {}, [1])
    module.run(workspaces[0])
    img = workspaces[0].image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = img.pixel_data
    diff = pixel_data - numpy.min(pixel_data) - bg
    numpy.all(diff < 0.05)


def test_splines_masked():
    #
    # Make an image with a random background
    #
    numpy.random.seed(37)
    image = numpy.random.uniform(size=(21, 31)) * 0.05
    #
    # Mask 1/2 of the pixels
    #
    mask = numpy.random.uniform(size=(21, 31)) < 0.5
    #
    # Make a background function
    #
    ii, jj = numpy.mgrid[-10:11, -15:16]
    bg = ((ii.astype(float) / 10) ** 2) * ((jj.astype(float) / 15) ** 2)
    bg *= 0.2
    image += bg
    #
    # Offset the background within the mask
    #
    image[~mask] += bg[~mask]

    workspaces, module = make_workspaces(((image, mask),))
    assert isinstance(
        module,
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate,
    )
    module.intensity_choice.value = (
        cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
    )
    module.each_or_all.value = cellprofiler.modules.correctilluminationcalculate.EA_EACH
    module.rescale_option.value = "No"
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_SPLINES
    )
    module.automatic_splines.value = True
    module.prepare_group(workspaces[0], {}, [1])
    module.run(workspaces[0])
    img = workspaces[0].image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = img.pixel_data
    diff = pixel_data - numpy.min(pixel_data) - bg
    assert numpy.all(diff < 0.05)
    #
    # Make sure test fails w/o mask
    #
    workspaces, module = make_workspaces(((image, None),))
    assert isinstance(
        module,
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate,
    )
    module.intensity_choice.value = (
        cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
    )
    module.each_or_all.value = cellprofiler.modules.correctilluminationcalculate.EA_EACH
    module.rescale_option.value = "No"
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_SPLINES
    )
    module.automatic_splines.value = True
    module.prepare_group(workspaces[0], {}, [1])
    module.run(workspaces[0])
    img = workspaces[0].image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = img.pixel_data
    diff = pixel_data - numpy.min(pixel_data) - bg
    assert not numpy.all(diff < 0.05)


def test_splines_cropped():
    #
    # Make an image with a random background
    #
    numpy.random.seed(37)
    image = numpy.random.uniform(size=(21, 31)) * 0.05
    #
    # Mask 1/2 of the pixels
    #
    mask = numpy.zeros(image.shape, bool)
    mask[4:-4, 6:-6] = True
    #
    # Make a background function
    #
    ii, jj = numpy.mgrid[-10:11, -15:16]
    bg = ((ii.astype(float) / 10) ** 2) * ((jj.astype(float) / 15) ** 2)
    bg *= 0.2
    image += bg
    #
    # Offset the background within the mask
    #
    image[~mask] += bg[~mask]

    workspaces, module = make_workspaces(((image, mask),))
    assert isinstance(
        module,
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate,
    )
    module.intensity_choice.value = (
        cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
    )
    module.each_or_all.value = cellprofiler.modules.correctilluminationcalculate.EA_EACH
    module.rescale_option.value = "No"
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_SPLINES
    )
    module.automatic_splines.value = True
    module.prepare_group(workspaces[0], {}, [1])
    module.run(workspaces[0])
    img = workspaces[0].image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = img.pixel_data
    diff = pixel_data - numpy.min(pixel_data) - bg
    assert numpy.all(diff < 0.05)
    #
    # Make sure test fails w/o mask
    #
    workspaces, module = make_workspaces(((image, None),))
    assert isinstance(
        module,
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate,
    )
    module.intensity_choice.value = (
        cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
    )
    module.each_or_all.value = cellprofiler.modules.correctilluminationcalculate.EA_EACH
    module.rescale_option.value = "No"
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_SPLINES
    )
    module.automatic_splines.value = True
    module.prepare_group(workspaces[0], {}, [1])
    module.run(workspaces[0])
    img = workspaces[0].image_set.get_image(OUTPUT_IMAGE_NAME)
    pixel_data = img.pixel_data
    diff = pixel_data - numpy.min(pixel_data) - bg
    assert not numpy.all(diff < 0.05)


def test_intermediate_images():
    """Make sure the average and dilated image flags work"""
    for average_flag, dilated_flag in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        pipeline = cellprofiler_core.pipeline.Pipeline()
        pipeline.add_listener(error_callback)
        inj_module = cellprofiler_core.modules.injectimage.InjectImage(
            "InputImage", numpy.zeros((10, 10))
        )
        inj_module.set_module_num(1)
        pipeline.add_module(inj_module)
        module = (
            cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
        )
        module.set_module_num(2)
        pipeline.add_module(module)
        module.image_name.value = "InputImage"
        module.illumination_image_name.value = "OutputImage"
        module.save_average_image.value = average_flag
        module.average_image_name.value = "AverageImage"
        module.save_dilated_image.value = dilated_flag
        module.dilated_image_name.value = "DilatedImage"
        measurements = cellprofiler_core.measurement.Measurements()
        image_set_list = cellprofiler_core.image.ImageSetList()
        workspace = cellprofiler_core.workspace.Workspace(
            pipeline, None, None, None, measurements, image_set_list
        )
        pipeline.prepare_run(workspace)
        inj_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cellprofiler_core.object.ObjectSet()
        workspace = cellprofiler_core.workspace.Workspace(
            pipeline, inj_module, image_set, object_set, measurements, image_set_list
        )
        inj_module.run(workspace)
        module.run(workspace)
        if average_flag:
            img = image_set.get_image("AverageImage")
        else:
            with pytest.raises(AssertionError):
                image_set.get_image("AverageImage")
        if dilated_flag:
            img = image_set.get_image("DilatedImage")
        else:
            with pytest.raises(AssertionError):
                image_set.get_image("DilatedImage")


def test_rescale():
    """Test basic rescaling of an image with two values"""
    input_image = numpy.ones((10, 10))
    input_image[0:5, :] *= 0.5
    image_name = "InputImage"
    expected_image = input_image * 2
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    inj_module = cellprofiler_core.modules.injectimage.InjectImage(
        image_name, input_image
    )
    inj_module.set_module_num(1)
    pipeline.add_module(inj_module)
    module = (
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
    )
    module.set_module_num(2)
    pipeline.add_module(module)
    module.image_name.value = image_name
    module.illumination_image_name.value = "OutputImage"
    module.intensity_choice.value = (
        cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
    )
    module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_NONE
    )
    module.automatic_object_width.value = (
        cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY
    )
    module.size_of_smoothing_filter.value = 10
    module.rescale_option.value = "Yes"
    module.dilate_objects.value = False
    measurements = cellprofiler_core.measurement.Measurements()
    image_set_list = cellprofiler_core.image.ImageSetList()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, None, None, None, measurements, image_set_list
    )
    pipeline.prepare_run(workspace)
    inj_module.prepare_group(workspace, {}, [1])
    module.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, inj_module, image_set, object_set, measurements, image_set_list
    )
    inj_module.run(workspace)
    module.run(workspace)
    image = image_set.get_image("OutputImage")
    assert numpy.all(image.pixel_data == expected_image)


def test_rescale_outlier():
    """Test rescaling with one low outlier"""
    input_image = numpy.ones((10, 10))
    input_image[0:5, :] *= 0.5
    input_image[0, 0] = 0.1
    image_name = "InputImage"
    expected_image = input_image * 2
    expected_image[0, 0] = 1
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_listener(error_callback)
    inj_module = cellprofiler_core.modules.injectimage.InjectImage(
        image_name, input_image
    )
    inj_module.set_module_num(1)
    pipeline.add_module(inj_module)
    module = (
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
    )
    module.set_module_num(2)
    pipeline.add_module(module)
    module.image_name.value = image_name
    module.illumination_image_name.value = "OutputImage"
    module.intensity_choice.value = (
        cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
    )
    module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
    module.smoothing_method.value = (
        cellprofiler.modules.correctilluminationcalculate.SM_NONE
    )
    module.automatic_object_width.value = (
        cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY
    )
    module.size_of_smoothing_filter.value = 10
    module.rescale_option.value = "Yes"
    module.dilate_objects.value = False
    measurements = cellprofiler_core.measurement.Measurements()
    image_set_list = cellprofiler_core.image.ImageSetList()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, None, None, None, measurements, image_set_list
    )
    pipeline.prepare_run(workspace)
    inj_module.prepare_group(workspace, {}, [1])
    module.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, inj_module, image_set, object_set, measurements, image_set_list
    )
    inj_module.run(workspace)
    module.run(workspace)
    image = image_set.get_image("OutputImage")
    assert numpy.all(image.pixel_data == expected_image)


def test_load_v2():
    file = tests.frontend.modules.get_test_resources_directory(
        "correctilluminationcalculate/v2.pipeline"
    )
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(StringIO(data))
    assert len(pipeline.modules()) == 5
    module = pipeline.modules()[0]
    assert isinstance(
        module,
        cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate,
    )
    assert module.image_name == "Masked"
    assert module.illumination_image_name == "Illum"
    assert (
        module.intensity_choice
        == cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
    )
    assert not module.dilate_objects
    assert module.object_dilation_radius == 2
    assert module.block_size == 55
    assert module.rescale_option == "No"
    assert (
        module.each_or_all == cellprofiler.modules.correctilluminationcalculate.EA_EACH
    )
    assert (
        module.smoothing_method
        == cellprofiler.modules.correctilluminationcalculate.SM_SPLINES
    )
    assert (
        module.automatic_object_width
        == cellprofiler.modules.correctilluminationcalculate.FI_AUTOMATIC
    )
    assert module.object_width == 11
    assert module.size_of_smoothing_filter == 12
    assert not module.save_average_image
    assert module.average_image_name == "IllumAverage"
    assert not module.save_dilated_image
    assert module.dilated_image_name == "IllumDilated"
    assert not module.automatic_splines
    assert (
        module.spline_bg_mode
        == MODE_BRIGHT
    )
    assert module.spline_points == 4
    assert module.spline_threshold == 2
    assert module.spline_rescale == 2
    assert module.spline_maximum_iterations == 40
    assert round(abs(module.spline_convergence.value - 0.001), 7) == 0

    assert pipeline.modules()[1].automatic_splines

    for module, spline_bg_mode in zip(
        pipeline.modules()[1:4],
        (
            MODE_AUTO,
            MODE_DARK,
            MODE_GRAY,
        ),
    ):
        assert isinstance(
            module,
            cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate,
        )
        assert module.spline_bg_mode == spline_bg_mode

    module = pipeline.modules()[4]
    assert (
        module.smoothing_method
        == cellprofiler.modules.correctilluminationcalculate.SM_CONVEX_HULL
    )
