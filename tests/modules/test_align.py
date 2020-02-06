import numpy
import six.moves

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.align
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace
import tests.modules

cellprofiler.preferences.set_headless()


def make_workspace(images, masks):
    pipeline = cellprofiler.pipeline.Pipeline()
    object_set = cellprofiler.object.ObjectSet()
    image_set_list = cellprofiler.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    module = cellprofiler.modules.align.Align()
    workspace = cellprofiler.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler.measurement.Measurements(),
        image_set_list,
    )
    for index, (pixels, mask) in enumerate(zip(images, masks)):
        if mask is None:
            image = cellprofiler.image.Image(pixels)
        else:
            image = cellprofiler.image.Image(pixels, mask=mask)
        input_name = "Channel%d" % index
        output_name = "Aligned%d" % index
        image_set.add(input_name, image)
        if index == 0:
            module.first_input_image.value = input_name
            module.first_output_image.value = output_name
        elif index == 1:
            module.second_input_image.value = input_name
            module.second_output_image.value = output_name
        else:
            module.add_image()
            ai = module.additional_images[-1]
            ai.input_image_name.value = input_name
            ai.output_image_name.value = output_name
    return workspace, module


def test_load_v2():
    with open("./tests/resources/modules/align/v2.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.align.Align)
    assert module.alignment_method == cellprofiler.modules.align.M_MUTUAL_INFORMATION
    assert module.crop_mode == cellprofiler.modules.align.C_CROP
    assert module.first_input_image == "Image1"
    assert module.second_input_image == "Image2"
    assert module.first_output_image, "AlignedImage1"
    assert module.second_output_image, "AlignedImage2"


def test_load_v3():
    with open("./tests/resources/modules/align/v3.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 3
    for module, crop_method in zip(
        pipeline.modules(),
        (
            cellprofiler.modules.align.C_SAME_SIZE,
            cellprofiler.modules.align.C_CROP,
            cellprofiler.modules.align.C_PAD,
        ),
    ):
        assert isinstance(module, cellprofiler.modules.align.Align)
        assert (
            module.alignment_method == cellprofiler.modules.align.M_MUTUAL_INFORMATION
        )
        assert module.crop_mode == crop_method
        assert module.first_input_image == "Image1"
        assert module.second_input_image == "Image2"
        assert module.first_output_image, "AlignedImage1"
        assert module.second_output_image, "AlignedImage2"


def test_crop():
    """Align two images and crop the result"""
    numpy.random.seed(0)
    shape = (50, 45)
    i, j = numpy.mgrid[0 : shape[0], 0 : shape[1]]
    for offset in ((3, 5), (-3, 5), (3, -5), (-3, -5), (0, 5), (3, 0), (0, 0)):
        #
        # Do something to give the image some information over
        # the distance, 5,5
        #
        for mask1 in (None, numpy.random.uniform(size=shape) > 0.1):
            for mask2 in (None, numpy.random.uniform(size=shape) > 0.1):
                for method in (
                    cellprofiler.modules.align.M_MUTUAL_INFORMATION,
                    cellprofiler.modules.align.M_CROSS_CORRELATION,
                ):
                    if method == cellprofiler.modules.align.M_CROSS_CORRELATION and (
                        (mask1 is not None) or (mask2 is not None)
                    ):
                        continue

                    image1 = (
                        numpy.random.randint(0, 10, size=shape).astype(float) / 10.0
                    )
                    image1[
                        numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2))
                        < 20
                    ] = 0.5
                    si1, si2 = slice_helper(offset[0], image1.shape[0])
                    sj1, sj2 = slice_helper(offset[1], image1.shape[1])
                    image2 = numpy.zeros(image1.shape)
                    if method == cellprofiler.modules.align.M_MUTUAL_INFORMATION:
                        image2[si2, sj2] = 1 - image1[si1, sj1]
                    else:
                        image2 = image1[
                            (i + shape[0] - offset[0]) % shape[0],
                            (j + shape[1] - offset[1]) % shape[1],
                        ]
                        image2 += (
                            (numpy.random.uniform(size=shape) - 0.5)
                            * 0.1
                            * numpy.std(image2)
                        )
                    if mask1 is not None:
                        image1[~mask1] = numpy.random.uniform(size=numpy.sum(~mask1))
                    if mask2 is not None:
                        image2[~mask2] = numpy.random.uniform(size=numpy.sum(~mask2))
                    workspace, module = make_workspace((image1, image2), (mask1, mask2))
                    assert isinstance(module, cellprofiler.modules.align.Align)
                    module.alignment_method.value = method
                    module.crop_mode.value = cellprofiler.modules.align.C_CROP
                    module.run(workspace)
                    output = workspace.image_set.get_image("Aligned0")
                    m = workspace.measurements
                    assert isinstance(m, cellprofiler.measurement.Measurements)
                    off_i0 = -m.get_current_image_measurement("Align_Yshift_Aligned0")
                    off_j0 = -m.get_current_image_measurement("Align_Xshift_Aligned0")
                    off_i1 = -m.get_current_image_measurement("Align_Yshift_Aligned1")
                    off_j1 = -m.get_current_image_measurement("Align_Xshift_Aligned1")

                    assert off_i0 - off_i1 == offset[0]
                    assert off_j0 - off_j1 == offset[1]
                    out_shape = output.pixel_data.shape
                    assert out_shape[0] == shape[0] - abs(offset[0])
                    assert out_shape[1] == shape[1] - abs(offset[1])
                    i_slice = single_slice_helper(-off_i0, out_shape[0])
                    j_slice = single_slice_helper(-off_j0, out_shape[1])
                    numpy.testing.assert_almost_equal(
                        image1[i_slice, j_slice], output.pixel_data
                    )
                    if mask1 is not None:
                        assert numpy.all(output.mask == mask1[i_slice, j_slice])

                    if offset[0] == 0 and offset[1] == 0:
                        assert not output.has_crop_mask
                    else:
                        temp = output.crop_mask.copy()
                        assert tuple(temp.shape) == shape
                        assert numpy.all(temp[i_slice, j_slice])
                        temp[i_slice, j_slice] = False
                        assert numpy.all(~temp)

                    output = workspace.image_set.get_image("Aligned1")
                    i_slice = single_slice_helper(-off_i1, out_shape[0])
                    j_slice = single_slice_helper(-off_j1, out_shape[1])

                    numpy.testing.assert_almost_equal(
                        image2[i_slice, j_slice], output.pixel_data
                    )
                    if mask2 is not None:
                        assert numpy.all(output.mask == mask2[i_slice, j_slice])
                    if offset[0] == 0 and offset[1] == 0:
                        assert not output.has_crop_mask
                    else:
                        temp = output.crop_mask.copy()
                        assert tuple(temp.shape) == shape
                        assert numpy.all(temp[i_slice, j_slice])
                        temp[i_slice, j_slice] = False
                        assert numpy.all(~temp)


def single_slice_helper(offset, size):
    """Return a single slice starting at the offset (or zero)"""
    if offset < 0:
        offset = 0
    return slice(offset, offset + size)


def slice_helper(offset, size):
    """Return slices for the first and second images for copying

    offset - amount to offset the second image relative to the first

    returns two slices, the first to apply to the first image, second
    to apply to the second image.
    """
    if offset < 0:
        return slice(-offset, size), slice(0, size + offset)
    else:
        return slice(0, size - offset), slice(offset, size)


def test_pad():
    """Align two images with padded output"""
    numpy.random.seed(0)
    shape = (50, 45)
    i, j = numpy.mgrid[0 : shape[0], 0 : shape[1]]
    for offset in (
        (1, 0),
        (0, 1),
        (1, 1),
        (3, 5),
        (-3, 5),
        (3, -5),
        (-3, -5),
        (0, 5),
        (3, 0),
        (0, 0),
    ):
        for mask1 in (None, numpy.random.uniform(size=shape) > 0.1):
            for mask2 in (None, numpy.random.uniform(size=shape) > 0.1):
                for method in (
                    cellprofiler.modules.align.M_MUTUAL_INFORMATION,
                    cellprofiler.modules.align.M_CROSS_CORRELATION,
                ):
                    if method == cellprofiler.modules.align.M_CROSS_CORRELATION and (
                        (mask1 is not None) or (mask2 is not None)
                    ):
                        continue
                    image1 = (
                        numpy.random.randint(0, 10, size=shape).astype(float) / 10.0
                    )
                    image1[
                        numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2))
                        < 20
                    ] = 0.5
                    si1, si2 = slice_helper(offset[0], image1.shape[0])
                    sj1, sj2 = slice_helper(offset[1], image1.shape[1])
                    image2 = numpy.zeros(image1.shape)
                    if method == cellprofiler.modules.align.M_MUTUAL_INFORMATION:
                        image2[si2, sj2] = 1 - image1[si1, sj1]
                    else:
                        image2 = image1[
                            (i + shape[0] - offset[0]) % shape[0],
                            (j + shape[1] - offset[1]) % shape[1],
                        ]
                        image2 += (
                            (numpy.random.uniform(size=shape) - 0.5)
                            * 0.1
                            * numpy.std(image2)
                        )
                    if mask1 is not None:
                        image1[~mask1] = numpy.random.uniform(size=numpy.sum(~mask1))
                    if mask2 is not None:
                        image2[~mask2] = numpy.random.uniform(size=numpy.sum(~mask2))
                    workspace, module = make_workspace((image1, image2), (mask1, mask2))
                    assert isinstance(module, cellprofiler.modules.align.Align)
                    module.alignment_method.value = method
                    module.crop_mode.value = cellprofiler.modules.align.C_PAD
                    module.run(workspace)
                    output = workspace.image_set.get_image("Aligned0")
                    m = workspace.measurements
                    assert isinstance(m, cellprofiler.measurement.Measurements)
                    off_i0 = -m.get_current_image_measurement("Align_Yshift_Aligned0")
                    off_j0 = -m.get_current_image_measurement("Align_Xshift_Aligned0")
                    off_i1 = -m.get_current_image_measurement("Align_Yshift_Aligned1")
                    off_j1 = -m.get_current_image_measurement("Align_Xshift_Aligned1")

                    assert off_i0 - off_i1 == offset[0]
                    assert off_j0 - off_j1 == offset[1]

                    i_slice = slice(off_i0, off_i0 + image1.shape[0])
                    j_slice = slice(off_j0, off_j0 + image1.shape[1])
                    numpy.testing.assert_almost_equal(
                        image1, output.pixel_data[i_slice, j_slice]
                    )
                    if mask1 is not None:
                        assert numpy.all(output.mask[i_slice, j_slice] == mask1)

                    temp = output.mask.copy()
                    temp[i_slice, j_slice] = False
                    assert numpy.all(~temp)

                    output = workspace.image_set.get_image("Aligned1")
                    i_slice = slice(off_i1, off_i1 + image2.shape[0])
                    j_slice = slice(off_j1, off_j1 + image2.shape[1])
                    numpy.testing.assert_almost_equal(
                        image2, output.pixel_data[i_slice, j_slice]
                    )
                    if mask2 is not None:
                        assert numpy.all(mask2 == output.mask[i_slice, j_slice])
                    temp = output.mask.copy()
                    temp[i_slice, j_slice] = False
                    assert numpy.all(~temp)


def test_same_size():
    """Align two images keeping sizes the same"""
    numpy.random.seed(0)
    shape = (50, 45)
    i, j = numpy.mgrid[0 : shape[0], 0 : shape[1]]
    for offset in (
        (1, 0),
        (0, 1),
        (1, 1),
        (3, 5),
        (-3, 5),
        (3, -5),
        (-3, -5),
        (0, 5),
        (3, 0),
        (0, 0),
    ):
        for mask1 in (None, numpy.random.uniform(size=shape) > 0.1):
            for mask2 in (None, numpy.random.uniform(size=shape) > 0.1):
                for method in (
                    cellprofiler.modules.align.M_MUTUAL_INFORMATION,
                    cellprofiler.modules.align.M_CROSS_CORRELATION,
                ):
                    if method == cellprofiler.modules.align.M_CROSS_CORRELATION and (
                        (mask1 is not None) or (mask2 is not None)
                    ):
                        continue
                    image1 = (
                        numpy.random.randint(0, 10, size=shape).astype(float) / 10.0
                    )
                    image1[
                        numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2))
                        < 20
                    ] = 0.5
                    si1, si2 = slice_helper(offset[0], image1.shape[0])
                    sj1, sj2 = slice_helper(offset[1], image1.shape[1])
                    image2 = numpy.zeros(image1.shape)
                    if method == cellprofiler.modules.align.M_MUTUAL_INFORMATION:
                        image2[si2, sj2] = 1 - image1[si1, sj1]
                    else:
                        image2 = image1[
                            (i + shape[0] - offset[0]) % shape[0],
                            (j + shape[1] - offset[1]) % shape[1],
                        ]
                        image2 += (
                            (numpy.random.uniform(size=shape) - 0.5)
                            * 0.1
                            * numpy.std(image2)
                        )
                    if mask1 is not None:
                        image1[~mask1] = numpy.random.uniform(size=numpy.sum(~mask1))
                    if mask2 is not None:
                        image2[~mask2] = numpy.random.uniform(size=numpy.sum(~mask2))
                    workspace, module = make_workspace((image1, image2), (mask1, mask2))
                    assert isinstance(module, cellprofiler.modules.align.Align)
                    module.alignment_method.value = method
                    module.crop_mode.value = cellprofiler.modules.align.C_SAME_SIZE
                    module.run(workspace)
                    output = workspace.image_set.get_image("Aligned0")
                    m = workspace.measurements
                    assert isinstance(m, cellprofiler.measurement.Measurements)
                    off_i0 = -m.get_current_image_measurement("Align_Yshift_Aligned0")
                    off_j0 = -m.get_current_image_measurement("Align_Xshift_Aligned0")
                    off_i1 = -m.get_current_image_measurement("Align_Yshift_Aligned1")
                    off_j1 = -m.get_current_image_measurement("Align_Xshift_Aligned1")

                    assert off_i0 - off_i1 == offset[0]
                    assert off_j0 - off_j1 == offset[1]

                    si_in, si_out = slice_same(off_i0, shape[0])
                    sj_in, sj_out = slice_same(off_j0, shape[1])
                    numpy.testing.assert_almost_equal(
                        image1[si_in, sj_in], output.pixel_data[si_out, sj_out]
                    )
                    if mask1 is not None:
                        assert numpy.all(
                            output.mask[si_out, sj_out] == mask1[si_in, sj_in]
                        )

                    temp = output.mask.copy()
                    temp[si_out, sj_out] = False
                    assert numpy.all(~temp)

                    output = workspace.image_set.get_image("Aligned1")
                    si_in, si_out = slice_same(off_i1, shape[0])
                    sj_in, sj_out = slice_same(off_j1, shape[1])
                    numpy.testing.assert_almost_equal(
                        image2[si_in, sj_in], output.pixel_data[si_out, sj_out]
                    )
                    if mask2 is not None:
                        assert numpy.all(
                            mask2[si_in, sj_in] == output.mask[si_out, sj_out]
                        )
                    temp = output.mask.copy()
                    temp[si_out, sj_out] = False
                    assert numpy.all(~temp)


def slice_same(offset, orig_size):
    if offset < 0:
        return slice(-offset, orig_size), slice(0, orig_size + offset)
    else:
        return slice(0, orig_size - offset), slice(offset, orig_size)


def test_align_similarly():
    """Align a third image similarly to the other two"""
    numpy.random.seed(0)
    shape = (53, 62)
    i, j = numpy.mgrid[0 : shape[0], 0 : shape[1]]
    for offset in ((3, 5), (-3, 5), (3, -5), (-3, -5)):
        image1 = numpy.random.randint(0, 10, size=shape).astype(float) / 10.0
        image1[
            numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20
        ] = 0.5
        si1, si2 = slice_helper(offset[0], image1.shape[0])
        sj1, sj2 = slice_helper(offset[1], image1.shape[1])
        image2 = numpy.zeros(image1.shape)
        image2 = image1[
            (i + shape[0] - offset[0]) % shape[0], (j + shape[1] - offset[1]) % shape[1]
        ]
        image2 += (numpy.random.uniform(size=shape) - 0.5) * 0.1 * numpy.std(image2)
        image3 = (i * 100 + j).astype(numpy.float32) / 10000
        workspace, module = make_workspace((image1, image2, image3), (None, None, None))
        assert isinstance(module, cellprofiler.modules.align.Align)
        module.alignment_method.value = cellprofiler.modules.align.M_CROSS_CORRELATION
        module.crop_mode.value = cellprofiler.modules.align.C_PAD
        module.additional_images[
            0
        ].align_choice.value = cellprofiler.modules.align.A_SIMILARLY
        module.run(workspace)
        output = workspace.image_set.get_image("Aligned2")
        m = workspace.measurements
        columns = module.get_measurement_columns(workspace.pipeline)
        assert len(columns) == 6
        align_measurements = [
            x
            for x in m.get_feature_names(cellprofiler.measurement.IMAGE)
            if x.startswith("Align")
        ]
        assert len(align_measurements) == 6
        assert isinstance(m, cellprofiler.measurement.Measurements)
        off_i0 = -m.get_current_image_measurement("Align_Yshift_Aligned0")
        off_j0 = -m.get_current_image_measurement("Align_Xshift_Aligned0")
        off_i1 = -m.get_current_image_measurement("Align_Yshift_Aligned1")
        off_j1 = -m.get_current_image_measurement("Align_Xshift_Aligned1")
        off_i2 = -m.get_current_image_measurement("Align_Yshift_Aligned2")
        off_j2 = -m.get_current_image_measurement("Align_Xshift_Aligned2")
        assert off_i0 - off_i1 == offset[0]
        assert off_j0 - off_j1 == offset[1]
        assert off_i0 - off_i2 == offset[0]
        assert off_j0 - off_j2 == offset[1]

        i_slice = single_slice_helper(off_i2, shape[0])
        j_slice = single_slice_helper(off_j2, shape[1])
        numpy.testing.assert_almost_equal(output.pixel_data[i_slice, j_slice], image3)


def test_align_separately():
    """Align a third image to the first image"""
    numpy.random.seed(0)
    shape = (47, 53)
    i, j = numpy.mgrid[0 : shape[0], 0 : shape[1]]
    for offset in ((3, 5), (-3, 5), (3, -5), (-3, -5)):
        image1 = numpy.random.uniform(size=shape).astype(numpy.float32)
        image2 = image1[
            (i + shape[0] - offset[0] - 5) % shape[0],
            (j + shape[1] - offset[1] - 5) % shape[1],
        ]
        image3 = image1[
            (i + shape[0] - offset[0]) % shape[0], (j + shape[1] - offset[1]) % shape[1]
        ]
        workspace, module = make_workspace((image1, image2, image3), (None, None, None))
        assert isinstance(module, cellprofiler.modules.align.Align)
        module.alignment_method.value = cellprofiler.modules.align.M_CROSS_CORRELATION
        module.crop_mode.value = cellprofiler.modules.align.C_PAD
        module.additional_images[
            0
        ].align_choice.value = cellprofiler.modules.align.A_SEPARATELY
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cellprofiler.measurement.Measurements)
        off_i0 = -m.get_current_image_measurement("Align_Yshift_Aligned0")
        off_j0 = -m.get_current_image_measurement("Align_Xshift_Aligned0")
        off_i1 = -m.get_current_image_measurement("Align_Yshift_Aligned1")
        off_j1 = -m.get_current_image_measurement("Align_Xshift_Aligned1")
        off_i2 = -m.get_current_image_measurement("Align_Yshift_Aligned2")
        off_j2 = -m.get_current_image_measurement("Align_Xshift_Aligned2")
        assert off_i0 - off_i1 == offset[0] + 5
        assert off_j0 - off_j1 == offset[1] + 5
        assert off_i0 - off_i2 == offset[0]
        assert off_j0 - off_j2 == offset[1]
        output = workspace.image_set.get_image("Aligned2")
        i_slice = single_slice_helper(off_i2, shape[0])
        j_slice = single_slice_helper(off_j2, shape[1])
        numpy.testing.assert_almost_equal(output.pixel_data[i_slice, j_slice], image3)


def test_align_color():
    numpy.random.seed(0)
    shape = (50, 45, 3)
    i, j = numpy.mgrid[0 : shape[0], 0 : shape[1]]
    for offset in (
        (1, 0),
        (0, 1),
        (1, 1),
        (3, 5),
        (-3, 5),
        (3, -5),
        (-3, -5),
        (0, 5),
        (3, 0),
        (0, 0),
    ):
        for mask1 in (None, numpy.random.uniform(size=shape[:2]) > 0.1):
            for mask2 in (None, numpy.random.uniform(size=shape[:2]) > 0.1):
                for method in (
                    cellprofiler.modules.align.M_MUTUAL_INFORMATION,
                    cellprofiler.modules.align.M_CROSS_CORRELATION,
                ):
                    if method == cellprofiler.modules.align.M_CROSS_CORRELATION and (
                        (mask1 is not None) or (mask2 is not None)
                    ):
                        continue
                    image1 = numpy.dstack(
                        [
                            numpy.random.randint(0, 10, size=shape[:2]).astype(float)
                            / 10.0
                        ]
                        * 3
                    )
                    image1[
                        numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2))
                        < 20,
                        :,
                    ] = 0.5
                    si1, si2 = slice_helper(offset[0], image1.shape[0])
                    sj1, sj2 = slice_helper(offset[1], image1.shape[1])
                    image2 = numpy.zeros(image1.shape)
                    if method == cellprofiler.modules.align.M_MUTUAL_INFORMATION:
                        image2[si2, sj2, :] = 1 - image1[si1, sj1, :]
                    else:
                        image2 = image1[
                            (i + shape[0] - offset[0]) % shape[0],
                            (j + shape[1] - offset[1]) % shape[1],
                            :,
                        ]
                        image2 += (
                            (numpy.random.uniform(size=shape) - 0.5)
                            * 0.1
                            * numpy.std(image2)
                        )
                    if mask1 is not None:
                        image1[~mask1, :] = numpy.random.uniform(
                            size=(numpy.sum(~mask1), shape[2])
                        )
                    if mask2 is not None:
                        image2[~mask2, :] = numpy.random.uniform(
                            size=(numpy.sum(~mask2), shape[2])
                        )
                    workspace, module = make_workspace((image1, image2), (mask1, mask2))
                    assert isinstance(module, cellprofiler.modules.align.Align)
                    module.alignment_method.value = method
                    module.crop_mode.value = cellprofiler.modules.align.C_SAME_SIZE
                    module.run(workspace)
                    output = workspace.image_set.get_image("Aligned0")
                    m = workspace.measurements
                    assert isinstance(m, cellprofiler.measurement.Measurements)
                    off_i0 = -m.get_current_image_measurement("Align_Yshift_Aligned0")
                    off_j0 = -m.get_current_image_measurement("Align_Xshift_Aligned0")
                    off_i1 = -m.get_current_image_measurement("Align_Yshift_Aligned1")
                    off_j1 = -m.get_current_image_measurement("Align_Xshift_Aligned1")

                    assert off_i0 - off_i1 == offset[0]
                    assert off_j0 - off_j1 == offset[1]

                    si_in, si_out = slice_same(off_i0, shape[0])
                    sj_in, sj_out = slice_same(off_j0, shape[1])
                    numpy.testing.assert_almost_equal(
                        image1[si_in, sj_in, :], output.pixel_data[si_out, sj_out, :]
                    )
                    if mask1 is not None:
                        assert numpy.all(
                            output.mask[si_out, sj_out] == mask1[si_in, sj_in]
                        )

                    temp = output.mask.copy()
                    temp[si_out, sj_out] = False
                    assert numpy.all(~temp)

                    output = workspace.image_set.get_image("Aligned1")
                    si_in, si_out = slice_same(off_i1, shape[0])
                    sj_in, sj_out = slice_same(off_j1, shape[1])
                    numpy.testing.assert_almost_equal(
                        image2[si_in, sj_in, :], output.pixel_data[si_out, sj_out, :]
                    )
                    if mask2 is not None:
                        assert numpy.all(
                            mask2[si_in, sj_in] == output.mask[si_out, sj_out]
                        )
                    temp = output.mask.copy()
                    temp[si_out, sj_out] = False
                    assert numpy.all(~temp)


def test_align_binary():
    numpy.random.seed(0)
    shape = (50, 45)
    i, j = numpy.mgrid[0 : shape[0], 0 : shape[1]]
    for offset in (
        (1, 0),
        (0, 1),
        (1, 1),
        (3, 5),
        (-3, 5),
        (3, -5),
        (-3, -5),
        (0, 5),
        (3, 0),
        (0, 0),
    ):
        for mask1 in (None, numpy.random.uniform(size=shape) > 0.1):
            for mask2 in (None, numpy.random.uniform(size=shape) > 0.1):
                for method in (
                    cellprofiler.modules.align.M_MUTUAL_INFORMATION,
                    cellprofiler.modules.align.M_CROSS_CORRELATION,
                ):
                    if method == cellprofiler.modules.align.M_CROSS_CORRELATION and (
                        (mask1 is not None) or (mask2 is not None)
                    ):
                        continue
                    image1 = numpy.random.randint(0, 1, size=shape).astype(bool)
                    image1[
                        numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2))
                        < 10
                    ] = True
                    si1, si2 = slice_helper(offset[0], image1.shape[0])
                    sj1, sj2 = slice_helper(offset[1], image1.shape[1])
                    image2 = numpy.zeros(image1.shape, bool)
                    if method == cellprofiler.modules.align.M_MUTUAL_INFORMATION:
                        image2[si2, sj2] = 1 - image1[si1, sj1]
                    else:
                        image2 = image1[
                            (i + shape[0] - offset[0]) % shape[0],
                            (j + shape[1] - offset[1]) % shape[1],
                        ]
                    if mask1 is not None:
                        image1[~mask1] = numpy.random.uniform(size=numpy.sum(~mask1))
                    if mask2 is not None:
                        image2[~mask2] = numpy.random.uniform(size=numpy.sum(~mask2))
                    workspace, module = make_workspace((image1, image2), (mask1, mask2))
                    assert isinstance(module, cellprofiler.modules.align.Align)
                    module.alignment_method.value = method
                    module.crop_mode.value = cellprofiler.modules.align.C_SAME_SIZE
                    module.run(workspace)
                    output = workspace.image_set.get_image("Aligned0")
                    m = workspace.measurements
                    assert isinstance(m, cellprofiler.measurement.Measurements)
                    off_i0 = -m.get_current_image_measurement("Align_Yshift_Aligned0")
                    off_j0 = -m.get_current_image_measurement("Align_Xshift_Aligned0")
                    off_i1 = -m.get_current_image_measurement("Align_Yshift_Aligned1")
                    off_j1 = -m.get_current_image_measurement("Align_Xshift_Aligned1")

                    assert off_i0 - off_i1 == offset[0]
                    assert off_j0 - off_j1 == offset[1]

                    si_in, si_out = slice_same(off_i0, shape[0])
                    sj_in, sj_out = slice_same(off_j0, shape[1])
                    assert output.pixel_data.dtype.kind == "b"
                    numpy.testing.assert_equal(
                        image1[si_in, sj_in], output.pixel_data[si_out, sj_out]
                    )
                    if mask1 is not None:
                        assert numpy.all(
                            output.mask[si_out, sj_out] == mask1[si_in, sj_in]
                        )

                    temp = output.mask.copy()
                    temp[si_out, sj_out] = False
                    assert numpy.all(~temp)

                    output = workspace.image_set.get_image("Aligned1")
                    si_in, si_out = slice_same(off_i1, shape[0])
                    sj_in, sj_out = slice_same(off_j1, shape[1])
                    numpy.testing.assert_equal(
                        image2[si_in, sj_in], output.pixel_data[si_out, sj_out]
                    )
                    if mask2 is not None:
                        assert numpy.all(
                            mask2[si_in, sj_in] == output.mask[si_out, sj_out]
                        )
                    temp = output.mask.copy()
                    temp[si_out, sj_out] = False
                    assert numpy.all(~temp)


def test_measurement_columns():
    workspace, module = make_workspace(
        (numpy.zeros((10, 10)), numpy.zeros((10, 10)), numpy.zeros((10, 10))),
        (None, None, None),
    )
    assert isinstance(module, cellprofiler.modules.align.Align)
    columns = module.get_measurement_columns(workspace.pipeline)
    assert len(columns) == 6
    for i in range(3):
        for axis in ("X", "Y"):
            feature = cellprofiler.modules.align.MEASUREMENT_FORMAT % (
                axis,
                "Aligned%d" % i,
            )
            assert feature in [c[1] for c in columns]
    assert all([c[0] == cellprofiler.measurement.IMAGE for c in columns])
    assert all([c[2] == cellprofiler.measurement.COLTYPE_INTEGER for c in columns])


def test_categories():
    workspace, module = make_workspace(
        (numpy.zeros((10, 10)), numpy.zeros((10, 10)), numpy.zeros((10, 10))),
        (None, None, None),
    )
    assert isinstance(module, cellprofiler.modules.align.Align)
    c = module.get_categories(workspace.pipeline, cellprofiler.measurement.IMAGE)
    assert len(c) == 1
    assert c[0] == cellprofiler.modules.align.C_ALIGN

    c = module.get_categories(workspace.pipeline, "Aligned0")
    assert len(c) == 0


def test_measurements():
    workspace, module = make_workspace(
        (numpy.zeros((10, 10)), numpy.zeros((10, 10)), numpy.zeros((10, 10))),
        (None, None, None),
    )
    assert isinstance(module, cellprofiler.modules.align.Align)
    m = module.get_measurements(
        workspace.pipeline,
        cellprofiler.measurement.IMAGE,
        cellprofiler.modules.align.C_ALIGN,
    )
    assert len(m) == 2
    assert "Xshift" in m
    assert "Yshift" in m


def test_measurement_images():
    workspace, module = make_workspace(
        (numpy.zeros((10, 10)), numpy.zeros((10, 10)), numpy.zeros((10, 10))),
        (None, None, None),
    )
    assert isinstance(module, cellprofiler.modules.align.Align)
    for measurement in ("Xshift", "Yshift"):
        image_names = module.get_measurement_images(
            workspace.pipeline,
            cellprofiler.measurement.IMAGE,
            cellprofiler.modules.align.C_ALIGN,
            measurement,
        )
        assert len(image_names) == 3
        for i in range(3):
            assert "Aligned%d" % i in image_names


def test_align_self():
    """Align an image from the fly screen against it

    This is a regression test for the bug, IMG-284
    """
    image = tests.modules.read_example_image("ExampleFlyImages", "01_POS002_D.TIF")
    image = image[0:300, 0:300]  # make smaller so as to be faster
    workspace, module = make_workspace((image, image), (None, None))
    module.alignment_method.value = cellprofiler.modules.align.M_MUTUAL_INFORMATION
    module.crop_mode.value = cellprofiler.modules.align.C_PAD
    module.run(workspace)
    m = workspace.measurements
    assert m.get_current_image_measurement("Align_Xshift_Aligned1") == 0
    assert m.get_current_image_measurement("Align_Yshift_Aligned1") == 0


def test_different_sizes_crop():
    """Test align with images of different sizes

    regression test of img-1300
    """
    numpy.random.seed(61)
    shape = (61, 43)
    for method in (
        cellprofiler.modules.align.M_CROSS_CORRELATION,
        cellprofiler.modules.align.M_MUTUAL_INFORMATION,
    ):
        i, j = numpy.mgrid[0 : shape[0], 0 : shape[1]]
        image1 = numpy.random.randint(0, 10, size=shape).astype(float) / 10.0
        image1[
            numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20
        ] = 0.5
        image2 = image1[2:-2, 2:-2]
        for order, i1_name, i2_name in (
            ((image1, image2), "Aligned0", "Aligned1"),
            ((image2, image1), "Aligned1", "Aligned0"),
        ):
            workspace, module = make_workspace(order, (None, None))
            assert isinstance(module, cellprofiler.modules.align.Align)
            module.alignment_method.value = method
            module.crop_mode.value = cellprofiler.modules.align.C_CROP
            module.run(workspace)
            i1 = workspace.image_set.get_image(i1_name)
            assert isinstance(i1, cellprofiler.image.Image)
            p1 = i1.pixel_data
            i2 = workspace.image_set.get_image(i2_name)
            p2 = i2.pixel_data
            assert tuple(p1.shape) == tuple(p2.shape)
            assert numpy.all(p1 == p2)
            assert i1.has_crop_mask
            crop_mask = numpy.zeros(shape, bool)
            crop_mask[2:-2, 2:-2] = True
            assert numpy.all(i1.crop_mask == crop_mask)
            assert not i2.has_crop_mask


def test_different_sizes_pad():
    """Test align with images of different sizes

    regression test of img-1300
    """
    numpy.random.seed(612)
    shape = (61, 43)
    i, j = numpy.mgrid[0 : shape[0], 0 : shape[1]]
    image1 = numpy.random.randint(0, 10, size=shape).astype(float) / 10.0
    image1[numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20] = 0.5
    image2 = image1[2:-2, 2:-2]
    workspace, module = make_workspace((image1, image2), (None, None))
    assert isinstance(module, cellprofiler.modules.align.Align)
    module.crop_mode.value = cellprofiler.modules.align.C_PAD
    module.run(workspace)
    i1 = workspace.image_set.get_image("Aligned0")
    assert isinstance(i1, cellprofiler.image.Image)
    p1 = i1.pixel_data
    i2 = workspace.image_set.get_image("Aligned1")
    p2 = i2.pixel_data
    assert tuple(p1.shape) == tuple(p2.shape)
    assert numpy.all(p1[2:-2, 2:-2] == p2[2:-2, 2:-2])
    assert not i1.has_mask
    mask = numpy.zeros(shape, bool)
    mask[2:-2, 2:-2] = True
    assert i2.has_mask
    assert numpy.all(mask == i2.mask)
