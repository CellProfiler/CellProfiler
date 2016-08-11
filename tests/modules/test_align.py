import StringIO
import unittest

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.align
import cellprofiler.modules.loadimages
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import numpy
import tests.modules

cellprofiler.preferences.set_headless()


class TestAlign(unittest.TestCase):
    def make_workspace(self, images, masks):
        pipeline = cellprofiler.pipeline.Pipeline()
        object_set = cellprofiler.region.Set()
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        module = cellprofiler.modules.align.Align()
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     module,
                                                     image_set,
                                                     object_set,
                                                     cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        for index, (pixels, mask) in enumerate(zip(images, masks)):
            if mask is None:
                image = cellprofiler.image.Image(pixels)
            else:
                image = cellprofiler.image.Image(pixels, mask=mask)
            input_name = 'Channel%d' % index
            output_name = 'Aligned%d' % index
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

    def test_01_03_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8945

Align:[module_num:1|svn_version:'8942'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
Select the alignment method:Mutual Information
Crop output images to retain just the aligned regions?:Yes
Select the first input image:Image1
Name the first output image:AlignedImage1
Select the second input image:Image2
Name the second output image:AlignedImage2
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
        self.assertEqual(module.alignment_method, cellprofiler.modules.align.M_MUTUAL_INFORMATION)
        self.assertEqual(module.crop_mode, cellprofiler.modules.align.C_CROP)
        self.assertEqual(module.first_input_image, "Image1")
        self.assertEqual(module.second_input_image, "Image2")
        self.assertTrue(module.first_output_image, "AlignedImage1")
        self.assertTrue(module.second_output_image, "AlignedImage2")

    def test_01_04_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8945

Align:[module_num:1|svn_version:'8942'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
Select the alignment method:Mutual Information
Crop mode:Keep size
Select the first input image:Image1
Name the first output image:AlignedImage1
Select the second input image:Image2
Name the second output image:AlignedImage2

Align:[module_num:1|svn_version:'8942'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
Select the alignment method:Mutual Information
Crop mode:Crop to aligned region
Select the first input image:Image1
Name the first output image:AlignedImage1
Select the second input image:Image2
Name the second output image:AlignedImage2

Align:[module_num:1|svn_version:'8942'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
Select the alignment method:Mutual Information
Crop mode:Pad images
Select the first input image:Image1
Name the first output image:AlignedImage1
Select the second input image:Image2
Name the second output image:AlignedImage2
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 3)
        for module, crop_method in zip(
                pipeline.modules(),
                (cellprofiler.modules.align.C_SAME_SIZE, cellprofiler.modules.align.C_CROP, cellprofiler.modules.align.C_PAD)):
            self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
            self.assertEqual(module.alignment_method, cellprofiler.modules.align.M_MUTUAL_INFORMATION)
            self.assertEqual(module.crop_mode, crop_method)
            self.assertEqual(module.first_input_image, "Image1")
            self.assertEqual(module.second_input_image, "Image2")
            self.assertTrue(module.first_output_image, "AlignedImage1")
            self.assertTrue(module.second_output_image, "AlignedImage2")

    def test_02_01_crop(self):
        """Align two images and crop the result"""
        numpy.random.seed(0)
        shape = (50, 45)
        i, j = numpy.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((3, 5), (-3, 5), (3, -5), (-3, -5), (0, 5), (3, 0), (0, 0)):
            #
            # Do something to give the image some information over
            # the distance, 5,5
            #
            for mask1 in (None,
                          numpy.random.uniform(size=shape) > .1):
                for mask2 in (None,
                              numpy.random.uniform(size=shape) > .1):
                    for method in (cellprofiler.modules.align.M_MUTUAL_INFORMATION, cellprofiler.modules.align.M_CROSS_CORRELATION):
                        if method == cellprofiler.modules.align.M_CROSS_CORRELATION and (
                                    (mask1 is not None) or (mask2 is not None)):
                            continue

                        image1 = numpy.random.randint(0, 10, size=shape).astype(float) / 10.0
                        image1[numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20] = .5
                        si1, si2 = self.slice_helper(offset[0], image1.shape[0])
                        sj1, sj2 = self.slice_helper(offset[1], image1.shape[1])
                        image2 = numpy.zeros(image1.shape)
                        if method == cellprofiler.modules.align.M_MUTUAL_INFORMATION:
                            image2[si2, sj2] = 1 - image1[si1, sj1]
                        else:
                            image2 = image1[(i + shape[0] - offset[0]) % shape[0],
                                            (j + shape[1] - offset[1]) % shape[1]]
                            image2 += (numpy.random.uniform(size=shape) - .5) * .1 * numpy.std(image2)
                        if mask1 is not None:
                            image1[~ mask1] = numpy.random.uniform(size=numpy.sum(~mask1))
                        if mask2 is not None:
                            image2[~ mask2] = numpy.random.uniform(size=numpy.sum(~mask2))
                        workspace, module = self.make_workspace((image1, image2),
                                                                (mask1, mask2))
                        self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
                        module.alignment_method.value = method
                        module.crop_mode.value = cellprofiler.modules.align.C_CROP
                        module.run(workspace)
                        output = workspace.image_set.get_image('Aligned0')
                        m = workspace.measurements
                        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
                        off_i0 = -m.get_current_image_measurement('Align_Yshift_Aligned0')
                        off_j0 = -m.get_current_image_measurement('Align_Xshift_Aligned0')
                        off_i1 = -m.get_current_image_measurement('Align_Yshift_Aligned1')
                        off_j1 = -m.get_current_image_measurement('Align_Xshift_Aligned1')

                        self.assertEqual(off_i0 - off_i1, offset[0])
                        self.assertEqual(off_j0 - off_j1, offset[1])
                        out_shape = output.pixel_data.shape
                        self.assertEqual(out_shape[0],
                                         shape[0] - abs(offset[0]))
                        self.assertEqual(out_shape[1],
                                         shape[1] - abs(offset[1]))
                        i_slice = self.single_slice_helper(-off_i0, out_shape[0])
                        j_slice = self.single_slice_helper(-off_j0, out_shape[1])
                        numpy.testing.assert_almost_equal(
                                image1[i_slice, j_slice], output.pixel_data)
                        if mask1 is not None:
                            self.assertTrue(numpy.all(output.mask == mask1[i_slice, j_slice]))

                        if offset[0] == 0 and offset[1] == 0:
                            self.assertFalse(output.has_crop_mask)
                        else:
                            temp = output.crop_mask.copy()
                            self.assertEqual(tuple(temp.shape), shape)
                            self.assertTrue(numpy.all(temp[i_slice, j_slice]))
                            temp[i_slice, j_slice] = False
                            self.assertTrue(numpy.all(~ temp))

                        output = workspace.image_set.get_image("Aligned1")
                        i_slice = self.single_slice_helper(-off_i1, out_shape[0])
                        j_slice = self.single_slice_helper(-off_j1, out_shape[1])

                        numpy.testing.assert_almost_equal(
                                image2[i_slice, j_slice], output.pixel_data)
                        if mask2 is not None:
                            self.assertTrue(numpy.all(output.mask == mask2[i_slice, j_slice]))
                        if offset[0] == 0 and offset[1] == 0:
                            self.assertFalse(output.has_crop_mask)
                        else:
                            temp = output.crop_mask.copy()
                            self.assertEqual(tuple(temp.shape), shape)
                            self.assertTrue(numpy.all(temp[i_slice, j_slice]))
                            temp[i_slice, j_slice] = False
                            self.assertTrue(numpy.all(~ temp))

    def single_slice_helper(self, offset, size):
        """Return a single slice starting at the offset (or zero)"""
        if offset < 0:
            offset = 0
        return slice(offset, offset + size)

    def slice_helper(self, offset, size):
        """Return slices for the first and second images for copying

        offset - amount to offset the second image relative to the first

        returns two slices, the first to apply to the first image, second
        to apply to the second image.
        """
        if offset < 0:
            return slice(-offset, size), slice(0, size + offset)
        else:
            return slice(0, size - offset), slice(offset, size)

    def test_02_02_pad(self):
        """Align two images with padded output"""
        numpy.random.seed(0)
        shape = (50, 45)
        i, j = numpy.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((1, 0), (0, 1), (1, 1), (3, 5), (-3, 5), (3, -5), (-3, -5), (0, 5), (3, 0), (0, 0)):
            for mask1 in (None,
                          numpy.random.uniform(size=shape) > .1):
                for mask2 in (None,
                              numpy.random.uniform(size=shape) > .1):
                    for method in (cellprofiler.modules.align.M_MUTUAL_INFORMATION, cellprofiler.modules.align.M_CROSS_CORRELATION):
                        if method == cellprofiler.modules.align.M_CROSS_CORRELATION and (
                                    (mask1 is not None) or (mask2 is not None)):
                            continue
                        image1 = numpy.random.randint(0, 10, size=shape).astype(float) / 10.0
                        image1[numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20] = .5
                        si1, si2 = self.slice_helper(offset[0], image1.shape[0])
                        sj1, sj2 = self.slice_helper(offset[1], image1.shape[1])
                        image2 = numpy.zeros(image1.shape)
                        if method == cellprofiler.modules.align.M_MUTUAL_INFORMATION:
                            image2[si2, sj2] = 1 - image1[si1, sj1]
                        else:
                            image2 = image1[(i + shape[0] - offset[0]) % shape[0],
                                            (j + shape[1] - offset[1]) % shape[1]]
                            image2 += (numpy.random.uniform(size=shape) - .5) * .1 * numpy.std(image2)
                        if mask1 is not None:
                            image1[~ mask1] = numpy.random.uniform(size=numpy.sum(~mask1))
                        if mask2 is not None:
                            image2[~ mask2] = numpy.random.uniform(size=numpy.sum(~mask2))
                        workspace, module = self.make_workspace((image1, image2),
                                                                (mask1, mask2))
                        self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
                        module.alignment_method.value = method
                        module.crop_mode.value = cellprofiler.modules.align.C_PAD
                        module.run(workspace)
                        output = workspace.image_set.get_image('Aligned0')
                        m = workspace.measurements
                        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
                        off_i0 = -m.get_current_image_measurement('Align_Yshift_Aligned0')
                        off_j0 = -m.get_current_image_measurement('Align_Xshift_Aligned0')
                        off_i1 = -m.get_current_image_measurement('Align_Yshift_Aligned1')
                        off_j1 = -m.get_current_image_measurement('Align_Xshift_Aligned1')

                        self.assertEqual(off_i0 - off_i1, offset[0])
                        self.assertEqual(off_j0 - off_j1, offset[1])

                        i_slice = slice(off_i0, off_i0 + image1.shape[0])
                        j_slice = slice(off_j0, off_j0 + image1.shape[1])
                        numpy.testing.assert_almost_equal(
                                image1, output.pixel_data[i_slice, j_slice])
                        if mask1 is not None:
                            self.assertTrue(numpy.all(output.mask[i_slice, j_slice] == mask1))

                        temp = output.mask.copy()
                        temp[i_slice, j_slice] = False
                        self.assertTrue(numpy.all(~ temp))

                        output = workspace.image_set.get_image("Aligned1")
                        i_slice = slice(off_i1, off_i1 + image2.shape[0])
                        j_slice = slice(off_j1, off_j1 + image2.shape[1])
                        numpy.testing.assert_almost_equal(
                                image2, output.pixel_data[i_slice, j_slice])
                        if mask2 is not None:
                            self.assertTrue(numpy.all(mask2 == output.mask[i_slice, j_slice]))
                        temp = output.mask.copy()
                        temp[i_slice, j_slice] = False
                        self.assertTrue(numpy.all(~ temp))

    def test_02_03_same_size(self):
        """Align two images keeping sizes the same"""
        numpy.random.seed(0)
        shape = (50, 45)
        i, j = numpy.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((1, 0), (0, 1), (1, 1), (3, 5), (-3, 5), (3, -5), (-3, -5), (0, 5), (3, 0), (0, 0)):
            for mask1 in (None,
                          numpy.random.uniform(size=shape) > .1):
                for mask2 in (None,
                              numpy.random.uniform(size=shape) > .1):
                    for method in (cellprofiler.modules.align.M_MUTUAL_INFORMATION, cellprofiler.modules.align.M_CROSS_CORRELATION):
                        if method == cellprofiler.modules.align.M_CROSS_CORRELATION and (
                                    (mask1 is not None) or (mask2 is not None)):
                            continue
                        image1 = numpy.random.randint(0, 10, size=shape).astype(float) / 10.0
                        image1[numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20] = .5
                        si1, si2 = self.slice_helper(offset[0], image1.shape[0])
                        sj1, sj2 = self.slice_helper(offset[1], image1.shape[1])
                        image2 = numpy.zeros(image1.shape)
                        if method == cellprofiler.modules.align.M_MUTUAL_INFORMATION:
                            image2[si2, sj2] = 1 - image1[si1, sj1]
                        else:
                            image2 = image1[(i + shape[0] - offset[0]) % shape[0],
                                            (j + shape[1] - offset[1]) % shape[1]]
                            image2 += (numpy.random.uniform(size=shape) - .5) * .1 * numpy.std(image2)
                        if mask1 is not None:
                            image1[~ mask1] = numpy.random.uniform(size=numpy.sum(~mask1))
                        if mask2 is not None:
                            image2[~ mask2] = numpy.random.uniform(size=numpy.sum(~mask2))
                        workspace, module = self.make_workspace((image1, image2),
                                                                (mask1, mask2))
                        self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
                        module.alignment_method.value = method
                        module.crop_mode.value = cellprofiler.modules.align.C_SAME_SIZE
                        module.run(workspace)
                        output = workspace.image_set.get_image('Aligned0')
                        m = workspace.measurements
                        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
                        off_i0 = -m.get_current_image_measurement('Align_Yshift_Aligned0')
                        off_j0 = -m.get_current_image_measurement('Align_Xshift_Aligned0')
                        off_i1 = -m.get_current_image_measurement('Align_Yshift_Aligned1')
                        off_j1 = -m.get_current_image_measurement('Align_Xshift_Aligned1')

                        self.assertEqual(off_i0 - off_i1, offset[0])
                        self.assertEqual(off_j0 - off_j1, offset[1])

                        si_in, si_out = self.slice_same(off_i0, shape[0])
                        sj_in, sj_out = self.slice_same(off_j0, shape[1])
                        numpy.testing.assert_almost_equal(
                                image1[si_in, sj_in], output.pixel_data[si_out, sj_out])
                        if mask1 is not None:
                            self.assertTrue(numpy.all(output.mask[si_out, sj_out] == mask1[si_in, sj_in]))

                        temp = output.mask.copy()
                        temp[si_out, sj_out] = False
                        self.assertTrue(numpy.all(~ temp))

                        output = workspace.image_set.get_image("Aligned1")
                        si_in, si_out = self.slice_same(off_i1, shape[0])
                        sj_in, sj_out = self.slice_same(off_j1, shape[1])
                        numpy.testing.assert_almost_equal(
                                image2[si_in, sj_in], output.pixel_data[si_out, sj_out])
                        if mask2 is not None:
                            self.assertTrue(numpy.all(mask2[si_in, sj_in] == output.mask[si_out, sj_out]))
                        temp = output.mask.copy()
                        temp[si_out, sj_out] = False
                        self.assertTrue(numpy.all(~ temp))

    def slice_same(self, offset, orig_size):
        if offset < 0:
            return slice(-offset, orig_size), slice(0, orig_size + offset)
        else:
            return slice(0, orig_size - offset), slice(offset, orig_size)

    def test_03_01_align_similarly(self):
        """Align a third image similarly to the other two"""
        numpy.random.seed(0)
        shape = (53, 62)
        i, j = numpy.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((3, 5), (-3, 5), (3, -5), (-3, -5)):
            image1 = numpy.random.randint(0, 10, size=shape).astype(float) / 10.0
            image1[numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20] = .5
            si1, si2 = self.slice_helper(offset[0], image1.shape[0])
            sj1, sj2 = self.slice_helper(offset[1], image1.shape[1])
            image2 = numpy.zeros(image1.shape)
            image2 = image1[(i + shape[0] - offset[0]) % shape[0],
                            (j + shape[1] - offset[1]) % shape[1]]
            image2 += (numpy.random.uniform(size=shape) - .5) * .1 * numpy.std(image2)
            image3 = (i * 100 + j).astype(numpy.float32) / 10000
            workspace, module = self.make_workspace((image1, image2, image3),
                                                    (None, None, None))
            self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
            module.alignment_method.value = cellprofiler.modules.align.M_CROSS_CORRELATION
            module.crop_mode.value = cellprofiler.modules.align.C_PAD
            module.additional_images[0].align_choice.value = cellprofiler.modules.align.A_SIMILARLY
            module.run(workspace)
            output = workspace.image_set.get_image('Aligned2')
            m = workspace.measurements
            columns = module.get_measurement_columns(workspace.pipeline)
            self.assertEqual(len(columns), 6)
            align_measurements = [x for x in m.get_feature_names(cellprofiler.measurement.IMAGE)
                                  if x.startswith('Align')]
            self.assertEqual(len(align_measurements), 6)
            self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
            off_i0 = -m.get_current_image_measurement('Align_Yshift_Aligned0')
            off_j0 = -m.get_current_image_measurement('Align_Xshift_Aligned0')
            off_i1 = -m.get_current_image_measurement('Align_Yshift_Aligned1')
            off_j1 = -m.get_current_image_measurement('Align_Xshift_Aligned1')
            off_i2 = -m.get_current_image_measurement('Align_Yshift_Aligned2')
            off_j2 = -m.get_current_image_measurement('Align_Xshift_Aligned2')
            self.assertEqual(off_i0 - off_i1, offset[0])
            self.assertEqual(off_j0 - off_j1, offset[1])
            self.assertEqual(off_i0 - off_i2, offset[0])
            self.assertEqual(off_j0 - off_j2, offset[1])

            i_slice = self.single_slice_helper(off_i2, shape[0])
            j_slice = self.single_slice_helper(off_j2, shape[1])
            numpy.testing.assert_almost_equal(output.pixel_data[i_slice, j_slice],
                                              image3)

    def test_03_02_align_separately(self):
        """Align a third image to the first image"""
        numpy.random.seed(0)
        shape = (47, 53)
        i, j = numpy.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((3, 5), (-3, 5), (3, -5), (-3, -5)):
            image1 = numpy.random.uniform(size=shape).astype(numpy.float32)
            image2 = image1[(i + shape[0] - offset[0] - 5) % shape[0],
                            (j + shape[1] - offset[1] - 5) % shape[1]]
            image3 = image1[(i + shape[0] - offset[0]) % shape[0],
                            (j + shape[1] - offset[1]) % shape[1]]
            workspace, module = self.make_workspace((image1, image2, image3),
                                                    (None, None, None))
            self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
            module.alignment_method.value = cellprofiler.modules.align.M_CROSS_CORRELATION
            module.crop_mode.value = cellprofiler.modules.align.C_PAD
            module.additional_images[0].align_choice.value = cellprofiler.modules.align.A_SEPARATELY
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
            off_i0 = -m.get_current_image_measurement('Align_Yshift_Aligned0')
            off_j0 = -m.get_current_image_measurement('Align_Xshift_Aligned0')
            off_i1 = -m.get_current_image_measurement('Align_Yshift_Aligned1')
            off_j1 = -m.get_current_image_measurement('Align_Xshift_Aligned1')
            off_i2 = -m.get_current_image_measurement('Align_Yshift_Aligned2')
            off_j2 = -m.get_current_image_measurement('Align_Xshift_Aligned2')
            self.assertEqual(off_i0 - off_i1, offset[0] + 5)
            self.assertEqual(off_j0 - off_j1, offset[1] + 5)
            self.assertEqual(off_i0 - off_i2, offset[0])
            self.assertEqual(off_j0 - off_j2, offset[1])
            output = workspace.image_set.get_image('Aligned2')
            i_slice = self.single_slice_helper(off_i2, shape[0])
            j_slice = self.single_slice_helper(off_j2, shape[1])
            numpy.testing.assert_almost_equal(output.pixel_data[i_slice, j_slice],
                                              image3)

    def test_03_03_align_color(self):
        numpy.random.seed(0)
        shape = (50, 45, 3)
        i, j = numpy.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((1, 0), (0, 1), (1, 1), (3, 5), (-3, 5), (3, -5), (-3, -5), (0, 5), (3, 0), (0, 0)):
            for mask1 in (None,
                          numpy.random.uniform(size=shape[:2]) > .1):
                for mask2 in (None,
                              numpy.random.uniform(size=shape[:2]) > .1):
                    for method in (cellprofiler.modules.align.M_MUTUAL_INFORMATION, cellprofiler.modules.align.M_CROSS_CORRELATION):
                        if method == cellprofiler.modules.align.M_CROSS_CORRELATION and (
                                    (mask1 is not None) or (mask2 is not None)):
                            continue
                        image1 = numpy.dstack([
                                                  numpy.random.randint(0, 10, size=shape[:2])
                                              .astype(float) / 10.0] * 3)
                        image1[numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20, :] = .5
                        si1, si2 = self.slice_helper(offset[0], image1.shape[0])
                        sj1, sj2 = self.slice_helper(offset[1], image1.shape[1])
                        image2 = numpy.zeros(image1.shape)
                        if method == cellprofiler.modules.align.M_MUTUAL_INFORMATION:
                            image2[si2, sj2, :] = 1 - image1[si1, sj1, :]
                        else:
                            image2 = image1[(i + shape[0] - offset[0]) % shape[0],
                                     (j + shape[1] - offset[1]) % shape[1], :]
                            image2 += (numpy.random.uniform(size=shape) - .5) * .1 * numpy.std(image2)
                        if mask1 is not None:
                            image1[~ mask1, :] = numpy.random.uniform(
                                    size=(numpy.sum(~mask1), shape[2]))
                        if mask2 is not None:
                            image2[~ mask2, :] = numpy.random.uniform(
                                    size=(numpy.sum(~mask2), shape[2]))
                        workspace, module = self.make_workspace(
                                (image1, image2), (mask1, mask2))
                        self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
                        module.alignment_method.value = method
                        module.crop_mode.value = cellprofiler.modules.align.C_SAME_SIZE
                        module.run(workspace)
                        output = workspace.image_set.get_image('Aligned0')
                        m = workspace.measurements
                        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
                        off_i0 = -m.get_current_image_measurement('Align_Yshift_Aligned0')
                        off_j0 = -m.get_current_image_measurement('Align_Xshift_Aligned0')
                        off_i1 = -m.get_current_image_measurement('Align_Yshift_Aligned1')
                        off_j1 = -m.get_current_image_measurement('Align_Xshift_Aligned1')

                        self.assertEqual(off_i0 - off_i1, offset[0])
                        self.assertEqual(off_j0 - off_j1, offset[1])

                        si_in, si_out = self.slice_same(off_i0, shape[0])
                        sj_in, sj_out = self.slice_same(off_j0, shape[1])
                        numpy.testing.assert_almost_equal(
                                image1[si_in, sj_in, :], output.pixel_data[si_out, sj_out, :])
                        if mask1 is not None:
                            self.assertTrue(numpy.all(output.mask[si_out, sj_out] == mask1[si_in, sj_in]))

                        temp = output.mask.copy()
                        temp[si_out, sj_out] = False
                        self.assertTrue(numpy.all(~ temp))

                        output = workspace.image_set.get_image("Aligned1")
                        si_in, si_out = self.slice_same(off_i1, shape[0])
                        sj_in, sj_out = self.slice_same(off_j1, shape[1])
                        numpy.testing.assert_almost_equal(
                                image2[si_in, sj_in, :], output.pixel_data[si_out, sj_out, :])
                        if mask2 is not None:
                            self.assertTrue(numpy.all(mask2[si_in, sj_in] == output.mask[si_out, sj_out]))
                        temp = output.mask.copy()
                        temp[si_out, sj_out] = False
                        self.assertTrue(numpy.all(~ temp))

    def test_03_04_align_binary(self):
        numpy.random.seed(0)
        shape = (50, 45)
        i, j = numpy.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((1, 0), (0, 1), (1, 1), (3, 5), (-3, 5), (3, -5), (-3, -5), (0, 5), (3, 0), (0, 0)):
            for mask1 in (None,
                          numpy.random.uniform(size=shape) > .1):
                for mask2 in (None,
                              numpy.random.uniform(size=shape) > .1):
                    for method in (cellprofiler.modules.align.M_MUTUAL_INFORMATION, cellprofiler.modules.align.M_CROSS_CORRELATION):
                        if method == cellprofiler.modules.align.M_CROSS_CORRELATION and (
                                    (mask1 is not None) or (mask2 is not None)):
                            continue
                        image1 = numpy.random.randint(0, 1, size=shape).astype(bool)
                        image1[numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 10] = True
                        si1, si2 = self.slice_helper(offset[0], image1.shape[0])
                        sj1, sj2 = self.slice_helper(offset[1], image1.shape[1])
                        image2 = numpy.zeros(image1.shape, bool)
                        if method == cellprofiler.modules.align.M_MUTUAL_INFORMATION:
                            image2[si2, sj2] = 1 - image1[si1, sj1]
                        else:
                            image2 = image1[(i + shape[0] - offset[0]) % shape[0],
                                            (j + shape[1] - offset[1]) % shape[1]]
                        if mask1 is not None:
                            image1[~ mask1] = numpy.random.uniform(size=numpy.sum(~mask1))
                        if mask2 is not None:
                            image2[~ mask2] = numpy.random.uniform(size=numpy.sum(~mask2))
                        workspace, module = self.make_workspace((image1, image2),
                                                                (mask1, mask2))
                        self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
                        module.alignment_method.value = method
                        module.crop_mode.value = cellprofiler.modules.align.C_SAME_SIZE
                        module.run(workspace)
                        output = workspace.image_set.get_image('Aligned0')
                        m = workspace.measurements
                        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
                        off_i0 = -m.get_current_image_measurement('Align_Yshift_Aligned0')
                        off_j0 = -m.get_current_image_measurement('Align_Xshift_Aligned0')
                        off_i1 = -m.get_current_image_measurement('Align_Yshift_Aligned1')
                        off_j1 = -m.get_current_image_measurement('Align_Xshift_Aligned1')

                        self.assertEqual(off_i0 - off_i1, offset[0])
                        self.assertEqual(off_j0 - off_j1, offset[1])

                        si_in, si_out = self.slice_same(off_i0, shape[0])
                        sj_in, sj_out = self.slice_same(off_j0, shape[1])
                        self.assertEqual(output.pixel_data.dtype.kind, "b")
                        numpy.testing.assert_equal(
                                image1[si_in, sj_in], output.pixel_data[si_out, sj_out])
                        if mask1 is not None:
                            self.assertTrue(numpy.all(output.mask[si_out, sj_out] == mask1[si_in, sj_in]))

                        temp = output.mask.copy()
                        temp[si_out, sj_out] = False
                        self.assertTrue(numpy.all(~ temp))

                        output = workspace.image_set.get_image("Aligned1")
                        si_in, si_out = self.slice_same(off_i1, shape[0])
                        sj_in, sj_out = self.slice_same(off_j1, shape[1])
                        numpy.testing.assert_equal(
                                image2[si_in, sj_in], output.pixel_data[si_out, sj_out])
                        if mask2 is not None:
                            self.assertTrue(numpy.all(mask2[si_in, sj_in] == output.mask[si_out, sj_out]))
                        temp = output.mask.copy()
                        temp[si_out, sj_out] = False
                        self.assertTrue(numpy.all(~ temp))

    def test_04_01_measurement_columns(self):
        workspace, module = self.make_workspace((numpy.zeros((10, 10)),
                                                 numpy.zeros((10, 10)),
                                                 numpy.zeros((10, 10))),
                                                (None, None, None))
        self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
        columns = module.get_measurement_columns(workspace.pipeline)
        self.assertEqual(len(columns), 6)
        for i in range(3):
            for axis in ("X", "Y"):
                feature = cellprofiler.modules.align.MEASUREMENT_FORMAT % (axis, "Aligned%d" % i)
                self.assertTrue(feature in [c[1] for c in columns])
        self.assertTrue(all([c[0] == cellprofiler.measurement.IMAGE for c in columns]))
        self.assertTrue(all([c[2] == cellprofiler.measurement.COLTYPE_INTEGER for c in columns]))

    def test_04_02_categories(self):
        workspace, module = self.make_workspace((numpy.zeros((10, 10)),
                                                 numpy.zeros((10, 10)),
                                                 numpy.zeros((10, 10))),
                                                (None, None, None))
        self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
        c = module.get_categories(workspace.pipeline, cellprofiler.measurement.IMAGE)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0], cellprofiler.modules.align.C_ALIGN)

        c = module.get_categories(workspace.pipeline, 'Aligned0')
        self.assertEqual(len(c), 0)

    def test_04_03_measurements(self):
        workspace, module = self.make_workspace((numpy.zeros((10, 10)),
                                                 numpy.zeros((10, 10)),
                                                 numpy.zeros((10, 10))),
                                                (None, None, None))
        self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
        m = module.get_measurements(workspace.pipeline, cellprofiler.measurement.IMAGE, cellprofiler.modules.align.C_ALIGN)
        self.assertEqual(len(m), 2)
        self.assertTrue("Xshift" in m)
        self.assertTrue("Yshift" in m)

    def test_04_04_measurement_images(self):
        workspace, module = self.make_workspace((numpy.zeros((10, 10)),
                                                 numpy.zeros((10, 10)),
                                                 numpy.zeros((10, 10))),
                                                (None, None, None))
        self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
        for measurement in ("Xshift", "Yshift"):
            image_names = module.get_measurement_images(
                    workspace.pipeline, cellprofiler.measurement.IMAGE, cellprofiler.modules.align.C_ALIGN, measurement)
            self.assertEqual(len(image_names), 3)
            for i in range(3):
                self.assertTrue("Aligned%d" % i in image_names)

    def test_05_01_align_self(self):
        """Align an image from the fly screen against itself.

        This is a regression test for the bug, IMG-284
        """
        image = tests.modules.read_example_image("ExampleFlyImages",
                                   '01_POS002_D.TIF')
        image = image[0:300, 0:300]  # make smaller so as to be faster
        workspace, module = self.make_workspace((image, image), (None, None))
        module.alignment_method.value = cellprofiler.modules.align.M_MUTUAL_INFORMATION
        module.crop_mode.value = cellprofiler.modules.align.C_PAD
        module.run(workspace)
        m = workspace.measurements
        self.assertEqual(m.get_current_image_measurement('Align_Xshift_Aligned1'), 0)
        self.assertEqual(m.get_current_image_measurement('Align_Yshift_Aligned1'), 0)

    def test_06_01_different_sizes_crop(self):
        """Test align with images of different sizes

        regression test of img-1300
        """
        numpy.random.seed(61)
        shape = (61, 43)
        for method in (cellprofiler.modules.align.M_CROSS_CORRELATION, cellprofiler.modules.align.M_MUTUAL_INFORMATION):
            i, j = numpy.mgrid[0:shape[0], 0:shape[1]]
            image1 = numpy.random.randint(0, 10, size=shape).astype(float) / 10.0
            image1[numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20] = .5
            image2 = image1[2:-2, 2:-2]
            for order, i1_name, i2_name in (
                    ((image1, image2), "Aligned0", "Aligned1"),
                    ((image2, image1), "Aligned1", "Aligned0")):
                workspace, module = self.make_workspace(order, (None, None))
                self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
                module.alignment_method.value = method
                module.crop_mode.value = cellprofiler.modules.align.C_CROP
                module.run(workspace)
                i1 = workspace.image_set.get_image(i1_name)
                self.assertTrue(isinstance(i1, cellprofiler.image.Image))
                p1 = i1.pixel_data
                i2 = workspace.image_set.get_image(i2_name)
                p2 = i2.pixel_data
                self.assertEqual(tuple(p1.shape), tuple(p2.shape))
                self.assertTrue(numpy.all(p1 == p2))
                self.assertTrue(i1.has_crop_mask)
                crop_mask = numpy.zeros(shape, bool)
                crop_mask[2:-2, 2:-2] = True
                self.assertTrue(numpy.all(i1.crop_mask == crop_mask))
                self.assertFalse(i2.has_crop_mask)

    def test_06_02_different_sizes_pad(self):
        """Test align with images of different sizes

        regression test of img-1300
        """
        numpy.random.seed(612)
        shape = (61, 43)
        i, j = numpy.mgrid[0:shape[0], 0:shape[1]]
        image1 = numpy.random.randint(0, 10, size=shape).astype(float) / 10.0
        image1[numpy.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20] = .5
        image2 = image1[2:-2, 2:-2]
        workspace, module = self.make_workspace((image1, image2), (None, None))
        self.assertTrue(isinstance(module, cellprofiler.modules.align.Align))
        module.crop_mode.value = cellprofiler.modules.align.C_PAD
        module.run(workspace)
        i1 = workspace.image_set.get_image("Aligned0")
        self.assertTrue(isinstance(i1, cellprofiler.image.Image))
        p1 = i1.pixel_data
        i2 = workspace.image_set.get_image("Aligned1")
        p2 = i2.pixel_data
        self.assertEqual(tuple(p1.shape), tuple(p2.shape))
        self.assertTrue(numpy.all(p1[2:-2, 2:-2] == p2[2:-2, 2:-2]))
        self.assertFalse(i1.has_mask)
        mask = numpy.zeros(shape, bool)
        mask[2:-2, 2:-2] = True
        self.assertTrue(i2.has_mask)
        self.assertTrue(numpy.all(mask == i2.mask))
