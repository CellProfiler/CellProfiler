import StringIO
import base64
import unittest
import zlib

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.resize
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace
import numpy
import numpy.testing
import skimage.exposure
import skimage.transform

cellprofiler.preferences.set_headless()


INPUT_IMAGE_NAME = 'input'
OUTPUT_IMAGE_NAME = 'output'


class TestResize(unittest.TestCase):
    def test_01_01_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUgjJKFXwKs1RMDJWMDS1MjWyAjKMDAwsFUgGDIyevvwMDAwBTAwM'
                'FXPeTr/rd9tA5Li/V1CCc8fG2bFvlKbHMHh3a73dJKLGm3lazcFjctljNbWt'
                'U26F2gUk6h9/KpfYv+H4DN6fW5zm1mWsFejTeW1ht7/4T3+y+m2Ggv3cF568'
                'ZVvPW6HjlpTdqvWGW8a1pWoho9Ue71/vK/c6MTrWyscr2h9TTFF9a7/4+ptP'
                'ZVPu/jp158ZxMykx9s9Jxw/b8R3Kv74yuei3tWTy/6Y7iRrTS25Y7lNUTDT5'
                '6P+jKF9sumvQndy4/znPrDIm8Vmtz+f+m1Qj/uljVWuedetHa8+9Rx6/eHBZ'
                'wnKipctEYwEZ74OR2oXXHlyxa0v9sfLZ76hZ6zf4s3UEvdnOvyYs926cp0zi'
                '/McT7GvClwTN3qxauK3Ir3/9pbJYv6t7PtX0bar5d/2zRU551s9MVXveKONn'
                'M48Fvwz5esdvpmdkyrHET4+4zrmzpsc+4tidGvl7z7l39x/+m2v5h/uj+NLm'
                'ruCt9TckWC16fKSF8y+kO879meMhk/i9w18j/ewq936LwvkW3Fcfr/zWaj/T'
                '/q1dwZYnNfbaP/JV9m3gLD3JW5j8ZHZKx543X+a5T3fT9GPO2RYxl1/0Z6no'
                'Hl9P9sRqxl28sk8ec93b8vPU7z/8615pvdR87m6/oNMm9Hf0/D+14THln/fx'
                '83df4Y6wiYr1i427ajnls8+/rEcXi5+dqLqaG/9I3/xfcb7u25d/fW5YTfm+'
                'Wc9nf35N4glbbZkdTYt/xeV4ySTa/9Nf21b/8U6mVkl2QfZ1g57Ic/9X/V9r'
                '9ZHH9/KTylV/viz5//x3PP/0D/+5L6/t38v5+PKmVY//M5ac/+oEAKjKSvo=')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 4)
        #
        # 4 modules - first resizes by factor, second resizes to a fixed size
        #             interpolation is different for each
        #
        for module in pipeline.modules()[1:]:
            self.assertTrue(isinstance(module, cellprofiler.modules.resize.Resize))
        module = pipeline.modules()[1]
        self.assertEqual(module.x_name, 'DNA')
        self.assertEqual(module.y_name, 'ResizedDNA')
        self.assertEqual(module.size_method, cellprofiler.modules.resize.R_BY_FACTOR)
        self.assertAlmostEqual(module.resizing_factor.value, .25)
        self.assertEqual(module.interpolation, cellprofiler.modules.resize.I_NEAREST_NEIGHBOR)

        module = pipeline.modules()[2]
        self.assertEqual(module.size_method, cellprofiler.modules.resize.R_TO_SIZE)
        self.assertEqual(module.specific_width.value, 150)
        self.assertEqual(module.specific_height.value, 150)
        self.assertEqual(module.interpolation, cellprofiler.modules.resize.I_BILINEAR)

        module = pipeline.modules()[3]
        self.assertEqual(module.interpolation, cellprofiler.modules.resize.I_BICUBIC)

    def test_01_02_load_v1(self):
        data = ('eJztWF9v0zAQd7vuH0ioiAd4tPayFdYo7TZpq9C2siJRWEu1VWPTNMBdndZS'
                'EleJM1bQJB75WHykfQTsLmkSE5a03RBITRWld7nf/e7OzsVOrdzcL7+CG4oK'
                'a+VmXiM6hg0dMY1aRgmabBXuWRgx3IbULMFm14FvHR0W12Bho1TcLBXWYVFV'
                't8B4R6pae8Qv8AkAc/y6wM+0e2vWlVOBU8iHmDFiduxZkAHPXP1Pfh4hi6CW'
                'jo+Q7mDbp/D0VVOjzX5veKtG246O68gIGvOj7hgtbNnvNQ/o3m6QS6wfkq9Y'
                'SsEzO8AXxCbUdPGuf1k75KVM4hV1aCz4dUhJdRB1yQb0wv4N8O0zEXV7HLDP'
                'ujIx2+SCtB2kQ2KgzjAK4U+N8TcT8jcDKvXyALcbg8uCcBzibOJLln99ic4Z'
                'NBA77ybx80DyI+QDbPNBaYtQEueRCvlJgTUXdxyDW5b4l4f8sNWHCGo8G2pB'
                'qkHWxZBapENMXmdhkKhOUflVKDQpg46Nk+eXCfnJAFUpbiTBpUO4NKjT8eZF'
                'QVUTzc+nUr5CrmANOTqDVTE5YYVYWNS0P1EcQdychPMOD7cI/DrvxvBFzes6'
                'Rha2Gaxj0um2qDVO3Cf8qZwk7n+NL8m8GoVvM4ZvDoTHRch7OjVFs/ubdfX6'
                '433xyX2sEIG7Sz65r9R5SSfh+x7D9w6Ex1HIH1d2Gi/FAgVvKy9yn4T0Aev6'
                'Af2yfVrON85ynmaP6o5hbp+q+a2zb4XV4tWN8SHhyIEyF5n3KPF3Y+LflOIX'
                'sojhhPcIN7D1q1xeqGrUZF1XV3R1FdT3NZPEeTw/2rrivvpl1PttsAjpWNTp'
                '3T9/VL/2+SFfGuHeXT0/U9wUN8X9H7jdAG7aN6a4UXHXAZz8fpXXv8L+M7h9'
                'vj0H4fkm5HO+pOlZVHyfsRRj8BHBVnSK2je7eGWf/60GNvRJ9rNLEs/Sn3is'
                'wSZXudnrRtdrMcJ/MO80/2Xnb6+zXF+/7tc74/DNpH7nexiDy7iVErgfYLRx'
                'XbnF3sttXPtf+3UFIg==')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertEqual(module.x_name, 'DNA')
        self.assertEqual(module.y_name, 'ResizedDNA')
        self.assertEqual(module.size_method, cellprofiler.modules.resize.R_BY_FACTOR)
        self.assertAlmostEqual(module.resizing_factor.value, .25)
        self.assertEqual(module.interpolation, cellprofiler.modules.resize.I_NEAREST_NEIGHBOR)

    def test_01_03_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10104

Resize:[module_num:1|svn_version:\'10104\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Select the input image:DNA
    Name the output image:ResizedDNA
    Select resizing method:Resize by specifying desired final dimensions
    Resizing factor:0.25
    Width of the final image, in pixels:141
    Height of the final image, in pixels:169
    Interpolation method:Bilinear
    Additional image count:1
    Select the additional image?:Actin
    Name the output image:ResizedActin

Resize:[module_num:2|svn_version:\'10104\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Select the input image:DNA
    Name the output image:ResizedDNA
    Select resizing method:Resize by specifying desired final dimensions
    Resizing factor:0.25
    Width of the final image, in pixels:100
    Height of the final image, in pixels:100
    Interpolation method:Bicubic
    Additional image count:0
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.resize.Resize))
        self.assertEqual(module.x_name, "DNA")
        self.assertEqual(module.y_name, "ResizedDNA")
        self.assertEqual(module.size_method, cellprofiler.modules.resize.R_TO_SIZE)
        self.assertAlmostEqual(module.resizing_factor.value, 0.25)
        self.assertEqual(module.specific_width, 141)
        self.assertEqual(module.specific_height, 169)
        self.assertEqual(module.interpolation, cellprofiler.modules.resize.I_BILINEAR)
        self.assertEqual(module.additional_image_count.value, 1)
        additional_image = module.additional_images[0]
        self.assertEqual(additional_image.input_image_name, "Actin")
        self.assertEqual(additional_image.output_image_name, "ResizedActin")

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.resize.Resize))
        self.assertEqual(module.interpolation, cellprofiler.modules.resize.I_BICUBIC)

    def make_workspace(self, image, size_method, interpolation,
                       mask=None, cropping=None, dimensions=2):
        module = cellprofiler.modules.resize.Resize()
        module.x_name.value = INPUT_IMAGE_NAME
        module.y_name.value = OUTPUT_IMAGE_NAME
        module.size_method.value = size_method
        module.interpolation.value = interpolation
        module.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image = cellprofiler.image.Image(image, mask, cropping, dimensions=dimensions)
        image_set.add(INPUT_IMAGE_NAME, image)
        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set,
                                                     cellprofiler.object.ObjectSet(),
                                                     cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        return workspace, module

    def test_02_01_rescale_triple_color(self):
        i, j = numpy.mgrid[0:10, 0:10]
        image = numpy.zeros((10, 10, 3))
        image[:, :, 0] = i
        image[:, :, 1] = j
        image = skimage.exposure.rescale_intensity(1.0 * image)
        i, j = (numpy.mgrid[0:30, 0:30].astype(float) * 1.0 / 3.0).astype(int)
        expected = numpy.zeros((30, 30, 3))
        expected[:, :, 0] = i
        expected[:, :, 1] = j
        expected = skimage.exposure.rescale_intensity(1.0 * expected)
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_BY_FACTOR,
            cellprofiler.modules.resize.I_NEAREST_NEIGHBOR
        )
        module.resizing_factor.value = 3.0
        module.run(workspace)
        result_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        result = result_image.pixel_data
        numpy.testing.assert_array_almost_equal(result, expected)
        assert result_image.parent_image is workspace.image_set.get_image(INPUT_IMAGE_NAME)

    def test_02_02_rescale_triple_bw(self):
        i, j = numpy.mgrid[0:10, 0:10].astype(float)
        image = skimage.exposure.rescale_intensity(1.0 * i)
        i, j = (numpy.mgrid[0:30, 0:30].astype(float) * 1.0 / 3.0).astype(int)
        expected = skimage.exposure.rescale_intensity(1.0 * i)
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_BY_FACTOR,
            cellprofiler.modules.resize.I_NEAREST_NEIGHBOR
        )
        module.resizing_factor.value = 3.0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        numpy.testing.assert_array_almost_equal(result, expected)

    def test_02_03_third(self):
        i, j = numpy.mgrid[0:30, 0:30]
        image = skimage.exposure.rescale_intensity(1.0 * i)
        expected = skimage.transform.resize(
            image,
            (10, 10),
            order=0,
            mode="symmetric"
        )
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_BY_FACTOR,
            cellprofiler.modules.resize.I_NEAREST_NEIGHBOR
        )
        module.resizing_factor.value = 1.0 / 3.0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        numpy.testing.assert_array_almost_equal(result, expected)

    def test_03_01_bilinear(self):
        i, j = numpy.mgrid[0:10, 0:10]
        image = skimage.exposure.rescale_intensity(1.0 * i)
        expected = skimage.transform.resize(
            image,
            (30, 30),
            order=1,
            mode="symmetric"
        )
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_BY_FACTOR,
            cellprofiler.modules.resize.I_BILINEAR
        )
        module.resizing_factor.value = 3.0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        numpy.testing.assert_array_almost_equal(result, expected)

    def test_03_02_bicubic(self):
        i, j = numpy.mgrid[0:10, 0:10]
        image = skimage.exposure.rescale_intensity(1.0 * i)
        expected = skimage.transform.resize(
            image,
            (30, 30),
            order=3,
            mode="symmetric"
        )
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_BY_FACTOR,
            cellprofiler.modules.resize.I_BICUBIC
        )
        module.resizing_factor.value = 3.0
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        numpy.testing.assert_array_almost_equal(result, expected)

    def test_04_01_reshape_double(self):
        '''Make an image twice as large by changing the shape'''
        i, j = numpy.mgrid[0:10, 0:10].astype(float)
        image = skimage.exposure.rescale_intensity(i + j * 10.0)
        expected = skimage.transform.resize(
            image,
            (19, 19),
            order=1,
            mode="symmetric"
        )
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_TO_SIZE,
            cellprofiler.modules.resize.I_BILINEAR
        )
        module.specific_width.value = 19
        module.specific_height.value = 19
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        numpy.testing.assert_array_almost_equal(result, expected)

    def test_04_02_reshape_half(self):
        '''Make an image half as large by changing the shape'''
        i, j = numpy.mgrid[0:19, 0:19].astype(float) / 2.0
        image = skimage.exposure.rescale_intensity(i + j * 10)
        expected = skimage.transform.resize(
            image,
            (10, 10),
            order=1,
            mode="symmetric"
        )
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_TO_SIZE,
            cellprofiler.modules.resize.I_BILINEAR
        )
        module.specific_width.value = 10
        module.specific_height.value = 10
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        numpy.testing.assert_array_almost_equal(result, expected)

    def test_04_03_reshape_half_and_double(self):
        '''Make an image twice as large in one dimension and half in other'''
        i, j = numpy.mgrid[0:10, 0:19].astype(float)
        image = skimage.exposure.rescale_intensity(i + j * 5.0)
        expected = skimage.transform.resize(
            image,
            (19, 10),
            order=1,
            mode="symmetric"
        )
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_TO_SIZE,
            cellprofiler.modules.resize.I_BILINEAR
        )
        module.specific_width.value = 10
        module.specific_height.value = 19
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        numpy.testing.assert_array_almost_equal(result, expected)

    def test_04_04_reshape_using_another_images_dimensions(self):
        ''''Resize to another image's dimensions'''
        i, j = numpy.mgrid[0:10, 0:19].astype(float)
        image = skimage.exposure.rescale_intensity(1.0 * i + j)
        expected = skimage.transform.resize(
            image,
            (19, 10),
            order=1,
            mode="symmetric"
        )
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_TO_SIZE,
            cellprofiler.modules.resize.I_BILINEAR
        )
        module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE
        module.specific_image.value = 'AnotherImage'
        workspace.image_set.add(module.specific_image.value, cellprofiler.image.Image(expected))
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertTrue(expected.shape == result.shape)

    def test_05_01_resize_with_cropping(self):
        # This is a regression test for issue # 967
        r = numpy.random.RandomState()
        r.seed(501)
        i, j = numpy.mgrid[0:10, 0:20]
        image = i + j
        mask = r.uniform(size=image.shape) > .5
        imask = mask.astype(int)
        cropping = numpy.zeros((30, 40), bool)
        cropping[10:20, 10:30] = True
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_BY_FACTOR,
            cellprofiler.modules.resize.I_BILINEAR,
            mask,
            cropping
        )
        module.resizing_factor.value = .5
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertEqual(tuple(result.mask.shape), (5, 10))
        self.assertEqual(tuple(result.crop_mask.shape), (5, 10))
        x = result.crop_image_similarly(numpy.zeros(result.crop_mask.shape))
        self.assertEqual(tuple(x.shape), (5, 10))

    def test_05_02_resize_with_cropping_bigger(self):
        # This is a regression test for issue # 967
        r = numpy.random.RandomState()
        r.seed(501)
        i, j = numpy.mgrid[0:10, 0:20]
        image = i + j
        mask = r.uniform(size=image.shape) > .5
        imask = mask.astype(int)
        cropping = numpy.zeros((30, 40), bool)
        cropping[10:20, 10:30] = True
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_BY_FACTOR,
            cellprofiler.modules.resize.I_BILINEAR,
            mask,
            cropping
        )
        module.resizing_factor.value = 2
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertEqual(tuple(result.mask.shape), (20, 40))
        self.assertEqual(tuple(result.crop_mask.shape), (20, 40))
        x = result.crop_image_similarly(numpy.zeros(result.crop_mask.shape))
        self.assertEqual(tuple(x.shape), (20, 40))

    def test_05_03_resize_color(self):
        # Regression test of issue #1416
        image = numpy.zeros((20, 22, 3))
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_BY_FACTOR,
            cellprofiler.modules.resize.I_BILINEAR
        )
        module.resizing_factor.value = .5
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertEqual(tuple(result.pixel_data.shape), (10, 11, 3))

    def test_05_04_resize_color_bw(self):
        # Regression test of issue #1416
        image = numpy.zeros((20, 22, 3))
        tgt_image = numpy.zeros((5, 11))
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_TO_SIZE,
            cellprofiler.modules.resize.I_BILINEAR
        )
        module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE
        module.specific_image.value = 'AnotherImage'
        workspace.image_set.add(
            module.specific_image.value,
            cellprofiler.image.Image(tgt_image)
        )
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertEqual(tuple(result.pixel_data.shape), (5, 11, 3))

    def test_05_05_resize_color_color(self):
        # Regression test of issue #1416
        image = numpy.zeros((20, 22, 3))
        tgt_image = numpy.zeros((10, 11, 3))
        workspace, module = self.make_workspace(
            image,
            cellprofiler.modules.resize.R_TO_SIZE,
            cellprofiler.modules.resize.I_BILINEAR
        )
        module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE
        module.specific_image.value = 'AnotherImage'
        workspace.image_set.add(
            module.specific_image.value,
            cellprofiler.image.Image(tgt_image)
        )
        module.run(workspace)
        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertEqual(tuple(result.pixel_data.shape), (10, 11, 3))

    def test_06_01_resize_volume_factor_grayscale(self):
        numpy.random.seed(73)

        data = numpy.random.rand(10, 10, 10)

        mask = data > 0.5

        crop_mask = numpy.zeros_like(data, dtype=numpy.bool)

        crop_mask[1:-1, 1:-1, 1:-1] = True

        workspace, module = self.make_workspace(
            data,
            cellprofiler.modules.resize.R_BY_FACTOR,
            cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
            mask=mask,
            cropping=crop_mask,
            dimensions=3
        )

        module.resizing_factor.value = 0.5

        module.run(workspace)

        actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

        expected_data = skimage.transform.resize(
            data,
            (10, 5, 5),
            order=0,
            mode="symmetric"
        )

        expected_mask = skimage.img_as_bool(
            skimage.transform.resize(
                mask,
                (10, 5, 5),
                order=0,
                mode="constant"
            )
        )

        expected_crop_mask = skimage.img_as_bool(
            skimage.transform.resize(
                crop_mask,
                (10, 5, 5),
                order=0,
                mode="constant"
            )
        )

        assert actual.volumetric

        numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

        numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

        numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)

    def test_06_02_resize_volume_factor_color(self):
        numpy.random.seed(73)

        data = numpy.random.rand(10, 10, 10, 3)

        mask = data[:, :, :, 0] > 0.5

        crop_mask = numpy.zeros_like(mask, dtype=numpy.bool)

        crop_mask[1:-1, 1:-1, 1:-1] = True

        workspace, module = self.make_workspace(
            data,
            cellprofiler.modules.resize.R_BY_FACTOR,
            cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
            mask=mask,
            cropping=crop_mask,
            dimensions=3
        )

        module.resizing_factor.value = 2.5

        module.run(workspace)

        actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

        expected_data = numpy.zeros((10, 25, 25, 3), dtype=data.dtype)

        for idx in range(3):
            expected_data[:, :, :, idx] = skimage.transform.resize(
                data[:, :, :, idx],
                (10, 25, 25),
                order=0,
                mode="symmetric"
            )

        expected_mask = skimage.img_as_bool(
            skimage.transform.resize(
                mask,
                (10, 25, 25),
                order=0,
                mode="constant"
            )
        )

        expected_crop_mask = skimage.img_as_bool(
            skimage.transform.resize(
                crop_mask,
                (10, 25, 25),
                order=0,
                mode="constant"
            )
        )

        assert actual.volumetric

        numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

        numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

        numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)

    def test_06_03_resize_volume_manual_grayscale(self):
        numpy.random.seed(73)

        data = numpy.random.rand(10, 10, 10)

        mask = data > 0.5

        crop_mask = numpy.zeros_like(data, dtype=numpy.bool)

        crop_mask[1:-1, 1:-1, 1:-1] = True

        workspace, module = self.make_workspace(
            data,
            cellprofiler.modules.resize.R_TO_SIZE,
            cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
            mask=mask,
            cropping=crop_mask,
            dimensions=3
        )

        module.use_manual_or_image.value = cellprofiler.modules.resize.C_MANUAL

        module.specific_width.value = 25

        module.specific_height.value = 30

        module.run(workspace)

        actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

        expected_data = skimage.transform.resize(
            data,
            (10, 30, 25),
            order=0,
            mode="symmetric"
        )

        expected_mask = skimage.img_as_bool(
            skimage.transform.resize(
                mask,
                (10, 30, 25),
                order=0,
                mode="constant"
            )
        )

        expected_crop_mask = skimage.img_as_bool(
            skimage.transform.resize(
                crop_mask,
                (10, 30, 25),
                order=0,
                mode="constant"
            )
        )

        assert actual.volumetric

        numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

        numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

        numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)

    def test_06_04_resize_volume_manual_color(self):
        numpy.random.seed(73)

        data = numpy.random.rand(10, 10, 10, 3)

        mask = data[:, :, :, 0] > 0.5

        crop_mask = numpy.zeros_like(mask, dtype=numpy.bool)

        crop_mask[1:-1, 1:-1, 1:-1] = True

        workspace, module = self.make_workspace(
            data,
            cellprofiler.modules.resize.R_TO_SIZE,
            cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
            mask=mask,
            cropping=crop_mask,
            dimensions=3
        )

        module.use_manual_or_image.value = cellprofiler.modules.resize.C_MANUAL

        module.specific_width.value = 5

        module.specific_height.value = 8

        module.run(workspace)

        actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

        expected_data = numpy.zeros((10, 8, 5, 3), dtype=data.dtype)

        for idx in range(3):
            expected_data[:, :, :, idx] = skimage.transform.resize(
                data[:, :, :, idx],
                (10, 8, 5),
                order=0,
                mode="symmetric"
            )

        expected_mask = skimage.img_as_bool(
            skimage.transform.resize(
                mask,
                (10, 8, 5),
                order=0,
                mode="constant"
            )
        )

        expected_crop_mask = skimage.img_as_bool(
            skimage.transform.resize(
                crop_mask,
                (10, 8, 5),
                order=0,
                mode="constant"
            )
        )

        assert actual.volumetric

        numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

        numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

        numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)

    def test_06_05_resize_volume_grayscale_other_volume_grayscale(self):
        numpy.random.seed(73)

        data = numpy.random.rand(10, 10, 10)

        mask = data > 0.5

        crop_mask = numpy.zeros_like(data, dtype=numpy.bool)

        crop_mask[1:-1, 1:-1, 1:-1] = True

        workspace, module = self.make_workspace(
            data,
            cellprofiler.modules.resize.R_TO_SIZE,
            cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
            mask=mask,
            cropping=crop_mask,
            dimensions=3
        )

        module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE

        module.specific_image.value = "Other Image"

        expected_data = skimage.transform.resize(
            data,
            (10, 30, 25),
            order=0,
            mode="symmetric"
        )

        expected_mask = skimage.img_as_bool(
            skimage.transform.resize(
                mask,
                (10, 30, 25),
                order=0,
                mode="constant"
            )
        )

        expected_crop_mask = skimage.img_as_bool(
            skimage.transform.resize(
                crop_mask,
                (10, 30, 25),
                order=0,
                mode="constant"
            )
        )

        other_image = cellprofiler.image.Image(
            expected_data,
            mask=expected_mask,
            crop_mask=expected_crop_mask,
            dimensions=3
        )

        workspace.image_set.add(module.specific_image.value, other_image)

        module.run(workspace)

        actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

        assert actual.volumetric

        numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

        numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

        numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)

    def test_06_06_resize_volume_grayscale_other_volume_color(self):
        numpy.random.seed(73)

        data = numpy.random.rand(10, 10, 10)

        mask = data > 0.5

        crop_mask = numpy.zeros_like(data, dtype=numpy.bool)

        crop_mask[1:-1, 1:-1, 1:-1] = True

        workspace, module = self.make_workspace(
            data,
            cellprofiler.modules.resize.R_TO_SIZE,
            cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
            mask=mask,
            cropping=crop_mask,
            dimensions=3
        )

        module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE

        module.specific_image.value = "Other Image"

        expected_data = skimage.transform.resize(
            data,
            (10, 30, 25),
            order=0,
            mode="symmetric"
        )

        expected_mask = skimage.img_as_bool(
            skimage.transform.resize(
                mask,
                (10, 30, 25),
                order=0,
                mode="constant"
            )
        )

        expected_crop_mask = skimage.img_as_bool(
            skimage.transform.resize(
                crop_mask,
                (10, 30, 25),
                order=0,
                mode="constant"
            )
        )

        other_image = cellprofiler.image.Image(
            numpy.random.rand(10, 30, 25, 3),
            mask=expected_mask,
            crop_mask=expected_crop_mask,
            dimensions=3
        )

        workspace.image_set.add(module.specific_image.value, other_image)

        module.run(workspace)

        actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

        assert actual.volumetric

        numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

        numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

        numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)

    def test_06_07_resize_volume_color_other_volume_grayscale(self):
        numpy.random.seed(73)

        data = numpy.random.rand(10, 10, 10, 3)

        mask = data[:, :, :, 0] > 0.5

        crop_mask = numpy.zeros_like(mask, dtype=numpy.bool)

        crop_mask[1:-1, 1:-1, 1:-1] = True

        workspace, module = self.make_workspace(
            data,
            cellprofiler.modules.resize.R_TO_SIZE,
            cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
            mask=mask,
            cropping=crop_mask,
            dimensions=3
        )

        module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE

        module.specific_image.value = "Other Image"

        expected_data = numpy.zeros((10, 30, 25, 3), dtype=data.dtype)

        for idx in range(3):
            expected_data[:, :, :, idx] = skimage.transform.resize(
                data[:, :, :, idx],
                (10, 30, 25),
                order=0,
                mode="symmetric"
            )

        expected_mask = skimage.img_as_bool(
            skimage.transform.resize(
                mask,
                (10, 30, 25),
                order=0,
                mode="constant"
            )
        )

        expected_crop_mask = skimage.img_as_bool(
            skimage.transform.resize(
                crop_mask,
                (10, 30, 25),
                order=0,
                mode="constant"
            )
        )

        other_image = cellprofiler.image.Image(
            expected_data,
            mask=expected_mask,
            crop_mask=expected_crop_mask,
            dimensions=3
        )

        workspace.image_set.add(module.specific_image.value, other_image)

        module.run(workspace)

        actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

        assert actual.volumetric

        numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

        numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

        numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)

    def test_06_08_resize_volume_color_other_volume_color(self):
        numpy.random.seed(73)

        data = numpy.random.rand(10, 10, 10, 3)

        mask = data[:, :, :, 0] > 0.5

        crop_mask = numpy.zeros_like(mask, dtype=numpy.bool)

        crop_mask[1:-1, 1:-1, 1:-1] = True

        workspace, module = self.make_workspace(
            data,
            cellprofiler.modules.resize.R_TO_SIZE,
            cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
            mask=mask,
            cropping=crop_mask,
            dimensions=3
        )

        module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE

        module.specific_image.value = "Other Image"

        expected_data = numpy.zeros((10, 30, 25, 3), dtype=data.dtype)

        for idx in range(3):
            expected_data[:, :, :, idx] = skimage.transform.resize(
                data[:, :, :, idx],
                (10, 30, 25),
                order=0,
                mode="symmetric"
            )

        expected_mask = skimage.img_as_bool(
            skimage.transform.resize(
                mask,
                (10, 30, 25),
                order=0,
                mode="constant"
            )
        )

        expected_crop_mask = skimage.img_as_bool(
            skimage.transform.resize(
                crop_mask,
                (10, 30, 25),
                order=0,
                mode="constant"
            )
        )

        other_image = cellprofiler.image.Image(
            numpy.random.rand(10, 30, 25, 3),
            mask=expected_mask,
            crop_mask=expected_crop_mask,
            dimensions=3
        )

        workspace.image_set.add(module.specific_image.value, other_image)

        module.run(workspace)

        actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

        assert actual.volumetric

        numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

        numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

        numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)

    # https://github.com/CellProfiler/CellProfiler/issues/3080
    def test_resize_factor_rounding(self):
        data = numpy.zeros((99, 99))

        workspace, module = self.make_workspace(
            data,
            cellprofiler.modules.resize.R_BY_FACTOR,
            cellprofiler.modules.resize.I_NEAREST_NEIGHBOR
        )

        module.resizing_factor.value = 0.25

        module.run(workspace)

        assert workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data.shape == (25, 25)

    # https://github.com/CellProfiler/CellProfiler/issues/3531
    def test_resize_float(self):
        data = numpy.ones((10,10), dtype=numpy.float32)*2
        expected = numpy.ones((5,5), dtype=numpy.float32)*2
        workspace, module = self.make_workspace(
            data,
            cellprofiler.modules.resize.R_BY_FACTOR,
            cellprofiler.modules.resize.I_NEAREST_NEIGHBOR
        )

        module.resizing_factor.value = 0.5

        module.run(workspace)

        result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        numpy.testing.assert_allclose(result, expected)
