'''test_align.py - test the Align module
'''

import base64
import unittest
import zlib
from StringIO import StringIO

import numpy as np
from cellprofiler.preferences import set_headless

set_headless()

import tests.modules
import cellprofiler.pipeline as cpp
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw

import cellprofiler.modules.align as A


class TestAlign(unittest.TestCase):
    def make_workspace(self, images, masks):
        pipeline = cpp.Pipeline()
        object_set = cpo.ObjectSet()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        module = A.Align()
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set,
                                  object_set,
                                  cpmeas.Measurements(),
                                  image_set_list)
        for index, (pixels, mask) in enumerate(zip(images, masks)):
            if mask is None:
                image = cpi.Image(pixels)
            else:
                image = cpi.Image(pixels, mask=mask)
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

    def test_01_01_load_matlab(self):
        '''Load a Matlab pipeline'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUvDNz1PwTy5RMDBVMLCwMjWyMjRQMDIwsFQgGTAwevryMzAwlDIx'
                'MFTMeRvhmH/ZQKTs9pbXyzJLdh9gXCazM8AhxoKPV0XNcea1TCH39YXG3n4y'
                'queufDT+Ib+76NnyD8Zlc+Za5T5K3ZpuFNUZtWnR8z3f9763vf2xu5yB4V0w'
                'g9Wfm0/3ci5bq/vUbSF3ZYfE4YXztzQyzzv58f6e+QclDhc3nVPwDGRK05W3'
                '5zr+5YvWrL2/vPZOWcDI03VqD++lBzJHT24vzNOIl3/MqeT8h7mx6PRVvwuq'
                'fga8S/ceCXdPr9MM/vPuw7+06nntbZPaxd5XtxYurGd49svP9V4V6995b5Zn'
                'HIqrnXVAsJj5hJv4zHhev9Utl5keZs/4Wb9+4+umTf/zntl3cKScOLI+eFO5'
                'w/yHGztjF6dcZ6g9XGMb+IPrJ/PBuo49c7YF3slldT/hzvhl4pOPkcn7V994'
                'L/Zp2sMviy9Mf//6weLTp6cl6T/pv8KrWXyZZebnC2vZZftzj8adKKr7VJDV'
                'm977uf6RTHbBPG0LJ7kLFg0t8/nnVUtcVinkz5RR3rfp5/X60Mcxj3WOVb6w'
                '/Jzn+niDpfwNVv3IJbV3a3LO2ctU/RCq1lTRE7v/a09VFU/drsr8aQWzat7r'
                'tx+vLfW7a1NiMZurb8JfiThVZY9PVZWSPwNfX3tfc1Wk8sa5z/fX8Uf/q7y7'
                '9tEXM7H/5iKvb2n9ZubtX/XzhNlC+csmm46v/WU/8/GumtOLV/jNOd5VfvVR'
                'zPPsnT+jwvYs+xxQvmd1V3j4vivpdz/OerTup13Pe94d/2NCzoffrtgXYLG3'
                'xuNM4ap3M1nOCbTbzbz3IeL625u1HDui40tMPprv3LL96uZV1vejfmVbz7Uu'
                'fPVGZuXn10e//jv/d+V9+4SPNuLHW/+dZv1V+Xv6L7Xz0euv10cCALr4bf0=')
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.load(fd)
        #
        # The pipeline has three modules and Align is the third. It has
        # the following settings:
        # input image name = Template
        # output image name = AlignedTemplate
        # second input name = Image
        # second output name = AlignedImage
        # use normalized correlation
        #
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, A.Align))
        self.assertEqual(module.first_input_image.value, "Template")
        self.assertEqual(module.first_output_image.value, "AlignedTemplate")
        self.assertEqual(module.second_input_image.value, "Image")
        self.assertEqual(module.second_output_image.value, "AlignedImage")
        self.assertEqual(module.alignment_method.value, A.M_CROSS_CORRELATION)
        self.assertEqual(module.crop_mode, A.C_SAME_SIZE)

    def test_01_02_load_v1(self):
        '''Load a version 1 pipeline'''
        data = ('eJztWl9P2lAUvyAuui2b7mV7vCZ7kE0IFJYoWRQmS8YmjgjRGeO2K1zgJre'
                '9pLQqW0z2uI+xj7KPs8d9hN2LrS3XaoGKxtmSpj2n53fPOb+ec/uHlgu1jc'
                'Ib+CqZguVCLdEkFMMKRUaT6WoOasYSXNcxMnADMi0Hd/j2valBJQvT2Vw6l'
                'eM7Siq1AsZbIqXyI75pPwfgHt/O8DVqHZq25IhrFXIVGwbRWt1pEAPPLP1v'
                'vm4jnaADircRNXHXcWHrS1qT1Xqds0Nl1jAp3kSq25gvm6Z6gPXux6YNtA5'
                'XyDGmVfINSynYZlv4kHQJ0yy8Nb6sPfPLDMmv4OHXvMNDROJB8LLg0gv7d8'
                'Cxj3nwNu+yn7NkojXIIWmYiEKiotZZFGK8ZZ/xZqTxhLzeRpqGaVrg8z74O'
                'Qkv1ho+NhJvj1HdgCoy6u1x4yhQ0tJwox9HygcfGcBHQCZ4/sow+d+X8EIu'
                'MqgxA5pdHOw8WPkrw+QfHcBHwSYLnH9mmHp8KuGFXMRNZFIDlkQxwiLRcd1'
                'gei8oDxmZh3sS3l5s/Cxw+M/7+J2V/Aq5SlRCkU57rnH8zsPUwDhTYJd3Y8'
                'DzkL1uvxbfQ/mV6+4oPRzfXn1TxR2k8+uSTfhVzF9+9ftEwgu5bBpiLhWXF'
                '51PYHyud+IZdR5KD4m7qH/9cLEBXIzjNBykT374+Psg8SXkz4trldfiBgOv'
                'Jl/GvwhpB1O6xY5W9wqJyn7c1qwzaqra6l4qsbL/Pb2knJwaVwlH9pVxT55'
                'Hib/tE/+yFL+QRQy7GOlWYNmTeEKoykwz2pZOsXRF1HM0QeIcua+UYLwEvQ'
                '5OKk+5XxQP3FX6C/vljvRL5sb6JdB9yrj3uWG/hP0SqF+yN9Yv5+4zJ9kvW'
                'Q9c2C9hv3x6PNp7mkk9h3o9F/Vf6rR0ZnYm79/rfY7jHxKtgTth/mH+VzF/'
                'XgfOL9+Q7/8r7hB3N3F5ENZriLs9uDwI6zXE3R5cHoT1GuLGx/1x4eTna/l'
                '9lLD/Ci6vtxdgsN6EXMeUdnQmvnfRk2r/o4xukjLUOP0qIrnBd0uuDySEn4'
                'qPHyj5gRf5QeJP02T/r1OZp1mPcd35Rvlv4eHl/Mq8Onz/XRvHXzRy3t8DH'
                '1zMYkjgfoLRzufiJfZ2buPa/wMTtu04')
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        #
        # There are two modules and Align is the second. It has the
        # following settings:
        #
        # First image:   input = Channel1, output = Aligned1
        # Second image:  input = Channel2, output = Aligned2
        # Third image:   input = Channel3, output = Aligned3, similarly aligned
        # Fourth image:  input = Channel4, output = Aligned4, separately
        # Mutual information
        # Don't crop
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertEqual(len(module.additional_images), 2)
        for i, input, output in ((1, module.first_input_image, module.first_output_image),
                                 (2, module.second_input_image, module.second_output_image),
                                 (3, module.additional_images[0].input_image_name,
                                  module.additional_images[0].output_image_name),
                                 (4, module.additional_images[1].input_image_name,
                                  module.additional_images[1].output_image_name)):
            self.assertEqual(input.value, 'Channel%d' % i)
            self.assertEqual(output.value, 'Aligned%d' % i)
        self.assertEqual(module.additional_images[0].align_choice.value, A.A_SIMILARLY)
        self.assertEqual(module.additional_images[1].align_choice.value, A.A_SEPARATELY)
        self.assertEqual(module.alignment_method.value, A.M_MUTUAL_INFORMATION)
        self.assertEqual(module.crop_mode, A.C_SAME_SIZE)

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
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, A.Align))
        self.assertEqual(module.alignment_method, A.M_MUTUAL_INFORMATION)
        self.assertEqual(module.crop_mode, A.C_CROP)
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
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 3)
        for module, crop_method in zip(
                pipeline.modules(),
                (A.C_SAME_SIZE, A.C_CROP, A.C_PAD)):
            self.assertTrue(isinstance(module, A.Align))
            self.assertEqual(module.alignment_method, A.M_MUTUAL_INFORMATION)
            self.assertEqual(module.crop_mode, crop_method)
            self.assertEqual(module.first_input_image, "Image1")
            self.assertEqual(module.second_input_image, "Image2")
            self.assertTrue(module.first_output_image, "AlignedImage1")
            self.assertTrue(module.second_output_image, "AlignedImage2")

    def test_02_01_crop(self):
        '''Align two images and crop the result'''
        np.random.seed(0)
        shape = (50, 45)
        i, j = np.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((3, 5), (-3, 5), (3, -5), (-3, -5), (0, 5), (3, 0), (0, 0)):
            #
            # Do something to give the image some information over
            # the distance, 5,5
            #
            for mask1 in (None,
                          np.random.uniform(size=shape) > .1):
                for mask2 in (None,
                              np.random.uniform(size=shape) > .1):
                    for method in (A.M_MUTUAL_INFORMATION, A.M_CROSS_CORRELATION):
                        if method == A.M_CROSS_CORRELATION and (
                                    (mask1 is not None) or (mask2 is not None)):
                            continue

                        image1 = np.random.randint(0, 10, size=shape).astype(float) / 10.0
                        image1[np.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20] = .5
                        si1, si2 = self.slice_helper(offset[0], image1.shape[0])
                        sj1, sj2 = self.slice_helper(offset[1], image1.shape[1])
                        image2 = np.zeros(image1.shape)
                        if method == A.M_MUTUAL_INFORMATION:
                            image2[si2, sj2] = 1 - image1[si1, sj1]
                        else:
                            image2 = image1[(i + shape[0] - offset[0]) % shape[0],
                                            (j + shape[1] - offset[1]) % shape[1]]
                            image2 += (np.random.uniform(size=shape) - .5) * .1 * np.std(image2)
                        if mask1 is not None:
                            image1[~ mask1] = np.random.uniform(size=np.sum(~mask1))
                        if mask2 is not None:
                            image2[~ mask2] = np.random.uniform(size=np.sum(~mask2))
                        workspace, module = self.make_workspace((image1, image2),
                                                                (mask1, mask2))
                        self.assertTrue(isinstance(module, A.Align))
                        module.alignment_method.value = method
                        module.crop_mode.value = A.C_CROP
                        module.run(workspace)
                        output = workspace.image_set.get_image('Aligned0')
                        m = workspace.measurements
                        self.assertTrue(isinstance(m, cpmeas.Measurements))
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
                        np.testing.assert_almost_equal(
                                image1[i_slice, j_slice], output.pixel_data)
                        if mask1 is not None:
                            self.assertTrue(np.all(output.mask == mask1[i_slice, j_slice]))

                        if offset[0] == 0 and offset[1] == 0:
                            self.assertFalse(output.has_crop_mask)
                        else:
                            temp = output.crop_mask.copy()
                            self.assertEqual(tuple(temp.shape), shape)
                            self.assertTrue(np.all(temp[i_slice, j_slice]))
                            temp[i_slice, j_slice] = False
                            self.assertTrue(np.all(~ temp))

                        output = workspace.image_set.get_image("Aligned1")
                        i_slice = self.single_slice_helper(-off_i1, out_shape[0])
                        j_slice = self.single_slice_helper(-off_j1, out_shape[1])

                        np.testing.assert_almost_equal(
                                image2[i_slice, j_slice], output.pixel_data)
                        if mask2 is not None:
                            self.assertTrue(np.all(output.mask == mask2[i_slice, j_slice]))
                        if offset[0] == 0 and offset[1] == 0:
                            self.assertFalse(output.has_crop_mask)
                        else:
                            temp = output.crop_mask.copy()
                            self.assertEqual(tuple(temp.shape), shape)
                            self.assertTrue(np.all(temp[i_slice, j_slice]))
                            temp[i_slice, j_slice] = False
                            self.assertTrue(np.all(~ temp))

    def single_slice_helper(self, offset, size):
        '''Return a single slice starting at the offset (or zero)'''
        if offset < 0:
            offset = 0
        return slice(offset, offset + size)

    def slice_helper(self, offset, size):
        '''Return slices for the first and second images for copying

        offset - amount to offset the second image relative to the first

        returns two slices, the first to apply to the first image, second
        to apply to the second image.
        '''
        if offset < 0:
            return slice(-offset, size), slice(0, size + offset)
        else:
            return slice(0, size - offset), slice(offset, size)

    def test_02_02_pad(self):
        '''Align two images with padded output'''
        np.random.seed(0)
        shape = (50, 45)
        i, j = np.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((1, 0), (0, 1), (1, 1), (3, 5), (-3, 5), (3, -5), (-3, -5), (0, 5), (3, 0), (0, 0)):
            for mask1 in (None,
                          np.random.uniform(size=shape) > .1):
                for mask2 in (None,
                              np.random.uniform(size=shape) > .1):
                    for method in (A.M_MUTUAL_INFORMATION, A.M_CROSS_CORRELATION):
                        if method == A.M_CROSS_CORRELATION and (
                                    (mask1 is not None) or (mask2 is not None)):
                            continue
                        image1 = np.random.randint(0, 10, size=shape).astype(float) / 10.0
                        image1[np.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20] = .5
                        si1, si2 = self.slice_helper(offset[0], image1.shape[0])
                        sj1, sj2 = self.slice_helper(offset[1], image1.shape[1])
                        image2 = np.zeros(image1.shape)
                        if method == A.M_MUTUAL_INFORMATION:
                            image2[si2, sj2] = 1 - image1[si1, sj1]
                        else:
                            image2 = image1[(i + shape[0] - offset[0]) % shape[0],
                                            (j + shape[1] - offset[1]) % shape[1]]
                            image2 += (np.random.uniform(size=shape) - .5) * .1 * np.std(image2)
                        if mask1 is not None:
                            image1[~ mask1] = np.random.uniform(size=np.sum(~mask1))
                        if mask2 is not None:
                            image2[~ mask2] = np.random.uniform(size=np.sum(~mask2))
                        workspace, module = self.make_workspace((image1, image2),
                                                                (mask1, mask2))
                        self.assertTrue(isinstance(module, A.Align))
                        module.alignment_method.value = method
                        module.crop_mode.value = A.C_PAD
                        module.run(workspace)
                        output = workspace.image_set.get_image('Aligned0')
                        m = workspace.measurements
                        self.assertTrue(isinstance(m, cpmeas.Measurements))
                        off_i0 = -m.get_current_image_measurement('Align_Yshift_Aligned0')
                        off_j0 = -m.get_current_image_measurement('Align_Xshift_Aligned0')
                        off_i1 = -m.get_current_image_measurement('Align_Yshift_Aligned1')
                        off_j1 = -m.get_current_image_measurement('Align_Xshift_Aligned1')

                        self.assertEqual(off_i0 - off_i1, offset[0])
                        self.assertEqual(off_j0 - off_j1, offset[1])

                        i_slice = slice(off_i0, off_i0 + image1.shape[0])
                        j_slice = slice(off_j0, off_j0 + image1.shape[1])
                        np.testing.assert_almost_equal(
                                image1, output.pixel_data[i_slice, j_slice])
                        if mask1 is not None:
                            self.assertTrue(np.all(output.mask[i_slice, j_slice] == mask1))

                        temp = output.mask.copy()
                        temp[i_slice, j_slice] = False
                        self.assertTrue(np.all(~ temp))

                        output = workspace.image_set.get_image("Aligned1")
                        i_slice = slice(off_i1, off_i1 + image2.shape[0])
                        j_slice = slice(off_j1, off_j1 + image2.shape[1])
                        np.testing.assert_almost_equal(
                                image2, output.pixel_data[i_slice, j_slice])
                        if mask2 is not None:
                            self.assertTrue(np.all(mask2 == output.mask[i_slice, j_slice]))
                        temp = output.mask.copy()
                        temp[i_slice, j_slice] = False
                        self.assertTrue(np.all(~ temp))

    def test_02_03_same_size(self):
        '''Align two images keeping sizes the same'''
        np.random.seed(0)
        shape = (50, 45)
        i, j = np.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((1, 0), (0, 1), (1, 1), (3, 5), (-3, 5), (3, -5), (-3, -5), (0, 5), (3, 0), (0, 0)):
            for mask1 in (None,
                          np.random.uniform(size=shape) > .1):
                for mask2 in (None,
                              np.random.uniform(size=shape) > .1):
                    for method in (A.M_MUTUAL_INFORMATION, A.M_CROSS_CORRELATION):
                        if method == A.M_CROSS_CORRELATION and (
                                    (mask1 is not None) or (mask2 is not None)):
                            continue
                        image1 = np.random.randint(0, 10, size=shape).astype(float) / 10.0
                        image1[np.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20] = .5
                        si1, si2 = self.slice_helper(offset[0], image1.shape[0])
                        sj1, sj2 = self.slice_helper(offset[1], image1.shape[1])
                        image2 = np.zeros(image1.shape)
                        if method == A.M_MUTUAL_INFORMATION:
                            image2[si2, sj2] = 1 - image1[si1, sj1]
                        else:
                            image2 = image1[(i + shape[0] - offset[0]) % shape[0],
                                            (j + shape[1] - offset[1]) % shape[1]]
                            image2 += (np.random.uniform(size=shape) - .5) * .1 * np.std(image2)
                        if mask1 is not None:
                            image1[~ mask1] = np.random.uniform(size=np.sum(~mask1))
                        if mask2 is not None:
                            image2[~ mask2] = np.random.uniform(size=np.sum(~mask2))
                        workspace, module = self.make_workspace((image1, image2),
                                                                (mask1, mask2))
                        self.assertTrue(isinstance(module, A.Align))
                        module.alignment_method.value = method
                        module.crop_mode.value = A.C_SAME_SIZE
                        module.run(workspace)
                        output = workspace.image_set.get_image('Aligned0')
                        m = workspace.measurements
                        self.assertTrue(isinstance(m, cpmeas.Measurements))
                        off_i0 = -m.get_current_image_measurement('Align_Yshift_Aligned0')
                        off_j0 = -m.get_current_image_measurement('Align_Xshift_Aligned0')
                        off_i1 = -m.get_current_image_measurement('Align_Yshift_Aligned1')
                        off_j1 = -m.get_current_image_measurement('Align_Xshift_Aligned1')

                        self.assertEqual(off_i0 - off_i1, offset[0])
                        self.assertEqual(off_j0 - off_j1, offset[1])

                        si_in, si_out = self.slice_same(off_i0, shape[0])
                        sj_in, sj_out = self.slice_same(off_j0, shape[1])
                        np.testing.assert_almost_equal(
                                image1[si_in, sj_in], output.pixel_data[si_out, sj_out])
                        if mask1 is not None:
                            self.assertTrue(np.all(output.mask[si_out, sj_out] == mask1[si_in, sj_in]))

                        temp = output.mask.copy()
                        temp[si_out, sj_out] = False
                        self.assertTrue(np.all(~ temp))

                        output = workspace.image_set.get_image("Aligned1")
                        si_in, si_out = self.slice_same(off_i1, shape[0])
                        sj_in, sj_out = self.slice_same(off_j1, shape[1])
                        np.testing.assert_almost_equal(
                                image2[si_in, sj_in], output.pixel_data[si_out, sj_out])
                        if mask2 is not None:
                            self.assertTrue(np.all(mask2[si_in, sj_in] == output.mask[si_out, sj_out]))
                        temp = output.mask.copy()
                        temp[si_out, sj_out] = False
                        self.assertTrue(np.all(~ temp))

    def slice_same(self, offset, orig_size):
        if offset < 0:
            return slice(-offset, orig_size), slice(0, orig_size + offset)
        else:
            return slice(0, orig_size - offset), slice(offset, orig_size)

    def test_03_01_align_similarly(self):
        '''Align a third image similarly to the other two'''
        np.random.seed(0)
        shape = (53, 62)
        i, j = np.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((3, 5), (-3, 5), (3, -5), (-3, -5)):
            image1 = np.random.randint(0, 10, size=shape).astype(float) / 10.0
            image1[np.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20] = .5
            si1, si2 = self.slice_helper(offset[0], image1.shape[0])
            sj1, sj2 = self.slice_helper(offset[1], image1.shape[1])
            image2 = np.zeros(image1.shape)
            image2 = image1[(i + shape[0] - offset[0]) % shape[0],
                            (j + shape[1] - offset[1]) % shape[1]]
            image2 += (np.random.uniform(size=shape) - .5) * .1 * np.std(image2)
            image3 = (i * 100 + j).astype(np.float32) / 10000
            workspace, module = self.make_workspace((image1, image2, image3),
                                                    (None, None, None))
            self.assertTrue(isinstance(module, A.Align))
            module.alignment_method.value = A.M_CROSS_CORRELATION
            module.crop_mode.value = A.C_PAD
            module.additional_images[0].align_choice.value = A.A_SIMILARLY
            module.run(workspace)
            output = workspace.image_set.get_image('Aligned2')
            m = workspace.measurements
            columns = module.get_measurement_columns(workspace.pipeline)
            self.assertEqual(len(columns), 6)
            align_measurements = [x for x in m.get_feature_names(cpmeas.IMAGE)
                                  if x.startswith('Align')]
            self.assertEqual(len(align_measurements), 6)
            self.assertTrue(isinstance(m, cpmeas.Measurements))
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
            np.testing.assert_almost_equal(output.pixel_data[i_slice, j_slice],
                                           image3)

    def test_03_02_align_separately(self):
        '''Align a third image to the first image'''
        np.random.seed(0)
        shape = (47, 53)
        i, j = np.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((3, 5), (-3, 5), (3, -5), (-3, -5)):
            image1 = np.random.uniform(size=shape).astype(np.float32)
            image2 = image1[(i + shape[0] - offset[0] - 5) % shape[0],
                            (j + shape[1] - offset[1] - 5) % shape[1]]
            image3 = image1[(i + shape[0] - offset[0]) % shape[0],
                            (j + shape[1] - offset[1]) % shape[1]]
            workspace, module = self.make_workspace((image1, image2, image3),
                                                    (None, None, None))
            self.assertTrue(isinstance(module, A.Align))
            module.alignment_method.value = A.M_CROSS_CORRELATION
            module.crop_mode.value = A.C_PAD
            module.additional_images[0].align_choice.value = A.A_SEPARATELY
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cpmeas.Measurements))
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
            np.testing.assert_almost_equal(output.pixel_data[i_slice, j_slice],
                                           image3)

    def test_03_03_align_color(self):
        np.random.seed(0)
        shape = (50, 45, 3)
        i, j = np.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((1, 0), (0, 1), (1, 1), (3, 5), (-3, 5), (3, -5), (-3, -5), (0, 5), (3, 0), (0, 0)):
            for mask1 in (None,
                          np.random.uniform(size=shape[:2]) > .1):
                for mask2 in (None,
                              np.random.uniform(size=shape[:2]) > .1):
                    for method in (A.M_MUTUAL_INFORMATION, A.M_CROSS_CORRELATION):
                        if method == A.M_CROSS_CORRELATION and (
                                    (mask1 is not None) or (mask2 is not None)):
                            continue
                        image1 = np.dstack([
                                               np.random.randint(0, 10, size=shape[:2])
                                           .astype(float) / 10.0] * 3)
                        image1[np.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20, :] = .5
                        si1, si2 = self.slice_helper(offset[0], image1.shape[0])
                        sj1, sj2 = self.slice_helper(offset[1], image1.shape[1])
                        image2 = np.zeros(image1.shape)
                        if method == A.M_MUTUAL_INFORMATION:
                            image2[si2, sj2, :] = 1 - image1[si1, sj1, :]
                        else:
                            image2 = image1[(i + shape[0] - offset[0]) % shape[0],
                                     (j + shape[1] - offset[1]) % shape[1], :]
                            image2 += (np.random.uniform(size=shape) - .5) * .1 * np.std(image2)
                        if mask1 is not None:
                            image1[~ mask1, :] = np.random.uniform(
                                    size=(np.sum(~mask1), shape[2]))
                        if mask2 is not None:
                            image2[~ mask2, :] = np.random.uniform(
                                    size=(np.sum(~mask2), shape[2]))
                        workspace, module = self.make_workspace(
                                (image1, image2), (mask1, mask2))
                        self.assertTrue(isinstance(module, A.Align))
                        module.alignment_method.value = method
                        module.crop_mode.value = A.C_SAME_SIZE
                        module.run(workspace)
                        output = workspace.image_set.get_image('Aligned0')
                        m = workspace.measurements
                        self.assertTrue(isinstance(m, cpmeas.Measurements))
                        off_i0 = -m.get_current_image_measurement('Align_Yshift_Aligned0')
                        off_j0 = -m.get_current_image_measurement('Align_Xshift_Aligned0')
                        off_i1 = -m.get_current_image_measurement('Align_Yshift_Aligned1')
                        off_j1 = -m.get_current_image_measurement('Align_Xshift_Aligned1')

                        self.assertEqual(off_i0 - off_i1, offset[0])
                        self.assertEqual(off_j0 - off_j1, offset[1])

                        si_in, si_out = self.slice_same(off_i0, shape[0])
                        sj_in, sj_out = self.slice_same(off_j0, shape[1])
                        np.testing.assert_almost_equal(
                                image1[si_in, sj_in, :], output.pixel_data[si_out, sj_out, :])
                        if mask1 is not None:
                            self.assertTrue(np.all(output.mask[si_out, sj_out] == mask1[si_in, sj_in]))

                        temp = output.mask.copy()
                        temp[si_out, sj_out] = False
                        self.assertTrue(np.all(~ temp))

                        output = workspace.image_set.get_image("Aligned1")
                        si_in, si_out = self.slice_same(off_i1, shape[0])
                        sj_in, sj_out = self.slice_same(off_j1, shape[1])
                        np.testing.assert_almost_equal(
                                image2[si_in, sj_in, :], output.pixel_data[si_out, sj_out, :])
                        if mask2 is not None:
                            self.assertTrue(np.all(mask2[si_in, sj_in] == output.mask[si_out, sj_out]))
                        temp = output.mask.copy()
                        temp[si_out, sj_out] = False
                        self.assertTrue(np.all(~ temp))

    def test_03_04_align_binary(self):
        np.random.seed(0)
        shape = (50, 45)
        i, j = np.mgrid[0:shape[0], 0:shape[1]]
        for offset in ((1, 0), (0, 1), (1, 1), (3, 5), (-3, 5), (3, -5), (-3, -5), (0, 5), (3, 0), (0, 0)):
            for mask1 in (None,
                          np.random.uniform(size=shape) > .1):
                for mask2 in (None,
                              np.random.uniform(size=shape) > .1):
                    for method in (A.M_MUTUAL_INFORMATION, A.M_CROSS_CORRELATION):
                        if method == A.M_CROSS_CORRELATION and (
                                    (mask1 is not None) or (mask2 is not None)):
                            continue
                        image1 = np.random.randint(0, 1, size=shape).astype(bool)
                        image1[np.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 10] = True
                        si1, si2 = self.slice_helper(offset[0], image1.shape[0])
                        sj1, sj2 = self.slice_helper(offset[1], image1.shape[1])
                        image2 = np.zeros(image1.shape, bool)
                        if method == A.M_MUTUAL_INFORMATION:
                            image2[si2, sj2] = 1 - image1[si1, sj1]
                        else:
                            image2 = image1[(i + shape[0] - offset[0]) % shape[0],
                                            (j + shape[1] - offset[1]) % shape[1]]
                        if mask1 is not None:
                            image1[~ mask1] = np.random.uniform(size=np.sum(~mask1))
                        if mask2 is not None:
                            image2[~ mask2] = np.random.uniform(size=np.sum(~mask2))
                        workspace, module = self.make_workspace((image1, image2),
                                                                (mask1, mask2))
                        self.assertTrue(isinstance(module, A.Align))
                        module.alignment_method.value = method
                        module.crop_mode.value = A.C_SAME_SIZE
                        module.run(workspace)
                        output = workspace.image_set.get_image('Aligned0')
                        m = workspace.measurements
                        self.assertTrue(isinstance(m, cpmeas.Measurements))
                        off_i0 = -m.get_current_image_measurement('Align_Yshift_Aligned0')
                        off_j0 = -m.get_current_image_measurement('Align_Xshift_Aligned0')
                        off_i1 = -m.get_current_image_measurement('Align_Yshift_Aligned1')
                        off_j1 = -m.get_current_image_measurement('Align_Xshift_Aligned1')

                        self.assertEqual(off_i0 - off_i1, offset[0])
                        self.assertEqual(off_j0 - off_j1, offset[1])

                        si_in, si_out = self.slice_same(off_i0, shape[0])
                        sj_in, sj_out = self.slice_same(off_j0, shape[1])
                        self.assertEqual(output.pixel_data.dtype.kind, "b")
                        np.testing.assert_equal(
                                image1[si_in, sj_in], output.pixel_data[si_out, sj_out])
                        if mask1 is not None:
                            self.assertTrue(np.all(output.mask[si_out, sj_out] == mask1[si_in, sj_in]))

                        temp = output.mask.copy()
                        temp[si_out, sj_out] = False
                        self.assertTrue(np.all(~ temp))

                        output = workspace.image_set.get_image("Aligned1")
                        si_in, si_out = self.slice_same(off_i1, shape[0])
                        sj_in, sj_out = self.slice_same(off_j1, shape[1])
                        np.testing.assert_equal(
                                image2[si_in, sj_in], output.pixel_data[si_out, sj_out])
                        if mask2 is not None:
                            self.assertTrue(np.all(mask2[si_in, sj_in] == output.mask[si_out, sj_out]))
                        temp = output.mask.copy()
                        temp[si_out, sj_out] = False
                        self.assertTrue(np.all(~ temp))

    def test_04_01_measurement_columns(self):
        workspace, module = self.make_workspace((np.zeros((10, 10)),
                                                 np.zeros((10, 10)),
                                                 np.zeros((10, 10))),
                                                (None, None, None))
        self.assertTrue(isinstance(module, A.Align))
        columns = module.get_measurement_columns(workspace.pipeline)
        self.assertEqual(len(columns), 6)
        for i in range(3):
            for axis in ("X", "Y"):
                feature = A.MEASUREMENT_FORMAT % (axis, "Aligned%d" % i)
                self.assertTrue(feature in [c[1] for c in columns])
        self.assertTrue(all([c[0] == cpmeas.IMAGE for c in columns]))
        self.assertTrue(all([c[2] == cpmeas.COLTYPE_INTEGER for c in columns]))

    def test_04_02_categories(self):
        workspace, module = self.make_workspace((np.zeros((10, 10)),
                                                 np.zeros((10, 10)),
                                                 np.zeros((10, 10))),
                                                (None, None, None))
        self.assertTrue(isinstance(module, A.Align))
        c = module.get_categories(workspace.pipeline, cpmeas.IMAGE)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0], A.C_ALIGN)

        c = module.get_categories(workspace.pipeline, 'Aligned0')
        self.assertEqual(len(c), 0)

    def test_04_03_measurements(self):
        workspace, module = self.make_workspace((np.zeros((10, 10)),
                                                 np.zeros((10, 10)),
                                                 np.zeros((10, 10))),
                                                (None, None, None))
        self.assertTrue(isinstance(module, A.Align))
        m = module.get_measurements(workspace.pipeline, cpmeas.IMAGE, A.C_ALIGN)
        self.assertEqual(len(m), 2)
        self.assertTrue("Xshift" in m)
        self.assertTrue("Yshift" in m)

    def test_04_04_measurement_images(self):
        workspace, module = self.make_workspace((np.zeros((10, 10)),
                                                 np.zeros((10, 10)),
                                                 np.zeros((10, 10))),
                                                (None, None, None))
        self.assertTrue(isinstance(module, A.Align))
        for measurement in ("Xshift", "Yshift"):
            image_names = module.get_measurement_images(
                    workspace.pipeline, cpmeas.IMAGE, A.C_ALIGN, measurement)
            self.assertEqual(len(image_names), 3)
            for i in range(3):
                self.assertTrue("Aligned%d" % i in image_names)

    def test_05_01_align_self(self):
        '''Align an image from the fly screen against itself.

        This is a regression test for the bug, IMG-284
        '''
        image = tests.modules.read_example_image("ExampleFlyImages", '01_POS002_D.TIF')
        image = image[0:300, 0:300]  # make smaller so as to be faster
        workspace, module = self.make_workspace((image, image), (None, None))
        module.alignment_method.value = A.M_MUTUAL_INFORMATION
        module.crop_mode.value = A.C_PAD
        module.run(workspace)
        m = workspace.measurements
        self.assertEqual(m.get_current_image_measurement('Align_Xshift_Aligned1'), 0)
        self.assertEqual(m.get_current_image_measurement('Align_Yshift_Aligned1'), 0)

    def test_06_01_different_sizes_crop(self):
        '''Test align with images of different sizes

        regression test of img-1300
        '''
        np.random.seed(61)
        shape = (61, 43)
        for method in (A.M_CROSS_CORRELATION, A.M_MUTUAL_INFORMATION):
            i, j = np.mgrid[0:shape[0], 0:shape[1]]
            image1 = np.random.randint(0, 10, size=shape).astype(float) / 10.0
            image1[np.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20] = .5
            image2 = image1[2:-2, 2:-2]
            for order, i1_name, i2_name in (
                    ((image1, image2), "Aligned0", "Aligned1"),
                    ((image2, image1), "Aligned1", "Aligned0")):
                workspace, module = self.make_workspace(order, (None, None))
                self.assertTrue(isinstance(module, A.Align))
                module.alignment_method.value = method
                module.crop_mode.value = A.C_CROP
                module.run(workspace)
                i1 = workspace.image_set.get_image(i1_name)
                self.assertTrue(isinstance(i1, cpi.Image))
                p1 = i1.pixel_data
                i2 = workspace.image_set.get_image(i2_name)
                p2 = i2.pixel_data
                self.assertEqual(tuple(p1.shape), tuple(p2.shape))
                self.assertTrue(np.all(p1 == p2))
                self.assertTrue(i1.has_crop_mask)
                crop_mask = np.zeros(shape, bool)
                crop_mask[2:-2, 2:-2] = True
                self.assertTrue(np.all(i1.crop_mask == crop_mask))
                self.assertFalse(i2.has_crop_mask)

    def test_06_02_different_sizes_pad(self):
        '''Test align with images of different sizes

        regression test of img-1300
        '''
        np.random.seed(612)
        shape = (61, 43)
        i, j = np.mgrid[0:shape[0], 0:shape[1]]
        image1 = np.random.randint(0, 10, size=shape).astype(float) / 10.0
        image1[np.sqrt(((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)) < 20] = .5
        image2 = image1[2:-2, 2:-2]
        workspace, module = self.make_workspace((image1, image2), (None, None))
        self.assertTrue(isinstance(module, A.Align))
        module.crop_mode.value = A.C_PAD
        module.run(workspace)
        i1 = workspace.image_set.get_image("Aligned0")
        self.assertTrue(isinstance(i1, cpi.Image))
        p1 = i1.pixel_data
        i2 = workspace.image_set.get_image("Aligned1")
        p2 = i2.pixel_data
        self.assertEqual(tuple(p1.shape), tuple(p2.shape))
        self.assertTrue(np.all(p1[2:-2, 2:-2] == p2[2:-2, 2:-2]))
        self.assertFalse(i1.has_mask)
        mask = np.zeros(shape, bool)
        mask[2:-2, 2:-2] = True
        self.assertTrue(i2.has_mask)
        self.assertTrue(np.all(mask == i2.mask))
