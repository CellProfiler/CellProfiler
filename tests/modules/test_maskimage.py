import StringIO
import base64
import unittest
import zlib

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.maskimage
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import numpy

cellprofiler.preferences.set_headless()

MASKING_IMAGE_NAME = "maskingimage"
MASKED_IMAGE_NAME = "maskedimage"
IMAGE_NAME = "image"
OBJECTS_NAME = "objects"


class TestMaskImage(unittest.TestCase):
    def test_01_02_load_v1(self):
        data = ('eJztWutu0zAUTrpuYiBxEUjwB8k/+LFNa0g3qrEJQTvKpRLtKlpx0TbAS93V'
                '4MZV4owVxDvwODwGj8IjkJOla2qypet6A2Ipao/jz9/nk+PEOXExV32R20QZ'
                'TUfFXDVVp4ygMsOizq3mBmpxmx4uo8cWwYLUEDc3UJGbqMQP0KqO0qsbmczG'
                'io5WdH1dGaCoheJl9+dHSlHm3N8L7pHwT836tho4wK4QIai5b88qSeWWX//L'
                'PV5hi+I9Rl5h5hC7S9GpL5h1Xm23jk8Vec1hpISbwcZuKTnNPWLZW/UO0D9d'
                'poeEVegXIg2h0+wlOaA25aaP9/uXa495uZB4Kw3++anlypH638TCaFSE6/7e'
                'evDb9xtdv6mS32bc43agHto/V7rtkyF+vhZof9W3qVmjB7TmYIZoE+8fq4b+'
                '9Ij+Znr6m1HypZyHux+Bm5N0gF1yDEboEW82An9VwsNRJYci9eQQGwI1waXD'
                '0DHo+KNwag9OVVZ9f0fpnZX0gp3Wl+/pfeIvSHiwi9j+tOlOqX78flHCg53n'
                'yOQCObYfv4P47a0bdf3gEj24hFLi/fGNEjcn4Tqlg5tXun6Jmp83Jf+CnSd1'
                '7DCBCjA5UZ5axBDcag/Nz2fRPyguG6HzkjRusLeE7aBnjO9hdm7+UcWjjEtr'
                '+kh1nhSPg9z3dU33ynLa/3OCjnHoj8Ile3BJ0J6exjieV3r9DHbBFMS0qWgP'
                'gf9f0D3M56gcT2l9tDrl53Z6QNxaCG6YOuX5UuImmcTzohHBt6b0xh3Y7xYe'
                'lR/ACwp5qC0tvgfrNWHs4XYuVd7d1lPru19Xvi2+t+FEhbqtvLrFvuL8isQH'
                'dtlyl71We8sRjJrnHO+bCP47Ej/Y2tL2zs7dXRhO3h/0ccVLxwT7Tqf/v+V5'
                'H+ucrE59iOuQcd+XpkHnsNf9cXzK8ZmZynVQ2HtI9TNHBsO27WdGJqF7kPX9'
                'a0L3G5DTO4AElmmQQH/T5vewvMJTbpF9iztmbXK6/5X5J68/M+fk+3G9i1Ml'
                'XFhecpxx4yUxIXBa/fcTNu/53kdiiG5H456/AX5EzRppjcAv03YfCMvrdnUf'
                'uWHS8yvGxbgYF+Ni3Ohx2QAufl7EuDh+YlyMmzwuq5w+r6b1/SPGxbgYF+P+'
                'd1xD7eLk/J2c14T2HwI8Yff7JaX3fg+2QRhrWRz2m1pa09sUaWuM49rRLkPt'
                'hfu3ENhwCDytCJ6sxJM9iYfWiClovd2Cj7uO4E0sqKEV/Fr45Jvr1AJv1Hfb'
                'BYl34STeJrY/eePTYFudNzz5Os2H9B/0d8K1rt1Oev4I7ueTr6+c151Xgtf9'
                '16NBeJNJ9Q/eSxG4ZEAbFC8/rJwtvhZOad8p42x/Vr+p6pHffp5x3KfxyLoS'
                'E8L9BgVhjWE=')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.maskimage.MaskImage))
        self.assertEqual(module.source_choice, cellprofiler.modules.maskimage.IO_OBJECTS)
        self.assertEqual(module.object_name, "Nuclei")
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.masked_image_name, "MaskBlue")
        self.assertFalse(module.invert_mask)

    def test_01_03_load_v2(self):
        data = ('eJztXNFu2zYUpRzHa1pgcPuyvBTg3rIhNeS0Adq8zM6ybgZqJ1iMbnsbbdE2'
                'F1k0JCqJ9wX7jH3eHvsJExUpkhm5km1JpQMKEKRL89xzeXl5Scq2uu3+h/Yp'
                'PG7osNvuvxoRE8MLE7ERtacn0GKH8EcbI4YNSK0T2KUW7NFr+FqHzTcnevNE'
                'fwePdP0dWO/QOt2vvcs/HQBq3vWJd1aCj3YDWYudXL7EjBFr7OyCKtgPyj95'
                '50dkEzQw8UdkutiJKMLyjjWi/fns/qMuNVwT99A0Xtk7eu50gG3nfBQCg48v'
                'yC02L8nfWGhCWO1XfE0cQq0AH+gXS+95KRN4Lyf05r3tmSPoP0VsOLlkXg8s'
                'lnO//ftt5DdN8Bu/voyV8/q/gKh+NcHPz2P164FMLINcE8NFJiRTNL63mut7'
                'm6LviaCPy+c2GZ96XZQFXxPwXO65QxOTfPj1FHx1AV8FPWr5uFYKri7w8rOP'
                'b9mrn27RkMEp71IZ2p/WjqcCnsv9iY2dCddwV15GP2gLeA28zsi7K/Byuakf'
                'vtE3sLuLnKvQ7rTxtC/guXxKLGTP4cHARMMriCwD3kwIw9/lYU8rBZ/Un2cU'
                'WpRB14n1Z1p/7Czo2QF/eFkhC66ygKt44ykb37o4WexM69evwGK/cPl88Bce'
                'ss3sXWd8dHiSB9nmi28EPJfP8Ai5JoO+HnhGbK8V1J4X2l/L8vSqfHpDB3nm'
                'xVXzWiPApfE/E/i5fM4cF/5s0gEyE/lrgp7wCPXsgex2ZxkfZfNl8W+zYL+I'
                '8dQM4kk2vzyI+8NmqX7RJfWLmEc8O5ub2NlK4StzHOdp955gN5c7FsOWQ9g8'
                'B/7V4+m4UD4xLs7aFx0Z/JQ2Pyft57yY9o/DZnCTgx1Fr//F8dzUi42vsueN'
                'ZfG1qp3HJdu5aX4sa12XtI4s204Z/JK3na0UvqT5rX9D4dBEjhM8SZLRT3nH'
                'zzp5+jdMxhP+2PWaP2C0hjimr+z1QNn+aqXwJe2/3lMbj23qWka+7ZXZT9uI'
                'K3t/IDOuBT7vF1n3B7LFT5HPI7YFp/JNMi5LvHjbxy9up6z5pqj1msy4x7Re'
                'kyGutmG9JkNeUvm6/OccZeD+gxFOE3D8Kv5Oo8xx6P+ogw/EWXY9SfME9b+v'
                'jBTJaE+Z80GMHxLLwLOYvrL8Uhf01Bf03JmVpz2yzTtZ27+t/AqncDKMs6Lz'
                'w5fmVziFUziFUziFe0zrV4VTOBnivOj9uxqfCqdwCqdw24lrxXBq/69wCqdw'
                'CqdwCqdwCicf7m0lwmkCTgvutVj9P2M8Seu772P164E8xKY5syl/X4ndmPov'
                '1XAaJkXG3VsqGh+8207shRWcZ5bC0xJ4Wst4iIEtRkbzme2xuYxOESPDRico'
                'vfBK22Ep5/09hfdA4D1YxjtFzpXfvgb/27/fPK5/kqL/SNB/tEw/ms3MOfP/'
                'yUxNo9HmYj8UIz8W3Z74/8n2EvTH46cSyM9f1l7UwOfjFoDFeI3i+NMP6/JW'
                'q5WdCni4T3mWgud+ewoeHlzPvrba+DkAy+uHbX5M9dfpJ40fYHP/RnzVe9tC'
                'nsdQ/39X3jff')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 5)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, cellprofiler.modules.maskimage.MaskImage))
        self.assertEqual(module.source_choice, cellprofiler.modules.maskimage.IO_OBJECTS)
        self.assertEqual(module.object_name, "Nuclei")
        self.assertEqual(module.image_name, "OrigBlue")
        self.assertEqual(module.masked_image_name, "MaskBlue")
        self.assertFalse(module.invert_mask)

        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.maskimage.MaskImage))
        self.assertEqual(module.source_choice, cellprofiler.modules.maskimage.IO_IMAGE)
        self.assertEqual(module.masking_image_name, "ThreshBlue")
        self.assertEqual(module.image_name, "OrigBlue")
        self.assertEqual(module.masked_image_name, "MaskBlue")
        self.assertTrue(module.invert_mask)

    def test_02_01_mask_with_objects(self):
        labels = numpy.zeros((10, 15), int)
        labels[2:5, 3:8] = 1
        labels[5:8, 10:14] = 2
        object_set = cellprofiler.region.Set()
        objects = cellprofiler.region.Region()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)

        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        numpy.random.seed(0)
        pixel_data = numpy.random.uniform(size=(10, 15)).astype(numpy.float32)
        image_set.add(IMAGE_NAME, cellprofiler.image.Image(pixel_data))

        pipeline = cellprofiler.pipeline.Pipeline()
        module = cellprofiler.modules.maskimage.MaskImage()
        module.source_choice.value = cellprofiler.modules.maskimage.IO_OBJECTS
        module.object_name.value = OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.masked_image_name.value = MASKED_IMAGE_NAME
        module.invert_mask.value = False
        module.module_num = 1

        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set, object_set,
                                                     cellprofiler.measurement.Measurements(), image_set_list)
        module.run(workspace)
        masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
        self.assertTrue(isinstance(masked_image, cellprofiler.image.Image))
        self.assertTrue(numpy.all(masked_image.pixel_data[labels > 0] ==
                                  pixel_data[labels > 0]))
        self.assertTrue(numpy.all(masked_image.pixel_data[labels == 0] == 0))
        self.assertTrue(numpy.all(masked_image.mask == (labels > 0)))
        self.assertTrue(numpy.all(masked_image.masking_objects.segmented == labels))

    def test_02_02_mask_invert(self):
        labels = numpy.zeros((10, 15), int)
        labels[2:5, 3:8] = 1
        labels[5:8, 10:14] = 2
        object_set = cellprofiler.region.Set()
        objects = cellprofiler.region.Region()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)

        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        numpy.random.seed(0)
        pixel_data = numpy.random.uniform(size=(10, 15)).astype(numpy.float32)
        image_set.add(IMAGE_NAME, cellprofiler.image.Image(pixel_data))

        pipeline = cellprofiler.pipeline.Pipeline()
        module = cellprofiler.modules.maskimage.MaskImage()
        module.source_choice.value = cellprofiler.modules.maskimage.IO_OBJECTS
        module.object_name.value = OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.masked_image_name.value = MASKED_IMAGE_NAME
        module.invert_mask.value = True
        module.module_num = 1

        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set, object_set,
                                                     cellprofiler.measurement.Measurements(), image_set_list)
        module.run(workspace)
        masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
        self.assertTrue(isinstance(masked_image, cellprofiler.image.Image))
        self.assertTrue(numpy.all(masked_image.pixel_data[labels == 0] ==
                                  pixel_data[labels == 0]))
        self.assertTrue(numpy.all(masked_image.pixel_data[labels > 0] == 0))
        self.assertTrue(numpy.all(masked_image.mask == (labels == 0)))
        self.assertTrue(numpy.all(masked_image.masking_objects.segmented == labels))

    def test_02_03_double_mask(self):
        labels = numpy.zeros((10, 15), int)
        labels[2:5, 3:8] = 1
        labels[5:8, 10:14] = 2
        object_set = cellprofiler.region.Set()
        objects = cellprofiler.region.Region()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)

        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        numpy.random.seed(0)
        pixel_data = numpy.random.uniform(size=(10, 15)).astype(numpy.float32)
        mask = numpy.random.uniform(size=(10, 15)) > .5
        image_set.add(IMAGE_NAME, cellprofiler.image.Image(pixel_data, mask))

        expected_mask = (mask & (labels > 0))

        pipeline = cellprofiler.pipeline.Pipeline()
        module = cellprofiler.modules.maskimage.MaskImage()
        module.source_choice.value = cellprofiler.modules.maskimage.IO_OBJECTS
        module.object_name.value = OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.masked_image_name.value = MASKED_IMAGE_NAME
        module.invert_mask.value = False
        module.module_num = 1

        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set, object_set,
                                                     cellprofiler.measurement.Measurements(), image_set_list)
        module.run(workspace)
        masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
        self.assertTrue(isinstance(masked_image, cellprofiler.image.Image))
        self.assertTrue(numpy.all(masked_image.pixel_data[expected_mask] ==
                                  pixel_data[expected_mask]))
        self.assertTrue(numpy.all(masked_image.pixel_data[~ expected_mask] == 0))
        self.assertTrue(numpy.all(masked_image.mask == expected_mask))
        self.assertTrue(numpy.all(masked_image.masking_objects.segmented == labels))

    def test_03_01_binary_mask(self):
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        numpy.random.seed(0)
        pixel_data = numpy.random.uniform(size=(10, 15)).astype(numpy.float32)
        image_set.add(IMAGE_NAME, cellprofiler.image.Image(pixel_data))

        masking_image = numpy.random.uniform(size=(10, 15)) > .5
        image_set.add(MASKING_IMAGE_NAME, cellprofiler.image.Image(masking_image))

        pipeline = cellprofiler.pipeline.Pipeline()
        module = cellprofiler.modules.maskimage.MaskImage()
        module.source_choice.value = cellprofiler.modules.maskimage.IO_IMAGE
        module.object_name.value = OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.masking_image_name.value = MASKING_IMAGE_NAME
        module.masked_image_name.value = MASKED_IMAGE_NAME
        module.invert_mask.value = False
        module.module_num = 1

        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set, cellprofiler.region.Set(),
                                                     cellprofiler.measurement.Measurements(), image_set_list)
        module.run(workspace)
        masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
        self.assertTrue(isinstance(masked_image, cellprofiler.image.Image))
        self.assertTrue(numpy.all(masked_image.pixel_data[masking_image] ==
                                  pixel_data[masking_image]))
        self.assertTrue(numpy.all(masked_image.pixel_data[~masking_image] == 0))
        self.assertTrue(numpy.all(masked_image.mask == masking_image))
        self.assertFalse(masked_image.has_masking_objects)

    def test_03_02_gray_mask(self):
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        numpy.random.seed(0)
        pixel_data = numpy.random.uniform(size=(10, 15)).astype(numpy.float32)
        image_set.add(IMAGE_NAME, cellprofiler.image.Image(pixel_data))

        masking_image = numpy.random.uniform(size=(10, 15))
        image_set.add(MASKING_IMAGE_NAME, cellprofiler.image.Image(masking_image))
        masking_image = masking_image > .5

        pipeline = cellprofiler.pipeline.Pipeline()
        module = cellprofiler.modules.maskimage.MaskImage()
        module.source_choice.value = cellprofiler.modules.maskimage.IO_IMAGE
        module.object_name.value = OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.masking_image_name.value = MASKING_IMAGE_NAME
        module.masked_image_name.value = MASKED_IMAGE_NAME
        module.invert_mask.value = False
        module.module_num = 1

        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set, cellprofiler.region.Set(),
                                                     cellprofiler.measurement.Measurements(), image_set_list)
        module.run(workspace)
        masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
        self.assertTrue(isinstance(masked_image, cellprofiler.image.Image))
        self.assertTrue(numpy.all(masked_image.pixel_data[masking_image] ==
                                  pixel_data[masking_image]))
        self.assertTrue(numpy.all(masked_image.pixel_data[~masking_image] == 0))
        self.assertTrue(numpy.all(masked_image.mask == masking_image))
        self.assertFalse(masked_image.has_masking_objects)

    def test_03_03_color_mask(self):
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        numpy.random.seed(0)
        pixel_data = numpy.random.uniform(size=(10, 15, 3)).astype(numpy.float32)
        image_set.add(IMAGE_NAME, cellprofiler.image.Image(pixel_data))

        masking_image = numpy.random.uniform(size=(10, 15))

        image_set.add(MASKING_IMAGE_NAME, cellprofiler.image.Image(masking_image))
        expected_mask = masking_image > .5

        pipeline = cellprofiler.pipeline.Pipeline()
        module = cellprofiler.modules.maskimage.MaskImage()
        module.source_choice.value = cellprofiler.modules.maskimage.IO_IMAGE
        module.object_name.value = OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.masking_image_name.value = MASKING_IMAGE_NAME
        module.masked_image_name.value = MASKED_IMAGE_NAME
        module.invert_mask.value = False
        module.module_num = 1

        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set, cellprofiler.region.Set(),
                                                     cellprofiler.measurement.Measurements(), image_set_list)
        module.run(workspace)
        masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
        self.assertTrue(isinstance(masked_image, cellprofiler.image.Image))
        self.assertTrue(numpy.all(masked_image.pixel_data[expected_mask, :] ==
                               pixel_data[expected_mask, :]))
        self.assertTrue(numpy.all(masked_image.pixel_data[~expected_mask, :] == 0))
        self.assertTrue(numpy.all(masked_image.mask == expected_mask))
        self.assertFalse(masked_image.has_masking_objects)
