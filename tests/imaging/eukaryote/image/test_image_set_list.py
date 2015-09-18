import imaging.eukaryote.image.callback_image_provider
import imaging.eukaryote.image.image
import imaging.eukaryote.image.image_set
import imaging.eukaryote.image.image_set_list
import numpy
import unittest

IMAGE_NAME = "image"


class TestImageSetList(unittest.TestCase):
    def test_00_00_init(self):
        x = imaging.eukaryote.image.image_set_list.ImageSetList()
        self.assertEqual(x.count(), 0, "# of elements of an empty image set list is %d, not zero" % (x.count()))

    def test_01_01_add_image_set_by_number(self):
        x = imaging.eukaryote.image.image_set_list.ImageSetList()
        y = x.get_image_set(0)
        self.assertEqual(x.count(), 1, "# of elements was %d, should be 1" % (x.count()))
        self.assertEqual(y.get_number(), 0, "The image set should be #0, was %d" % (y.get_number()))
        self.assertTrue(y.get_keys().has_key("number"), "The image set was missing a number key")
        self.assertEqual(y.get_keys()["number"], 0, "The number key should be zero, was %s" % (repr(y.get_keys()["number"])))

    def test_01_02_add_image_set_by_key(self):
        x = imaging.eukaryote.image.image_set_list.ImageSetList()
        key = {"key": "value"}
        y = x.get_image_set(key)
        self.assertEqual(x.count(), 1, "# of elements was %d, should be 1" % (x.count()))
        self.assertEqual(y.get_number(), 0, "The image set should be #0, was %d" % (y.get_number()))
        self.assertEquals(y, x.get_image_set(0), "The image set should be retrievable by index")
        self.assertEquals(y, x.get_image_set(key), "The image set should be retrievable by key")
        self.assertEquals(repr(key), repr(y.get_keys()))

    def test_01_03_add_two_image_sets(self):
        x = imaging.eukaryote.image.image_set_list.ImageSetList()
        y = x.get_image_set(0)
        z = x.get_image_set(1)
        self.assertEqual(x.count(), 2, "# of elements was %d, should be 2" % (x.count()))
        self.assertEqual(y.number, 0, "The image set should be #0, was %d" % (y.get_number()))
        self.assertEqual(z.number, 1, "The image set should be #1, was %d" % (y.get_number()))
        self.assertEquals(y, x.get_image_set(0), "The first image set was not retrieved by index")
        self.assertEquals(z, x.get_image_set(1), "The second image set was not retrieved by index")

    def test_02_01_add_image_provider(self):
        x = imaging.eukaryote.image.image_set_list.ImageSetList()
        y = x.get_image_set(0)
        img = imaging.eukaryote.image.image.Image(numpy.ones((10, 10)))

        def fn(image_set, image_provider):
            self.assertEquals(y, image_set, "Callback was not called with the correct image provider")
            return img

        z = imaging.eukaryote.image.callback_image_provider.CallbackImageProvider("TestImageProvider", fn)
        y.providers.append(z)
        self.assertEquals(img, y.get_image("TestImageProvider"))

    def test_02_02_add_two_image_providers(self):
        x = imaging.eukaryote.image.image_set_list.ImageSetList()
        y = x.get_image_set(0)
        img1 = imaging.eukaryote.image.image.Image(np.ones((10, 10)))

        def fn1(image_set, image_provider):
            self.assertEquals(y, image_set, "Callback was not called with the correct image set")
            return img1

        img2 = imaging.eukaryote.image.image.Image(np.ones((5, 5)))

        def fn2(image_set, image_provider):
            self.assertEquals(y, image_set, "Callback was not called with the correct image set")
            return img2

        y.providers.append(imaging.eukaryote.image.callback_image_provider.CallbackImageProvider("IP1", fn1))
        y.providers.append(imaging.eukaryote.image.callback_image_provider.CallbackImageProvider("IP2", fn2))
        self.assertEquals(img1, y.get_image("IP1"), "Failed to get correct first image")
        self.assertEquals(img2, y.get_image("IP2"), "Failed to get correct second image")

    def test_03_01_serialize_no_key(self):
        x = imaging.eukaryote.image.image_set_list.ImageSetList()
        for i in range(5):
            x.get_image_set(i)
        s = x.save_state()
        y = imaging.eukaryote.image.image_set_list.ImageSetList()
        y.load_state(s)
        self.assertEquals(y.count(), 5)

    def test_03_02_serialize_key(self):
        x = imaging.eukaryote.image.image_set_list.ImageSetList()
        values = (('A', 'B'), ('C', 'D'), ('E', 'F'))
        for value1, value2 in values:
            d = {'K1': value1, 'K2': value2}
            x.get_image_set(d)
        s = x.save_state()
        y = imaging.eukaryote.image.image_set_list.ImageSetList()
        y.load_state(s)
        self.assertEquals(y.count(), len(values))
        for i in range(len(values)):
            image_set = y.get_image_set(i)
            self.assertTrue(isinstance(image_set, imaging.eukaryote.image.image_set.ImageSet))
            value1, value2 = values[i]
            for key, value in (('K1', value1), ('K2', value2)):
                self.assertEqual(image_set.keys[key], value)

    def test_03_03_serialize_legacy_fields(self):
        x = imaging.eukaryote.image.image_set_list.ImageSetList()
        for i in range(5):
            x.get_image_set(i)
        d = {'foo': 'bar', 'test': 'suite'}
        x.legacy_fields['dictionary'] = d
        s = x.save_state()
        y = imaging.eukaryote.image.image_set_list.ImageSetList()
        y.load_state(s)
        self.assertEquals(y.count(), 5)
        self.assertTrue(y.legacy_fields.has_key('dictionary'))
        for key in d.keys():
            self.assertTrue(y.legacy_fields['dictionary'].has_key(key))
            self.assertEqual(y.legacy_fields['dictionary'][key], d[key])
