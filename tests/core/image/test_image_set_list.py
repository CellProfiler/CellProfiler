import numpy

import cellprofiler_core.image
import cellprofiler_core.image.abstract_image._callback_image


class TestImageSetList:
    def test_init(self):
        x = cellprofiler_core.image.ImageSetList()
        assert (
            x.count() == 0
        ), "# of elements of an empty image set list is %d, not zero" % (x.count())

    def test_add_image_set_by_number(self):
        x = cellprofiler_core.image.ImageSetList()
        y = x.get_image_set(0)
        assert x.count() == 1, "# of elements was %d, should be 1" % (x.count())
        assert y.number == 0, "The image set should be #0, was %d" % y.number
        assert "number" in y.keys, "The image set was missing a number key"
        assert y.keys["number"] == 0, "The number key should be zero, was %s" % (
            repr(y.keys["number"])
        )

    def test_add_image_set_by_key(self):
        x = cellprofiler_core.image.ImageSetList()
        key = {"key": "value"}
        y = x.get_image_set(key)
        assert x.count() == 1, "# of elements was %d, should be 1" % (x.count())
        assert y.number == 0, "The image set should be #0, was %d" % y.number
        assert y == x.get_image_set(0), "The image set should be retrievable by index"
        assert y == x.get_image_set(key), "The image set should be retrievable by key"
        assert repr(key) == repr(y.keys)

    def test_add_two_image_sets(self):
        x = cellprofiler_core.image.ImageSetList()
        y = x.get_image_set(0)
        z = x.get_image_set(1)
        assert x.count() == 2, "# of elements was %d, should be 2" % (x.count())
        assert y.number == 0, "The image set should be #0, was %d" % y.number
        assert z.number == 1, "The image set should be #1, was %d" % y.number
        assert y == x.get_image_set(0), "The first image set was not retrieved by index"
        assert z == x.get_image_set(
            1
        ), "The second image set was not retrieved by index"

    def test_add_image_provider(self):
        x = cellprofiler_core.image.ImageSetList()
        y = x.get_image_set(0)
        img = cellprofiler_core.image.Image(numpy.ones((10, 10)))

        def fn(image_set, image_provider):
            assert (
                y == image_set
            ), "Callback was not called with the correct image provider"
            return img

        z = cellprofiler_core.image.abstract_image._callback_image.CallbackImage(
            "TestImageProvider", fn
        )
        y.add_provider(z)
        assert img == y.get_image("TestImageProvider")

    def test_add_two_image_providers(self):
        x = cellprofiler_core.image.ImageSetList()
        y = x.get_image_set(0)
        img1 = cellprofiler_core.image.Image(numpy.ones((10, 10)))

        def fn1(image_set, image_provider):
            assert y == image_set, "Callback was not called with the correct image set"
            return img1

        img2 = cellprofiler_core.image.Image(numpy.ones((5, 5)))

        def fn2(image_set, image_provider):
            assert y == image_set, "Callback was not called with the correct image set"
            return img2

        y.add_provider(
            cellprofiler_core.image.abstract_image._callback_image.CallbackImage(
                "IP1", fn1
            )
        )
        y.add_provider(
            cellprofiler_core.image.abstract_image._callback_image.CallbackImage(
                "IP2", fn2
            )
        )
        assert img1 == y.get_image("IP1"), "Failed to get correct first image"
        assert img2 == y.get_image("IP2"), "Failed to get correct second image"

    def test_serialize_no_key(self):
        """Serialize an image list with no keys in the image sets"""
        x = cellprofiler_core.image.ImageSetList()
        for i in range(5):
            x.get_image_set(i)
        s = x.save_state()

        y = cellprofiler_core.image.ImageSetList()
        y.load_state(s)
        assert y.count() == 5

    def test_serialize_key(self):
        x = cellprofiler_core.image.ImageSetList()
        values = (("A", "B"), ("C", "D"), ("E", "F"))
        for value1, value2 in values:
            d = {"K1": value1, "K2": value2}
            x.get_image_set(d)
        s = x.save_state()

        y = cellprofiler_core.image.ImageSetList()
        y.load_state(s)
        assert y.count() == len(values)
        for i in range(len(values)):
            image_set = y.get_image_set(i)
            assert isinstance(image_set, cellprofiler_core.image.ImageSet)
            value1, value2 = values[i]
            for key, value in (("K1", value1), ("K2", value2)):
                assert image_set.keys[key] == value

    def test_serialize_legacy_fields(self):
        x = cellprofiler_core.image.ImageSetList()
        for i in range(5):
            x.get_image_set(i)
        d = {"foo": "bar", "test": "suite"}
        x.legacy_fields["dictionary"] = d
        s = x.save_state()

        y = cellprofiler_core.image.ImageSetList()
        y.load_state(s)
        assert y.count() == 5
        assert "dictionary" in y.legacy_fields
        for key in list(d.keys()):
            assert key in y.legacy_fields["dictionary"]
            assert y.legacy_fields["dictionary"][key] == d[key]
