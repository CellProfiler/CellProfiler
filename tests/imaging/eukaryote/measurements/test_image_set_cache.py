IMAGE_NAME = "ImageName"
ALT_IMAGE_NAME = "AltImageName"
OBJECT_NAME = "ObjectName"
ALT_OBJECT_NAME = "AltObjectName"
METADATA_NAMES = ["Metadata_%d" % i for i in range(1, 10)]


class TestImageSetCache(unittest.TestCase):
    def setUp(self):
        if sys.platform == "darwin":
            fd, self.path = tempfile.mkstemp(".h5")
            os.close(fd)
            self.f = h5py.File(self.path)
        else:
            self.f = h5py.File("foo", driver="core", backing_store=False)

    def tearDown(self):
        self.f.close()
        if sys.platform == "darwin":
            os.unlink(self.path)

    def test_01_01_no_cache(self):
        cache = cpmeas.ImageSetCache(self.f)
        self.assertFalse(cache.has_cache)

    def test_02_01_one_image(self):
        cache = cpmeas.ImageSetCache(self.f)
        urls = ["file:///foo", "file:///bar"]
        cache.cache_image_set(
            [(IMAGE_NAME, cpmeas.IMAGE)],
            [cpmeas.ImageSetCache.ImageSetData(
                tuple(),
                [cpmeas.ImageSetCache.ImageData(url, None, None, None)], [])
             for url in urls])
        for reopen in (False, True):
            if reopen:
                cache = cpmeas.ImageSetCache(self.f)
            self.assertTrue(cache.has_cache)
            self.assertSequenceEqual(cache.image_names, [IMAGE_NAME])
            self.assertIsNone(cache.metadata_keys)
            for i, url in enumerate(urls):
                image_set_data = cache.get_image_set_data(i)
                self.assertEqual(len(image_set_data.errors), 0)
                self.assertEqual(len(image_set_data.key), 0)
                self.assertEqual(len(image_set_data.ipds), 1)
                ipd = image_set_data.ipds[0]
                self.assertEqual(ipd.url, url)
                self.assertIsNone(ipd.series)
                self.assertIsNone(ipd.index)
                self.assertIsNone(ipd.channel)

    def test_02_02_image_and_metadata(self):
        cache = cpmeas.ImageSetCache(self.f)
        metadata_column_name = "Metadata_well"
        urls = ["file:///foo", "file:///bar"]
        series = (0, 2)
        index = (0, 1)
        channel = (1, 0)
        metadata = ["A01", "A02"]
        cache.cache_image_set(
            [(IMAGE_NAME, cpmeas.IMAGE)],
            [cpmeas.ImageSetCache.ImageSetData(
                (md,), [cpmeas.ImageSetCache.ImageData(url, s, i, c)], [])
             for url, md, s, i, c in zip(urls, metadata, series, index, channel)],
            metadata_keys=[metadata_column_name])
        for reopen in (False, True):
            if reopen:
                cache = cpmeas.ImageSetCache(self.f)
            self.assertTrue(cache.has_cache)
            self.assertSequenceEqual(cache.image_names, [IMAGE_NAME])
            self.assertSequenceEqual(cache.image_or_object, [cpmeas.IMAGE])
            self.assertSequenceEqual(cache.metadata_keys, [metadata_column_name])
            for idx, url, md, s, i, c in zip(
                    (0, 1), urls, metadata, series, index, channel):
                image_set_data = cache.get_image_set_data(idx)
                self.assertEqual(len(image_set_data.errors), 0)
                self.assertEqual(len(image_set_data.key), 1)
                self.assertEqual(image_set_data.key[0], md)
                self.assertEqual(len(image_set_data.ipds), 1)
                self.assertEqual(image_set_data.ipds[0].url, url)
                self.assertEqual(image_set_data.ipds[0].series, s)
                self.assertEqual(image_set_data.ipds[0].index, i)
                self.assertEqual(image_set_data.ipds[0].channel, c)

    def test_02_03_images_and_objects(self):
        cache = cpmeas.ImageSetCache(self.f)
        metadata_columns = ["%s_%d" % (cpmeas.C_METADATA, i) for i in range(1, 3)]
        image_names = [IMAGE_NAME, ALT_IMAGE_NAME, OBJECT_NAME, ALT_OBJECT_NAME]
        image_or_object = [cpmeas.IMAGE, cpmeas.IMAGE, cpmeas.OBJECT, cpmeas.OBJECT]
        data = [cpmeas.ImageSetCache.ImageSetData(
            tuple([uuid.uuid4().hex for _ in metadata_columns]),
            [cpmeas.ImageSetCache.ImageData("file:///%s" % uuid.uuid4().hex, None, None, None)
             for _ in image_names], []) for __ in range(4)]
        cache.cache_image_set(
            [(x, y) for x, y in zip(image_names, image_or_object)],
            data,
            metadata_columns)
        for reopen in (False, True):
            if reopen:
                cache = cpmeas.ImageSetCache(self.f)
            self.assertTrue(cache.has_cache)
            self.assertSequenceEqual(cache.image_names, image_names)
            self.assertSequenceEqual(cache.image_or_object, image_or_object)
            for i, expected in enumerate(data):
                image_set_data = cache.get_image_set_data(i)
                self.assertEqual(len(image_set_data.errors), 0)
                self.assertSequenceEqual(image_set_data.key, expected.key)
                self.assertEqual(len(image_set_data.ipds), len(expected.ipds))
                for ipd, eipd in zip(image_set_data.ipds, expected.ipds):
                    self.assertEqual(ipd.url, eipd.url)
                    for x in (ipd.series, ipd.index, ipd.channel):
                        self.assertIsNone(x)

    def test_03_01_errors(self):
        cache = cpmeas.ImageSetCache(self.f)
        image_names = [IMAGE_NAME, ALT_IMAGE_NAME]
        data = [
            cpmeas.ImageSetCache.ImageSetData(
                [],
                [cpmeas.ImageSetCache.ImageData("file:///%s" % uuid.uuid4().hex, None, None, None)
                 for _ in image_names], [(i, uuid.uuid4().hex)])
            for i in range(len(image_names))]
        cache.cache_image_set([(n, cpmeas.IMAGE) for n in image_names],
                              data)
        for reopen in (False, True):
            if reopen:
                cache = cpmeas.ImageSetCache(self.f)
            self.assertTrue(cache.has_cache)
            for i, expected in enumerate(data):
                image_set_data = cache.get_image_set_data(i)
                self.assertEqual(len(image_set_data.errors), 1)
                idx, msg = image_set_data.errors[0]
                self.assertEqual(idx, i)
                self.assertEqual(msg, expected.errors[0][1])

    def test_04_01_close_measurements(self):
        m = cpmeas.Measurements()
        image = cpi.Image(np.zeros((10, 10)))
        m.add(IMAGE_NAME, image)
        m.cache()
        m.close()
