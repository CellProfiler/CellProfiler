class TestCropLabelsAndImage(unittest.TestCase):
    def test_01_01_crop_same(self):
        labels, image = cpo.crop_labels_and_image(np.zeros((10, 20)),
                                                  np.zeros((10, 20)))
        self.assertEqual(tuple(labels.shape), (10, 20))
        self.assertEqual(tuple(image.shape), (10, 20))

    def test_01_02_crop_image(self):
        labels, image = cpo.crop_labels_and_image(np.zeros((10, 20)),
                                                  np.zeros((10, 30)))
        self.assertEqual(tuple(labels.shape), (10, 20))
        self.assertEqual(tuple(image.shape), (10, 20))
        labels, image = cpo.crop_labels_and_image(np.zeros((10, 20)),
                                                  np.zeros((20, 20)))
        self.assertEqual(tuple(labels.shape), (10, 20))
        self.assertEqual(tuple(image.shape), (10, 20))

    def test_01_03_crop_labels(self):
        labels, image = cpo.crop_labels_and_image(np.zeros((10, 30)),
                                                  np.zeros((10, 20)))
        self.assertEqual(tuple(labels.shape), (10, 20))
        self.assertEqual(tuple(image.shape), (10, 20))
        labels, image = cpo.crop_labels_and_image(np.zeros((20, 20)),
                                                  np.zeros((10, 20)))
        self.assertEqual(tuple(labels.shape), (10, 20))
        self.assertEqual(tuple(image.shape), (10, 20))

    def test_01_04_crop_both(self):
        labels, image = cpo.crop_labels_and_image(np.zeros((10, 30)),
                                                  np.zeros((20, 20)))
        self.assertEqual(tuple(labels.shape), (10, 20))
        self.assertEqual(tuple(image.shape), (10, 20))
