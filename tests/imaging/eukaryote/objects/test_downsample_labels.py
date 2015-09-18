class TestDownsampleLabels(unittest.TestCase):
    def test_01_01_downsample_127(self):
        i, j = np.mgrid[0:16, 0:8]
        labels = (i * 8 + j).astype(int)
        result = cpo.downsample_labels(labels)
        self.assertEqual(result.dtype, np.dtype(np.int8))
        self.assertTrue(np.all(result == labels))

    def test_01_02_downsample_128(self):
        i, j = np.mgrid[0:16, 0:8]
        labels = (i * 8 + j).astype(int) + 1
        result = cpo.downsample_labels(labels)
        self.assertEqual(result.dtype, np.dtype(np.int16))
        self.assertTrue(np.all(result == labels))

    def test_01_03_downsample_32767(self):
        i, j = np.mgrid[0:256, 0:128]
        labels = (i * 128 + j).astype(int)
        result = cpo.downsample_labels(labels)
        self.assertEqual(result.dtype, np.dtype(np.int16))
        self.assertTrue(np.all(result == labels))

    def test_01_04_downsample_32768(self):
        i, j = np.mgrid[0:256, 0:128]
        labels = (i * 128 + j).astype(int) + 1
        result = cpo.downsample_labels(labels)
        self.assertEqual(result.dtype, np.dtype(np.int32))
        self.assertTrue(np.all(result == labels))
