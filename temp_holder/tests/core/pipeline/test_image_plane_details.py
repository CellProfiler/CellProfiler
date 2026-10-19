import unittest

import cellprofiler_core.pipeline


class TestImagePlaneDetails(unittest.TestCase):
    def get_ipd(
        self, url="http://nucleus.org", series=0, index=0, channel=0, metadata={}
    ):
        d = cellprofiler_core.pipeline.J.make_map(**metadata)
        jipd = cellprofiler_core.pipeline.J.run_script(
            """
            var uri = new java.net.URI(url);
            var f = new Packages.org.cellprofiler.imageset.ImageFile(uri);
            var fd = new Packages.org.cellprofiler.imageset.ImageFileDetails(f);
            var s = new Packages.org.cellprofiler.imageset.ImageSeries(f, series);
            var sd = new Packages.org.cellprofiler.imageset.ImageSeriesDetails(s, fd);
            var p = new Packages.org.cellprofiler.imageset.ImagePlane(s, index, channel);
            var ipd = new Packages.org.cellprofiler.imageset.ImagePlaneDetails(p, sd);
            ipd.putAll(d);
            ipd;
            """,
            dict(url=url, series=series, index=index, channel=channel, d=d),
        )
        return cellprofiler_core.pipeline.ImagePlane(jipd)

        # def test_01_01_init(self):
        #     self.get_ipd();

        # def test_02_01_path_url(self):
        #     url = "http://google.com"
        #     ipd = self.get_ipd(url=url)
        #     self.assertEquals(ipd.path, url)

        # def test_02_02_path_file(self):
        #     path = "file:" + cpp.urllib.pathname2url(__file__)
        #     ipd = self.get_ipd(url=path)
        #     if sys.platform == 'win32':
        #         self.assertEquals(ipd.path.lower(), __file__.lower())
        #     else:
        #         self.assertEquals(ipd.path, __file__)

        # def test_03_01_url(self):
        #     url = "http://google.com"
        #     ipd = self.get_ipd(url=url)
        #     self.assertEquals(ipd.url, url)

        # def test_04_01_series(self):
        #     ipd = self.get_ipd(series = 4)
        #     self.assertEquals(ipd.series, 4)

        # def test_05_01_index(self):
        #     ipd = self.get_ipd(index = 2)
        #     self.assertEquals(ipd.index, 2)

        # def test_06_01_channel(self):
        #     ipd = self.get_ipd(channel=3)
        #     self.assertEquals(ipd.channel, 3)

        # def test_07_01_metadata(self):
        #     ipd = self.get_ipd(metadata = dict(foo="Bar", baz="Blech"))
        #     self.assertEquals(ipd.metadata["foo"], "Bar")
        #     self.assertEquals(ipd.metadata["baz"], "Blech")

        # def test_08_01_save_pipeline_notes(self):
        #     fd = six.moves.StringIO()
        #     pipeline = cpp.Pipeline()
        #     module = ATestModule()
        #     module.set_module_num(1)
        #     module.notes.append("Hello")
        #     module.notes.append("World")
        #     pipeline.add_module(module)
        #     module = ATestModule()
        #     module.set_module_num(2)
        #     module.enabled = False
        #     pipeline.add_module(module)
        #     expected = "\n".join([
        #         "[   1] [ATestModule]",
        #         "  Hello",
        #         "  World",
        #         "",
        #         "[   2] [ATestModule] (disabled)",
        #         ""])
        #
        #     pipeline.save_pipeline_notes(fd)
        #     self.assertEqual(fd.getvalue(), expected)
