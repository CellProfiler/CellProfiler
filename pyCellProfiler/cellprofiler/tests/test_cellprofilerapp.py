import unittest
import cellprofiler.cellprofilerapp

class test_cellprofilerapp(unittest.TestCase):
    def test_00_00_Init(self):
        app = cellprofiler.cellprofilerapp.CellProfilerApp()
        app.Exit()

    