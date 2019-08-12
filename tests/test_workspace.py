"""test_workspace.py - test the workspace
"""
import logging
import os
import tempfile
import unittest

import h5py

import cellprofiler.measurement
import cellprofiler.pipeline
import cellprofiler.workspace
from cellprofiler.utilities.hdf5_dict import FILE_LIST_GROUP, TOP_LEVEL_GROUP_NAME

logger = logging.getLogger(__name__)


class TestWorkspace(unittest.TestCase):
    def setUp(self):
        self.workspace_files = []

    def tearDown(self):
        for path in self.workspace_files:
            try:
                os.remove(path)
            except:
                logger.warn("Failed to close file %s" % path,
                            exc_info=1)

    def make_workspace_file(self):
        """Make a very basic workspace file"""
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.init_modules()
        m = cellprofiler.measurement.Measurements()
        workspace = cellprofiler.workspace.Workspace(pipeline, None, m, None, m, None)
        fd, path = tempfile.mkstemp(".cpproj")
        file_list = workspace.get_file_list()
        file_list.add_files_to_filelist([os.path.join(os.path.dirname(__file__), 'resources/01_POS002_D.TIF')])
        workspace.save(path)
        self.workspace_files.append(path)
        os.close(fd)
        return path

    def test_01_01_is_workspace_file(self):
        path = self.make_workspace_file()
        self.assertTrue(cellprofiler.workspace.is_workspace_file(path))

    def test_01_02_is_not_workspace_file(self):
        self.assertFalse(cellprofiler.workspace.is_workspace_file(__file__))
        for group in TOP_LEVEL_GROUP_NAME, FILE_LIST_GROUP:
            path = self.make_workspace_file()
            h5file = h5py.File(path)
            del h5file[group]
            h5file.close()
            self.assertFalse(cellprofiler.workspace.is_workspace_file(path))

    def test_01_03_file_handle_closed(self):
        # regression test of issue #1326
        path = self.make_workspace_file()
        self.assertTrue(cellprofiler.workspace.is_workspace_file(path))
        os.remove(path)
        self.workspace_files.remove(path)
        self.assertFalse(os.path.isfile(path))
