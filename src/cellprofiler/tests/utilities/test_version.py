# https://github.com/CellProfiler/CellProfiler/issues/2226
#
# '''test_version - test the version number and other information
# '''
#
# import unittest
#
# import cellprofiler.utilities.version
#
#
# class TestVersion(unittest.TestCase):
#     def test_01_01_test_version_number(self):
#         '''Check that the version number is well-formed.'''
#         version_number = cellprofiler.utilities.version.version_number
#         # This variable was added in early 2012.
#         # This test will need updating in 2025.  I apologize to the future.
#         assert 20120000000000 < version_number < 20250000000000
#
#     def test_01_02_test_dotted_version(self):
#         '''Check that the dotted version is well-formed.'''
#         dotted_version = cellprofiler.utilities.version.dotted_version
#         # for now, assume Int.Int.Int.  We may need to relax update to be other
#         # than Int, at some point.
#         major, minor, update = [int(v) for v in dotted_version.split('.')]
#         assert major >= 2  # Otherwise, we've regressed to Matlab.
