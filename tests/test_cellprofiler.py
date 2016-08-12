'''test_cellprofiler - test the CellProfiler command-line interface
'''

import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib
from cStringIO import StringIO

import dateutil.parser

if hasattr(sys, 'frozen'):
    ARGLIST_START = [sys.executable]
else:
    ARGLIST_START = ["CellProfiler.py", "-b"]


@unittest.skipIf(sys.platform != 'win32', "Skip tests on all but Windows")
class TestCellProfiler(unittest.TestCase):
    def run_cellprofiler(self, *args):
        '''Run CellProfiler with the given arguments list

        returns STDOUT from running it.
        '''
        if hasattr(sys, "frozen"):
            args = [sys.argv[0]] + list(args)
            return subprocess.check_output(args)
        elif sys.platform == 'darwin':
            # hopeless to try and find the right homebrew command
            self.skipTest("Can't start Python properly on OS/X + homebrew")
        else:
            test_dir = os.path.dirname(__file__)
            cellprofiler_dir = os.path.dirname(test_dir)
            root_dir = os.path.dirname(cellprofiler_dir)
            cellprofiler_path = os.path.join(root_dir, "CellProfiler.py")
            self.assertTrue(os.path.isfile(cellprofiler_path))
            args = [sys.executable, cellprofiler_path,
                    "--do-not-build", "--do-not-fetch"] + list(args)
            return subprocess.check_output(args, cwd=root_dir)

    def test_01_01_html(self):
        path = tempfile.mkdtemp()
        try:
            self.run_cellprofiler("--html", "-o", path)
            filenames = os.listdir(path)
            self.assertTrue("index.html" in filenames)
        finally:
            shutil.rmtree(path)

    @unittest.skipIf(hasattr(sys, "frozen"),
                     "Code statistics are not available in frozen-mode")
    def test_01_02_code_statistics(self):
        old_stdout = sys.stdout
        fake_stdout = StringIO(
                self.run_cellprofiler("--code-statistics"))
        fake_stdout.seek(0)
        found_module_stats = False
        found_setting_stats = False
        found_lines_of_code = False
        for line in fake_stdout.readlines():
            if line.startswith("# of built-in modules"):
                found_module_stats = True
            elif line.startswith("# of settings"):
                found_setting_stats = True
            elif line.startswith("# of lines of code"):
                found_lines_of_code = True
        self.assertTrue(found_module_stats)
        self.assertTrue(found_setting_stats)
        self.assertTrue(found_lines_of_code)

    def test_01_03_version(self):
        import cellprofiler.utilities.version as V
        output = self.run_cellprofiler("--version")
        version = dict([tuple(line.strip().split(" "))
                        for line in output.split("\n")
                        if " " in line])
        self.assertEqual(version["CellProfiler"], V.dotted_version)
        self.assertEqual(version["Git"], V.git_hash)
        self.assertEqual(int(version["Version"][:8]),
                         int(V.version_number / 1000000))
        built = dateutil.parser.parse(version["Built"])
        self.assertLessEqual(built.date(), datetime.date.today())

    def test_02_01_run_headless(self):
        output_directory = tempfile.mkdtemp()
        temp_directory = os.path.join(output_directory, "temp")
        os.mkdir(temp_directory)
        try:
            #
            # Run with a .cp file
            #
            fly_pipe = \
                "http://cellprofiler.org/ExampleFlyImages/ExampleFlyURL.cppipe"
            urllib.URLopener().open(fly_pipe).close()
            measurements_file = os.path.join(output_directory, "Measurements.h5")
            done_file = os.path.join(output_directory, "Done.txt")
            self.run_cellprofiler("-c", "-r",
                                  "-o", output_directory,
                                  "-p", fly_pipe,
                                  "-d", done_file,
                                  "-t", temp_directory,
                                  "-f", "1",
                                  "-l", "1",
                                  measurements_file)
            import cellprofiler.preferences as cpprefs
            self.assertTrue(os.path.exists(measurements_file))
            self.assertTrue(os.path.exists(done_file))
            #
            # Re-run using the measurements file.
            #
            m2_file = os.path.join(output_directory, "M2.h5")
            self.run_cellprofiler("-c", "-r",
                                  "-o", output_directory,
                                  "-f", "1",
                                  "-l", "1",
                                  "-p", measurements_file,
                                  m2_file)
            self.assertTrue(os.path.exists(m2_file))
        except IOError, e:
            if e.args[0] != 'http error':
                raise e

            def bad_url(e=e):
                raise e

            unittest.expectedFailure(bad_url)()
        finally:
            shutil.rmtree(output_directory)
