"""test_cellprofiler - test the CellProfiler command-line interface
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from tests.modules import get_test_resources_directory

if hasattr(sys, "frozen"):
    ARGLIST_START = [sys.executable]
else:
    ARGLIST_START = ["-m", "cellprofiler", "-b"]


@unittest.skipIf(sys.platform != "win32", "Skip tests on all but Windows")
class TestCellProfiler(unittest.TestCase):
    def run_cellprofiler(self, *args):
        """Run CellProfiler with the given arguments list

        returns STDOUT from running it.
        """
        if hasattr(sys, "frozen"):
            args = [sys.argv[0]] + list(args)
            return subprocess.check_output(args)
        elif sys.platform == "darwin":
            # hopeless to try and find the right homebrew command
            self.skipTest("Can't start Python properly on OS/X + homebrew")
        else:
            test_dir = os.path.dirname(__file__)
            cellprofiler_dir = os.path.dirname(test_dir)
            root_dir = os.path.dirname(cellprofiler_dir)
            args = [
                sys.executable,
                "-m",
                "cellprofiler",
                "--do-not-fetch",
            ] + list(args)
            return subprocess.check_output(args, cwd=root_dir)

    def test_get_version(self):
        import cellprofiler
        output = self.run_cellprofiler("--version")
        version = output.decode().rstrip()
        assert version == cellprofiler.__version__

    def test_run_headless(self):
        output_directory = tempfile.mkdtemp()
        temp_directory = os.path.join(output_directory, "temp")
        os.mkdir(temp_directory)
        try:
            #
            # Run with a .cp file
            #
            fly_pipe = get_test_resources_directory("../ExampleFlyURL.cppipe")
            measurements_file = os.path.join(output_directory, "Measurements.h5")
            done_file = os.path.join(output_directory, "Done.txt")
            self.run_cellprofiler(
                "-c",
                "-r",
                "-o",
                output_directory,
                "-p",
                fly_pipe,
                "-d",
                done_file,
                "-t",
                temp_directory,
                "-f",
                "1",
                "-l",
                "1",
                measurements_file,
            )
            import cellprofiler_core.preferences as cpprefs

            self.assertTrue(os.path.exists(measurements_file))
            self.assertTrue(os.path.exists(done_file))
            #
            # Re-run using the measurements file.
            #
            m2_file = os.path.join(output_directory, "M2.h5")
            self.run_cellprofiler(
                "-c",
                "-r",
                "-o",
                output_directory,
                "-f",
                "1",
                "-l",
                "1",
                "-p",
                measurements_file,
                m2_file,
            )
            self.assertTrue(os.path.exists(m2_file))
        except IOError as e:
            if e.args[0] != "http error":
                raise e

            def bad_url(e=e):
                raise e

            unittest.expectedFailure(bad_url)()
        finally:
            shutil.rmtree(output_directory)
