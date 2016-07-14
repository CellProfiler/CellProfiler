import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib

if hasattr(sys, 'frozen'):
    ARGLIST_START = [sys.executable]
else:
    ARGLIST_START = ["CellProfiler.py", "-b"]


class TestCellProfiler(unittest.TestCase):
    def run_cellprofiler(self, *args):
        """Run CellProfiler with the given arguments list

        returns STDOUT from running it.
        """
        if hasattr(sys, "frozen"):
            args = [sys.argv[0]] + list(args)

            return subprocess.check_output(args)
        else:
            test_dir = os.path.dirname(__file__)

            cellprofiler_dir = os.path.dirname(test_dir)

            root_dir = os.path.dirname(cellprofiler_dir)

            cellprofiler_path = os.path.join(root_dir, "CellProfiler.py")

            self.assertTrue(os.path.isfile(cellprofiler_path))

            args = [sys.executable, cellprofiler_path] + list(args)

            return subprocess.check_output(args, cwd=root_dir)

    def test_example(self):
        output_directory = tempfile.mkdtemp()

        temp_directory = os.path.join(output_directory, "temp")

        os.mkdir(temp_directory)

        try:
            #
            # Run with a .cp file
            #
            fly_pipe = "http://cellprofiler.org/ExampleFlyImages/ExampleFlyURL.cppipe"

            urllib.URLopener().open(fly_pipe).close()

            measurements_file = os.path.join(output_directory, "Measurements.h5")

            done_file = os.path.join(output_directory, "Done.txt")

            self.run_cellprofiler(
                "-c",
                "-r",
                "-o", output_directory,
                "-p", fly_pipe,
                "-d", done_file,
                "-t", temp_directory,
                "-f", "1",
                "-l", "1",
                measurements_file
            )

            self.assertTrue(os.path.exists(measurements_file))

            self.assertTrue(os.path.exists(done_file))

            #
            # Re-run using the measurements file.
            #

            m2_file = os.path.join(output_directory, "M2.h5")

            self.run_cellprofiler(
                "-c",
                "-r",
                "-o", output_directory,
                "-f", "1",
                "-l", "1",
                "-p", measurements_file,
                m2_file
            )

            self.assertTrue(os.path.exists(m2_file))
        except IOError, error:
            if error.args[0] != 'http error':
                raise error

            def bad_url(e=error):
                raise e

            unittest.expectedFailure(bad_url)()
        finally:
            shutil.rmtree(output_directory)
