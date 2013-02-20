'''test_cellprofiler - test the CellProfiler command-line interface

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import os
import unittest
from cStringIO import StringIO
import shutil
import sys
import tempfile
from cellprofiler.modules.tests import example_images_directory

import CellProfiler

class TestCellProfiler(unittest.TestCase):
    def test_01_01_html(self):
        path = tempfile.mkdtemp()
        try:
            CellProfiler.main(["foo", "-b", "--html","-o",path])
            filenames = os.listdir(path)
            self.assertTrue("index.html" in filenames)
        finally:
            shutil.rmtree(path)
            
    def test_01_02_code_statistics(self):
        old_stdout = sys.stdout
        fake_stdout = StringIO()
        sys.stdout = fake_stdout
        try:
            CellProfiler.main(["foo", "-b", "--code-statistics"])
        finally:
            sys.stdout = old_stdout
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
        
    def test_02_01_run_headless(self):
        output_directory = tempfile.mkdtemp()
        try:
            #
            # Run with a .cp file
            #
            input_directory = os.path.join(
                example_images_directory(),
                "ExampleHT29")
            pipeline_file = os.path.join(input_directory, "ExampleHT29.cp")
            measurements_file = os.path.join(output_directory, "Measurements.h5")
            done_file = os.path.join(output_directory, "Done.txt")
            args = [ "CellProfiler.py",
                "-c", "-r", "-b", 
                "-i", input_directory,
                "-o", output_directory,
                "-p", pipeline_file,
                "-d", done_file,
                measurements_file]
            CellProfiler.main(args)
            self.assertTrue(os.path.exists(measurements_file))
            self.assertTrue(os.path.exists(done_file))
            #
            # Re-run using the measurements file.
            #
            m2_file = os.path.join(output_directory, "M2.h5")
            args = [ "CellProfiler.py",
                     "-c", "-r", "-b",
                     "-i", input_directory,
                     "-o", output_directory,
                     "-p", measurements_file,
                     m2_file]
            CellProfiler.main(args)
            self.assertTrue(os.path.exists(m2_file))
        finally:
            shutil.rmtree(output_directory)
            