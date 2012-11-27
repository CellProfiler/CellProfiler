# -*- Encoding: utf-8 -*-
'''test_ijbridge - test ijbridge.py

'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2012 Broad Institute
# 
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org

__version__ = "$Revision$"

import os
import sys
import unittest
import numpy as np
import cellprofiler.utilities.jutil as J
from imagej.ijbridge import *

PIXELS = np.zeros((20,30))
for r in xrange(PIXELS.shape[0]):
   PIXELS[r] = np.float32(r) / PIXELS.shape[0]
PIXELS[5:15,10:20] = 0.0
PIXELS[10,15] = 1.0
INVERTED = 1.0 - PIXELS

IJ_CMDS = ['3-3-2 RGB', 'Despeckle', 'Flip Vertically', 'Startup Macros...', 'Open', 'Northeast', 'Brightness/Contrast...', 'Invert LUT', 'Add Selection...', 'Sharpen', 'OR...', 'Concatenate...', 'Next Slice [>]', 'MRI Stack (528K)', 'Square', 'Add Specified Noise...', 'Gel Analyzer Options...', 'Canvas Size...', 'Red', 'Min...', 'Create Mask', 'Rotate... ', 'View 100%', 'Nile Bend (1.9M)', 'Apply LUT', 'Paste Control...', 'Scale...', 'Fill', 'Update ImageJ...', 'Fly Brain (1MB)', 'Reset', 'Enhance Contrast', 'Text...', 'Misc...', 'Window/Level...', 'Plugins...', 'AuPbSn 40 (56K)', 'URL...', 'Size...', 'HSB Stack', 'Benchmark', 'Installation...', 'Proxy Settings...', 'Fit Spline', 'Label...', 'Text Image...', '8-bit', 'Image Calculator...', 'Image Sequence...', 'Conversions...', 'Erode', 'Bandpass Filter...', 'Select Next Lane', 'Set Scale...', 'Delete Slice', 'Print...', 'Virtual Stack...', 'Embryos (42K)', 'Revert', 'Show Overlay', 'Out', 'Find Commands... ', 'Calibrate...', 'Fill Holes', 'Convex Hull', 'Calibration Bar...', 'Enlarge...', 'T1 Head (2.4M, 16-bits)', 'Update Menus', 'Cell Colony (31K)', 'Log', 'RGB Stack', 'Re-plot Lanes', 'Bridge (174K)', 'Label', 'Set Slice...', 'Cut', 'Properties...', 'Histogram', 'Combine...', 'Text File... ', 'Max...', 'Crop', 'New Hyperstack...', 'Macros...', 'RGB Color', 'About This Submenu...', 'Flip Horizontally', 'Clear Outside', 'Add...', 'Open Next', 'Convolve...', 'Split Channels', 'Merge Channels...', 'Edit...', 'Multiply...', 'Select All', 'To Selection', 'Macro... ', 'Install...', 'Z Project...', 'Channels Tool... ', 'Analyze Line Graph', 'Add Slice', 'Bat Cochlea Volume (19K)', 'Reduce Dimensionality...', 'Analyze Particles...', '32-bit', 'Original Scale', 'FD Math...', '16-bit', 'Particles (75K)', 'Gaussian Blur...', 'PNG...', 'Magenta', 'Control Panel...', 'Text Window', 'Fit Ellipse', 'Open...', 'Tile', 'TEM Filter (112K)', 'Macro', 'Install Plugin...', 'Threshold...', 'Blue', 'Raw...', 'Gif...', 'Colors...', 'Make Composite', 'HeLa Cells (1.3M, 48-bit RGB)', 'Wand Tool...', 'Set... ', 'FFT', 'Images to Stack', 'ROI Manager...', 'Southwest', 'Bat Cochlea Renderings (449K)', 'Copy to System', 'Variance...', 'Color Picker...', 'Stack to RGB', 'Minimum...', 'M51 Galaxy (177K, 16-bits)', 'Line Width...', 'Reciprocal', 'Median...', 'Add to Manager ', 'Arrow Tool...', 'System Clipboard', 'Show Info...', 'Ice', 'Specify...', 'Stack From List...', 'Raw Data...', 'Make Inverse', 'Profile Plot Options...', 'Documentation...', 'Reslice [/]...', 'Set...', 'XY Coordinates...', 'Square Root', 'Add Noise', 'LUT...', 'Close-', 'Restore Selection', 'Abs', 'Green', 'Color Balance...', 'ImageJ News...', 'Create Shortcut... ', 'Fluorescent Cells (400K)', 'BMP...', 'Animation Options...', 'Leaf (36K)', 'Selection...', 'Straighten...', 'Line Graph (21K)', 'Neuron (1.6M, 5 channels)', 'Scale Bar...', 'West', 'To ROI Manager', 'South', 'Run...', 'Southeast', 'Reset...', 'Add Image...', 'FFT Options...', 'Undo', 'Macro Functions...', 'Flatten', 'Clear Results', 'Show All', 'Tiff...', 'Convert...', 'Measure...', 'Select None', 'Results...', 'Capture Image', 'Image...', 'Exp', 'Flip Z', 'Plot Profile', 'From ROI Manager', 'Line Width... ', 'Text Window...', 'Save XY Coordinates...', 'Search Website...', 'Remove Outliers...', 'Record...', 'Previous Slice [<]', 'Stack to Hyperstack...', 'Macro...', 'About ImageJ...', 'Custom Filter...', 'Jpeg...', 'North', 'Watershed', 'Dev. Resources...', 'Page Setup...', 'Curve Fitting...', 'Remove...', 'AVI... ', 'Image Sequence... ', 'AND...', 'Ultimate Points', 'Clown (14K)', 'ZIP...', 'Plugin Frame', 'Plugin', 'East', 'Mean...', 'Put Behind [tab]', 'Make Band...', 'Gel (105K)', 'Text Image... ', 'Outline', 'Grays', 'Subtract Background...', 'Confocal Series (2.2MB)', 'Measure', 'Appearance...', 'Lena (68K)', 'Summarize', 'Capture Screen ', 'Measurements...', 'JavaScript', 'Set Measurements...', 'Compiler...', 'Mitosis (26MB, 5D stack)', 'Close', 'Compile and Run...', 'NaN Background', 'Redisplay Power Spectrum', 'Make Binary', 'Surface Plot...', 'Show LUT', 'PGM...', 'Maximum...', 'Show Circular Masks...', 'Gamma...', 'Swap Quadrants', 'Smooth', 'Fonts...', 'Dot Blot (7K)', 'Find Maxima...', 'Salt and Pepper', 'Blobs (25K)', 'Remove Overlay', 'Stop Animation', 'Boats (356K)', 'ImageJ Website...', 'LUT... ', 'Search...', 'Properties... ', 'Save', 'T1 Head Renderings (736K)', 'Hyperstack...', 'Skeletonize', 'Options...', 'Shadows Demo', 'TIFF Virtual Stack...', 'Invert', 'Table...', 'Organ of Corti (2.8M, 4D stack)', 'Internal Clipboard', 'Rotate 90 Degrees Right', 'Plot Lanes', 'Channels Tool...', 'XOR...', 'Hyperstack to Stack', 'Draw Lane Outlines', 'Rename...', 'Input/Output...', 'Fire', 'Rotate...', 'Plugin Filter', 'FITS...', 'Northwest', 'Cardio (768K, RGB DICOM)', 'AVI...', '3D Project...', 'Voronoi', 'Distribution...', 'List Archives...', 'Rotate 90 Degrees Left', 'Distance Map', 'List Shortcuts...', 'Tree Rings (48K)', 'Quit', 'Make Montage...', 'Yellow', 'Copy', 'Point Tool...', 'Color Threshold...', 'Hide Overlay', 'Spectrum', 'Dilate', 'In', 'Draw', 'Unsharp Mask...', '8-bit Color', 'Cascade', 'Inverse FFT', 'Label Peaks', 'Threads...', 'Find Edges', 'Memory & Threads...', 'Select First Lane', 'Divide...', 'Clear', 'Monitor Memory...', 'Plot Z-axis Profile', 'Red/Green', 'Fractal Box Count...', 'Duplicate...', 'Edit LUT...', 'Subtract...', 'Create Selection', 'Convert to Mask', 'CT (420K, 16-bit DICOM)', 'Stack to Images', 'Start Animation [\\]', 'ImageJ Properties...', 'Cyan', 'Orthogonal Views', 'Paste', 'Translate...']


class TestIJBridge(unittest.TestCase):
   '''Tests the basic functions that we want to be able to perform in ImageJ
   through the 2 kinds of bridges: in-proc (via JNI) and inter-proc (via TCP)
   '''
   
   def test_00_00_get_ij_bridge(self):
      ijb = get_ij_bridge()
      if sys.platform == 'darwin':
         assert ijb == inter_proc_ij_bridge.getInstance()
      else:
         assert ijb == in_proc_ij_bridge.getInstance()

   #
   # in-proc (JNI) tests
   #    Note: these will not run on OS X due to competition between WX and AWT
   #
   
   def test_01_01_jni_inject_and_get(self):
      ijb = in_proc_ij_bridge.getInstance()
      ijb.inject_image(PIXELS, 'name ignored')
      im = ijb.get_current_image()
      np.testing.assert_array_almost_equal(im, PIXELS)

   def test_01_02_jni_invert_macro(self):
      ijb = in_proc_ij_bridge.getInstance()
      ijb.inject_image(PIXELS, 'name ignored')
      ijb.execute_macro('run("Invert");')
      im = ijb.get_current_image()
      assert im.min() >= 0.0
      assert im.max() <= 1.0
      np.testing.assert_array_almost_equal(im, INVERTED)

   def test_01_03_jni_command(self):
      ijb = in_proc_ij_bridge.getInstance()
      ijb.inject_image(PIXELS, 'name ignored')
      ijb.execute_command('Smooth')
      im = ijb.get_current_image()
      assert (im != PIXELS).any()
      assert im.min() >= 0.0
      assert im.max() <= 1.0

   def test_01_04_jni_get_commands(self):
      ijb =  in_proc_ij_bridge.getInstance()
      cmds = ijb.get_commands()
      assert set(cmds).issuperset(set(IJ_CMDS))
