'''test_ijbridge - test ijbridge.py

'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org

__version__ = "$Revision$"

import os
import unittest
import numpy as np
from base64 import b64decode, b64encode
import zlib
import cellprofiler.utilities.jutil as J
from imagej.ijbridge import *

pixels = np.zeros((200,300))
for r in xrange(pixels.shape[0]):
   pixels[r] = float(r) / pixels.shape[0]
pixels[50:150,100:200] = 0.

INVERTED = np.fromstring(zlib.decompress(b64decode(
   r'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'))
   , dtype=np.float32).reshape(pixels.shape)

IJ_CMDS = ['3-3-2 RGB', 'Despeckle', 'Flip Vertically', 'Startup Macros...', 'Open', 'Northeast', 'Brightness/Contrast...', 'Invert LUT', 'Add Selection...', 'Sharpen', 'OR...', 'Concatenate...', 'Next Slice [>]', 'MRI Stack (528K)', 'Square', 'Add Specified Noise...', 'Gel Analyzer Options...', 'Canvas Size...', 'Red', 'Min...', 'Create Mask', 'Rotate... ', 'View 100%', 'Nile Bend (1.9M)', 'Apply LUT', 'Paste Control...', 'Scale...', 'Fill', 'Update ImageJ...', 'Fly Brain (1MB)', 'Reset', 'Enhance Contrast', 'Text...', 'Misc...', 'Window/Level...', 'Plugins...', 'AuPbSn 40 (56K)', 'URL...', 'Size...', 'HSB Stack', 'Benchmark', 'Installation...', 'Proxy Settings...', 'Fit Spline', 'Label...', 'Text Image...', '8-bit', 'Image Calculator...', 'Image Sequence...', 'Conversions...', 'Erode', 'Bandpass Filter...', 'Select Next Lane', 'Set Scale...', 'Delete Slice', 'Print...', 'Virtual Stack...', 'Embryos (42K)', 'Revert', 'Show Overlay', 'Out', 'Find Commands... ', 'Calibrate...', 'Fill Holes', 'Convex Hull', 'Calibration Bar...', 'Enlarge...', 'T1 Head (2.4M, 16-bits)', 'Update Menus', 'Cell Colony (31K)', 'Log', 'RGB Stack', 'Re-plot Lanes', 'Bridge (174K)', 'Label', 'Set Slice...', 'Cut', 'Properties...', 'Histogram', 'Combine...', 'Text File... ', 'Max...', 'Crop', 'New Hyperstack...', 'Macros...', 'RGB Color', 'About This Submenu...', 'Flip Horizontally', 'Clear Outside', 'Add...', 'Open Next', 'Convolve...', 'Split Channels', 'Merge Channels...', 'Edit...', 'Multiply...', 'Select All', 'To Selection', 'Macro... ', 'Install...', 'Z Project...', 'Channels Tool... ', 'Analyze Line Graph', 'Add Slice', 'Bat Cochlea Volume (19K)', 'Reduce Dimensionality...', 'Analyze Particles...', '32-bit', 'Original Scale', 'FD Math...', '16-bit', 'Particles (75K)', 'Gaussian Blur...', 'PNG...', 'Magenta', 'Control Panel...', 'Text Window', 'Fit Ellipse', 'Open...', 'Tile', 'TEM Filter (112K)', 'Macro', 'Install Plugin...', 'Threshold...', 'Blue', 'Raw...', 'Gif...', 'Colors...', 'Make Composite', 'HeLa Cells (1.3M, 48-bit RGB)', 'Wand Tool...', 'Set... ', 'FFT', 'Images to Stack', 'ROI Manager...', 'Southwest', 'Bat Cochlea Renderings (449K)', 'Copy to System', 'Variance...', 'Color Picker...', 'Stack to RGB', 'Minimum...', 'M51 Galaxy (177K, 16-bits)', 'Line Width...', 'Reciprocal', 'Median...', 'Add to Manager ', 'Arrow Tool...', 'System Clipboard', 'Show Info...', 'Ice', 'Specify...', 'Stack From List...', 'Raw Data...', 'Make Inverse', 'Profile Plot Options...', 'Documentation...', 'Reslice [/]...', 'Set...', 'XY Coordinates...', 'Square Root', 'Add Noise', 'LUT...', 'Close-', 'Restore Selection', 'Abs', 'Green', 'Color Balance...', 'ImageJ News...', 'Create Shortcut... ', 'Fluorescent Cells (400K)', 'BMP...', 'Animation Options...', 'Leaf (36K)', 'Selection...', 'Straighten...', 'Line Graph (21K)', 'Neuron (1.6M, 5 channels)', 'Scale Bar...', 'West', 'To ROI Manager', 'South', 'Run...', 'Southeast', 'Reset...', 'Add Image...', 'FFT Options...', 'Undo', 'Macro Functions...', 'Flatten', 'Clear Results', 'Show All', 'Tiff...', 'Convert...', 'Measure...', 'Select None', 'Results...', 'Capture Image', 'Image...', 'Exp', 'Flip Z', 'Plot Profile', 'From ROI Manager', 'Line Width... ', 'Text Window...', 'Save XY Coordinates...', 'Search Website...', 'Remove Outliers...', 'Record...', 'Previous Slice [<]', 'Stack to Hyperstack...', 'Macro...', 'About ImageJ...', 'Custom Filter...', 'Jpeg...', 'North', 'Watershed', 'Dev. Resources...', 'Page Setup...', 'Curve Fitting...', 'Remove...', 'AVI... ', 'Image Sequence... ', 'AND...', 'Ultimate Points', 'Clown (14K)', 'ZIP...', 'Plugin Frame', 'Plugin', 'East', 'Mean...', 'Put Behind [tab]', 'Make Band...', 'Gel (105K)', 'Text Image... ', 'Outline', 'Grays', 'Subtract Background...', 'Confocal Series (2.2MB)', 'Measure', 'Appearance...', 'Lena (68K)', 'Summarize', 'Capture Screen ', 'Measurements...', 'JavaScript', 'Set Measurements...', 'Compiler...', 'Mitosis (26MB, 5D stack)', 'Close', 'Compile and Run...', 'NaN Background', 'Redisplay Power Spectrum', 'Make Binary', 'Surface Plot...', 'Show LUT', 'PGM...', 'Maximum...', 'Show Circular Masks...', 'Gamma...', 'Swap Quadrants', 'Smooth', 'Fonts...', 'Dot Blot (7K)', 'Find Maxima...', 'Salt and Pepper', 'Blobs (25K)', 'Remove Overlay', 'Stop Animation', 'Boats (356K)', 'ImageJ Website...', 'LUT... ', 'Search...', 'Properties... ', 'Save', 'T1 Head Renderings (736K)', 'Hyperstack...', 'Skeletonize', 'Options...', 'Shadows Demo', 'TIFF Virtual Stack...', 'Invert', 'Table...', 'Organ of Corti (2.8M, 4D stack)', 'Internal Clipboard', 'Rotate 90 Degrees Right', 'Plot Lanes', 'Channels Tool...', 'XOR...', 'Hyperstack to Stack', 'Draw Lane Outlines', 'Rename...', 'Input/Output...', 'Fire', 'Rotate...', 'Plugin Filter', 'FITS...', 'Northwest', 'Cardio (768K, RGB DICOM)', 'AVI...', '3D Project...', 'Voronoi', 'Distribution...', 'List Archives...', 'Rotate 90 Degrees Left', 'Distance Map', 'List Shortcuts...', 'Tree Rings (48K)', 'Quit', 'Make Montage...', 'Yellow', 'Copy', 'Point Tool...', 'Color Threshold...', 'Hide Overlay', 'Spectrum', 'Dilate', 'In', 'Draw', 'Unsharp Mask...', '8-bit Color', 'Cascade', 'Inverse FFT', 'Label Peaks', 'Threads...', 'Find Edges', 'Memory & Threads...', 'Select First Lane', 'Divide...', 'Clear', 'Monitor Memory...', 'Plot Z-axis Profile', 'Red/Green', 'Fractal Box Count...', 'Duplicate...', 'Edit LUT...', 'Subtract...', 'Create Selection', 'Convert to Mask', 'CT (420K, 16-bit DICOM)', 'Stack to Images', 'Start Animation [\\]', 'ImageJ Properties...', 'Cyan', 'Orthogonal Views', 'Paste', 'Translate...']


class TestIJBridge(unittest.TestCase):

# may need to update these tests depending on how many channels get
# injected and returned by 2 bridges

   def test_01_01_inject_and_get(self):
      ij1 = get_ij_bridge()
      ij1.inject_image(pixels, 'name ignored')
      im1 = ij1.get_current_image()
      np.testing.assert_array_almost_equal(im1, pixels)

   def test_01_02_invert_macro(self):
      ij1 = get_ij_bridge()
      ij1.inject_image(pixels, 'name ignored')
      ij1.execute_macro('run("Invert");')
      im1 = ij1.get_current_image()
      np.testing.assert_array_almost_equal(im1, INVERTED)

   def test_01_03_add_noise_command(self):
      ij1 = get_ij_bridge()
      ij1.inject_image(pixels, 'name ignored')
      ij1.execute_command('Add Noise')
      im1 = ij1.get_current_image()
      # Add Noise 
      assert (im1 != pixels).any()

   def test_01_04_get_commands(self):
      ij1 = get_ij_bridge()
      cmds = ij1.get_commands()
      assert set(cmds) == set(IJ_CMDS)

##      ww = b64encode(zlib.compress(im1.astype(np.float32).tostring()))
##      ww = np.fromstring(zlib.decompress(b64decode(ww)), dtype=np.float32)
##      ww = ww.reshape(pixels.shape)
##      np.testing.assert_array_almost_equal(im1,ww)
##      f = open("c:\Users\Developer\Desktop\im.txt", "w")
##      f.write(ww)
##      f.close()
      
   def test_02_01_compare_inproc_to_interproc(self):

      #
      # The problem here appears to be that the in-proc bridge injects
      # and returns 3 channels regardless of what is given to it.
      # Need to look into this and adjust inter-proc bridge accordingly.
      #
##      ij1 = in_proc_ij_bridge.getInstance()
##      ij2 = inter_proc_ij_bridge.getInstance()
##      
##      ij1.inject_image(pixels, 'name ignored')
##      ij2.inject_image(pixels, 'name ignored')
##
##      im1 = ij1.get_current_image()
##      im2 = ij2.get_current_image()
##      np.testing.assert_array_almost_equal(im1, im2)
##      
##      ij1.execute_macro('run("Invert");')
##      ij2.execute_macro('run("Invert");')
##
##      im1 = ij1.get_current_image()
##      im2 = ij2.get_current_image()
##      np.testing.assert_array_almost_equal(im1, im2)
##      
##      ij1.execute_command('Add Noise')
##      ij2.execute_command('Add Noise')
##
##      im1 = ij1.get_current_image()
##      im2 = ij2.get_current_image()
##      np.testing.assert_array_almost_equal(im1, im2)
##      
##      cmds1 = ij1.get_commands()
##      cmds2 = ij2.get_commands()
##      assert cmds1 == cmds2
      
