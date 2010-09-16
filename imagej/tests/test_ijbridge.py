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

import cellprofiler.utilities.jutil as J
from imagej.ijbridge import *


class TestIJBridge(unittest.TestCase):

##   def test_01_01_in_proc_cmds(self):
##      ij1 = in_proc_ij_bridge()
##      pixels = np.random.standard_normal((400,600))
##      pixels[pixels<0] = 0
##      pixels /= pixels.max()
##      pixels[100:300,200:400] = 0
##      
##      ij1.inject_image(pixels, 'name ignored')
##      im1 = ij1.get_current_image()
##      np.testing.assert_array_almost_equal(im1, pixels)
##      
##      ij1.execute_macro('run("Invert");')
##      im1 = ij1.get_current_image()
###      np.testing.assert_array_almost_equal(im1, pixels)
##
##      ij1.execute_command('Add Noise')
##
##      im1 = ij1.get_current_image()
      
   def test_02_01_load_imageplus(self):
      ij1 = in_proc_ij_bridge()
      ij2 = inter_proc_ij_bridge()
      pixels = np.random.standard_normal((400,600))
      pixels[pixels<0] = 0
      pixels /= pixels.max()
      pixels[100:300,200:400] = 0
      
      ij1.inject_image(pixels, 'name ignored')
      ij2.inject_image(pixels, 'name ignored')

      im1 = ij1.get_current_image()
      im2 = ij2.get_current_image()
      print 'im1',im1[0:5,0:5]
      print 'im2',im2[0:5,0:5]
      assert im1 == im2
      
      ij1.execute_macro('run("Invert");')
      ij2.execute_macro('run("Invert");')

      im1 = ij1.get_current_image()
      im2 = ij2.get_current_image()
      assert im1 == im2

      ij1.execute_command('Add Noise')
      ij2.execute_command('Add Noise')

      im1 = ij1.get_current_image()
      im2 = ij2.get_current_image()
      assert im1 == im2

      cmds1 = ij1.get_commands()
      cmds2 = ij2.get_commands()
      assert cmds1 == cmds2
