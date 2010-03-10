"""test_cpfigure- test cpfigure functionality

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import numpy as np
import time
import unittest

import wx
import matplotlib
import cellprofiler.gui.cpfigure as cpfig


app = wx.PySimpleApp()

class TestCPFigure(unittest.TestCase):
    
    def test_01_01_imshow_raw(self):
        '''Make sure the image drawn by imshow matches the input image.'''
        image = np.zeros((100, 100))
        for y in range(image.shape[0]):
            image[y,:] = y / 200.0
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1))
        ax = my_frame.subplot_imshow(0, 0, image, normalize=False)
        
        assert (((ax.get_array()-image) < 0.000001).all()), 'Monochrome input image did not match subplot image.'
        
    def test_01_02_imshow_raw_rgb(self):
        '''Make sure the image drawn by imshow matches the input RGB image.'''
        image = np.zeros((100, 100, 3))
        for y in range(image.shape[0]):
            image[y,:,:] = y / 200.0
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1))
        ax = my_frame.subplot_imshow(0, 0, image, normalize=False)
        
        assert (((ax.get_array()-image) < 0.000001).all()), 'RGB input image did not match subplot image.'
        
    def test_01_03_imshow_normalized(self):
        '''Make sure the image drawn by imshow is normalized.'''
        image = np.zeros((100, 100))
        for y in range(image.shape[0]):
            image[y,:] = y / 200.0
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1))
        ax = my_frame.subplot_imshow(0, 0, image, normalize=True)
        
        normed = ((image - np.min(image)) / np.max(image))
        assert ((ax.get_array() - normed) < 0.000001).all(), 'Monochrome subplot image was not normalized.'

    def test_01_04_imshow_normalized_rgb(self):
        '''Make sure the RGB image drawn by imshow is normalized.'''
        image = np.zeros((100, 100, 3))
        for y in range(image.shape[0]):
            image[y,:,:] = y / 200.0
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1))
        ax = my_frame.subplot_imshow(0, 0, image, normalize=True)
        
        normed = ((image - np.min(image)) / np.max(image))
        assert ((ax.get_array() - normed) < 0.000001).all(), 'RGB subplot image was not normalized.'

    def test_01_05_imshow_log_normalized(self):
        '''Make sure the image drawn by imshow is log normalized.'''
        image = np.zeros((100, 100))
        for y in range(image.shape[0]):
            image[y,:] = y / 200.0
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1))
        ax = my_frame.subplot_imshow(0, 0, image, normalize='log')
        
        (min, max) = (image[image > 0].min(), image.max())
        normed = (np.log(image.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))
        assert ((ax.get_array() - normed) < 0.000001).all(), 'Monochrome subplot image was not log normalized.'

    def test_01_06_imshow_log_normalized_rgb(self):
        '''Make sure the RGB image drawn by imshow is log normalized.'''
        image = np.zeros((100, 100, 3))
        for y in range(image.shape[0]):
            image[y,:] = y / 200.0
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1))
        ax = my_frame.subplot_imshow(0, 0, image, normalize='log')
        
        (min, max) = (image[image > 0].min(), image.max())
        normed = (np.log(image.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))
        assert ((ax.get_array() - normed) < 0.000001).all(), 'RGB subplot image was not log normalized.'

    def test_02_01_show_pixel_data(self):
        '''Make sure the values reported by show_pixel_data are the raw image
        values for grayscale images.'''
        image = np.zeros((100, 100))
        for y in range(image.shape[0]):
            image[y,:] = y / 200.0
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1))
        ax = my_frame.subplot_imshow(0, 0, image, normalize=True)
        
        evt = matplotlib.backend_bases.MouseEvent('motion_notify_event',
                                                  ax.figure.canvas, 
                                                  x=0, y=10)
        evt.xdata = 0
        evt.ydata = 10
        evt.inaxes = my_frame.subplot(0, 0)
        my_frame.on_mouse_move_show_pixel_data(evt, 0, 0, 0, 0)
        expected = "Intensity: %.4f"%(evt.ydata / 200.0)
        assert expected in [str(f) for f in my_frame.status_bar.GetFields()], 'Did not find "%s" in StatusBar fields'%(expected)

    def test_02_02_show_pixel_data_rgb(self):
        '''Make sure the values reported by show_pixel_data are the raw image
        values for RGB images.'''
        image = np.zeros((100, 100, 3))
        for y in range(image.shape[0]):
            image[y,:,:] = y / 200.0
        image[:,:,1] = image[:,:,1] / 2.
        image[:,:,2] = image[:,:,2] / 4.
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1))
        ax = my_frame.subplot_imshow(0, 0, image, normalize=True)
        
        evt = matplotlib.backend_bases.MouseEvent('motion_notify_event',
                                                  ax.figure.canvas, 
                                                  x=0, y=10)
        evt.xdata = 0
        evt.ydata = 10
        evt.inaxes = my_frame.subplot(0, 0)
        my_frame.on_mouse_move_show_pixel_data(evt, 0, 0, 0, 0)
        expected = ["Red: %.4f"%(evt.ydata / 200.0),
                    "Green: %.4f"%(evt.ydata / 200.0 / 2.),
                    "Blue: %.4f"%(evt.ydata / 200.0 / 4.)]
        for field in expected:
            assert field in [str(f) for f in my_frame.status_bar.GetFields()], 'Did not find "%s" in StatusBar fields'%(field)
             
        
        
app.MainLoop()
