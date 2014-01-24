"""test_cpfigure- test cpfigure functionality

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import numpy as np
import time
import unittest

import wx
import matplotlib
from cellprofiler.preferences import set_headless
set_headless()
import cellprofiler.gui.cpfigure as cpfig



class TestCPFigure(unittest.TestCase):
    def setUp(self):
        self.app = wx.GetApp()
        if self.app is None:
            self.app = wx.PySimpleApp(False)
        self.frame = wx.Frame(None, title="Hello, world")
        self.frame.Show()

    def test_01_01_imshow_raw(self):
        '''Make sure the image drawn by imshow matches the input image.'''
        image = np.zeros((100, 100))
        for y in range(image.shape[0]):
            image[y,:] = y / 200.0
        my_frame = cpfig.create_or_find(self.frame, -1, subplots=(1,1), 
                                        name="test_01_01_imshow_raw")
        ax = my_frame.subplot_imshow(0, 0, image, normalize=False)
        # assert (((ax.get_array()-image) < 0.000001).all()), 'Monochrome input image did not match subplot image.'
        my_frame.Destroy()
        
    def test_01_02_imshow_raw_rgb(self):
        '''Make sure the image drawn by imshow matches the input RGB image.'''
        image = np.zeros((100, 100, 3))
        for y in range(image.shape[0]):
            image[y,:,:] = y / 200.0
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1),
                                        name = "test_01_02_imshow_raw_rgb")
        ax = my_frame.subplot_imshow(0, 0, image, normalize=False)
        # shown_im = ax.get_array().astype(float) / 255.0
        # np.testing.assert_almost_equal(shown_im, image, decimal=2)
        my_frame.Destroy()
        
    def test_01_03_imshow_normalized(self):
        '''Make sure the image drawn by imshow is normalized.'''
        image = np.zeros((100, 100))
        for y in range(image.shape[0]):
            image[y,:] = y / 200.0
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1),
                                        name = "test_01_03_imshow_normalized")
        ax = my_frame.subplot_imshow(0, 0, image, normalize=True)
        
        normed = ((image - np.min(image)) / np.max(image))
        # np.testing.assert_almost_equal(ax.get_array(), normed, decimal=2)
        my_frame.Destroy()

    def test_01_04_imshow_normalized_rgb(self):
        '''Make sure the RGB image drawn by imshow is normalized.'''
        image = np.zeros((100, 100, 3))
        for y in range(image.shape[0]):
            image[y,:,:] = y / 200.0
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1),
                                        name="test_01_04_imshow_normalized_rgb")
        ax = my_frame.subplot_imshow(0, 0, image, normalize=True)
        
        normed = ((image - np.min(image)) / np.max(image))
        # shown_im = ax.get_array().astype(float) / 255.0
        # np.testing.assert_almost_equal(normed, shown_im, decimal=2)
        my_frame.Destroy()

    def test_01_05_imshow_log_normalized(self):
        '''Make sure the image drawn by imshow is log normalized.'''
        image = np.zeros((100, 100))
        for y in range(image.shape[0]):
            image[y,:] = y / 200.0
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1),
                                        name = "test_01_05_imshow_log_normalized")
        ax = my_frame.subplot_imshow(0, 0, image, normalize='log')
        
        (min, max) = (image[image > 0].min(), image.max())
        normed = (np.log(image.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))
        # np.testing.assert_almost_equal(normed, ax.get_array(), decimal=2)
        my_frame.Destroy()

    def test_01_06_imshow_log_normalized_rgb(self):
        '''Make sure the RGB image drawn by imshow is log normalized.'''
        image = np.zeros((100, 100, 3))
        for y in range(image.shape[0]):
            image[y,:] = y / 200.0
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1),
                                        name="test_01_06_imshow_log_normalized_rgb")
        ax = my_frame.subplot_imshow(0, 0, image, normalize='log')
        
        (min, max) = (image[image > 0].min(), image.max())
        normed = (np.log(image.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))
        # shown_im = ax.get_array().astype(float) / 255.0
        # np.testing.assert_almost_equal(normed, shown_im, decimal=2)
        my_frame.Destroy()

    def test_02_01_show_pixel_data(self):
        '''Make sure the values reported by show_pixel_data are the raw image
        values for grayscale images.'''
        image = np.zeros((100, 100))
        for y in range(image.shape[0]):
            image[y,:] = y / 200.0
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1),
                                        name="test_02_01_show_pixel_data")
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
        my_frame.Destroy()

    def test_02_02_show_pixel_data_rgb(self):
        '''Make sure the values reported by show_pixel_data are the raw image
        values for RGB images.'''
        image = np.zeros((100, 100, 3))
        for y in range(image.shape[0]):
            image[y,:,:] = y / 200.0
        image[:,:,1] = image[:,:,1] / 2.
        image[:,:,2] = image[:,:,2] / 4.
        my_frame = cpfig.create_or_find(None, -1, subplots=(1,1),
                                        name="test_02_02_show_pixel_data_rgb")
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
        my_frame.Destroy()
             
    def test_03_01_menu_order(self):
        '''Make sure that the subplots submenus are presented in the right order
        no matter what order they are drawn in.
        Also tests that the order is not affected after calling clf()'''
        f = cpfig.create_or_find(None, -1, subplots=(4,2),
                                 name="test_03_01_menu_order")

        img = np.random.uniform(.5, .6, size=(5, 5, 3))
        
        f.subplot_histogram(0, 0, [1,1,1,2], 2, title="hist")
        f.subplot_imshow(1, 0, img, "rgb1")
        f.subplot_histogram(2, 0, [1,1,1,2], 2, title="hist")
        f.subplot_imshow(3, 0, img, "rgb2")

        f.subplot_imshow(0, 1, img, "rgb3")
        f.subplot_imshow(1, 1, img, "rgb4")
        f.subplot_imshow(2, 1, img, "rgb5")
        f.subplot_histogram(3, 1, [1,1,1,2], 2, title="hist")
    
        def test_01_01_imshow_raw(self):
            '''Make sure the image drawn by imshow matches the input image.'''
            image = np.zeros((100, 100))
            for y in range(image.shape[0]):
                image[y,:] = y / 200.0
            my_frame = cpfig.create_or_find(self.frame, -1, subplots=(1,1), 
                                            name="test_01_01_imshow_raw")
            ax = my_frame.subplot_imshow(0, 0, image, normalize=False)
            # assert (((ax.get_array()-image) < 0.000001).all()), 'Monochrome input image did not match subplot image.'
            my_frame.Destroy()
            
        f.clf()

        assert len(f.menu_subplots.MenuItems) == 0, 'Subplot menus should be empty after clf().'
        
        f.subplot_histogram(3, 1, [1,1,1,2], 2, title="hist")
        f.subplot_imshow(2, 1, img, "rgb5")
        f.subplot_imshow(0, 1, img, "rgb3")
        f.subplot_imshow(1, 1, img, "rgb4")
        f.subplot_histogram(2, 0, [1,1,1,2], 2, title="hist")
        f.subplot_imshow(1, 0, img, "rgb1")
        f.subplot_imshow(3, 0, img, "rgb2")
        f.subplot_histogram(0, 0, [1,1,1,2], 2, title="hist")
    
        for i, item in enumerate(f.menu_subplots.MenuItems):
            assert item.Label == 'rgb%s'%(i+1)
                        
        f.Destroy()

    def test_03_02_menu_order2(self):
        '''Make sure that the subplots submenus are presented in the right order
        after they are redrawn as a result of menu handlers 
        (e.g. change_contrast)'''
        f = cpfig.create_or_find(None, -1, subplots=(2,2),
                                 name="test_03_02_menu_order2")

        img = np.random.uniform(.5, .6, size=(5, 5, 3))
        
        f.subplot_histogram(0, 0, [1,1,1,2], 2, title="hist")
        f.subplot_imshow(1, 0, img, "rgb1")
        f.subplot_imshow(0, 1, img, "rgb2")
        f.subplot_imshow(1, 1, img, "rgb3")
    
        for i, item in enumerate(f.menu_subplots.MenuItems):
            assert item.Label == 'rgb%s'%(i+1)
            
        menu = f.get_imshow_menu((1,0))
        for item in menu.MenuItems:
            if item.Label == 'Image contrast':
                for item in item.SubMenu.MenuItems:
                    if item.Label == 'Raw':
                        event = wx.PyCommandEvent(wx.EVT_MENU.typeId, item.Id)
                        f.GetEventHandler().ProcessEvent(event)
                        self.app.ProcessPendingEvents()
    
        for i, item in enumerate(f.menu_subplots.MenuItems):
            assert item.Label == 'rgb%s'%(i+1)
            
        menu = f.get_imshow_menu((1,1))
        for item in menu.MenuItems:
            if item.Label == 'Image contrast':
                for item in item.SubMenu.MenuItems:
                    if item.Label == 'Log normalized':
                        event = wx.PyCommandEvent(wx.EVT_MENU.typeId, item.Id)
                        f.GetEventHandler().ProcessEvent(event)
                        self.app.ProcessPendingEvents()
    
        for i, item in enumerate(f.menu_subplots.MenuItems):
            assert item.Label == 'rgb%s'%(i+1)
            
        menu = f.get_imshow_menu((0,1))
        for item in menu.MenuItems:
            if item.Label == 'Channels':
                for item in item.SubMenu.MenuItems:
                    if item.Label == cpfig.COLOR_NAMES[0]:
                        event = wx.PyCommandEvent(wx.EVT_MENU.typeId, item.Id)
                        f.GetEventHandler().ProcessEvent(event)
                        self.app.ProcessPendingEvents()

        for i, item in enumerate(f.menu_subplots.MenuItems):
            assert item.Label == 'rgb%s'%(i+1)
        f.Destroy()

    def test_04_01_sharexy(self):
        '''Make sure we can use the sharexy argument.'''
        image = np.zeros((100, 100))
        for y in range(image.shape[0]):
            image[y,:] = y / 200.0
        my_frame = cpfig.create_or_find(self.frame, -1, subplots=(1, 2),
                                        name="test_04_01_sharexy")
        ax = my_frame.subplot_imshow(0, 0, image, normalize=False)
        ax2 = my_frame.subplot_imshow(0, 1, image, normalize=False, sharexy=ax)
        xgroup = ax.get_shared_x_axes()
        assert ax in xgroup
        assert ax2 in xgroup
        ygroup = ax.get_shared_y_axes()
        assert ax in ygroup
        assert ax2 in ygroup
        my_frame.Destroy()

    def test_04_02_no_sharexy_and_sharex_or_y(self):
        '''Make sure we can't specify sharex or sharey and sharexy'''
        image = np.zeros((100, 100))
        for y in range(image.shape[0]):
            image[y,:] = y / 200.0
        my_frame = cpfig.create_or_find(self.frame, -1, subplots=(1, 2),
                                        name="test_04_01_sharexy")
        ax = my_frame.subplot_imshow(0, 0, image, normalize=False)
        raised = False
        try:
            ax2 = my_frame.subplot_imshow(0, 1, image, normalize=False,
                                          sharex=ax, sharexy=ax)
        except Exception, e:
            raised = True
        assert raised, "Specifying sharex and sharexy did not raise exception"
        raised = False
        try:
            ax2 = my_frame.subplot_imshow(0, 1, image, normalize=False,
                                          sharey=ax, sharexy=ax)
        except Exception, e:
            raised = True
        assert raised, "Specifying sharey and sharexy did not raise exception"
        my_frame.Destroy()
