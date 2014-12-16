"""cpartists.py - Specialized matplotlib artists for CellProfiler

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import matplotlib
import matplotlib.artist
import numpy as np
from scipy.ndimage import distance_transform_edt

from cellprofiler.cpmath.cpmorphology import get_outline_pts
from cellprofiler.cpmath.outline import outline

'''Render the image in shades of gray'''
MODE_GRAYSCALE = "grayscale"
'''Render the image in shades of a color'''
MODE_COLORIZE = "colorize"
'''Render the image using a color map'''
MODE_COLORMAP = "colormap"
'''Render the image as RGB'''
MODE_RGB = "rgb"

NORMALIZE_RAW = "raw"
NORMALIZE_LOG = "log"
NORMALIZE_LINEAR = "linear"

INTERPOLATION_NEAREST = "nearest"
INTERPOLATION_BILINEAR = "bilinear"
INTERPOLATION_BICUBIC = "bicubic"

MODE_OUTLINES = "outlines"
MODE_LINES = "lines"
MODE_OVERLAY = "overlay"

class ImageData(object):
    '''The data and metadata needed to display an image
    
             name - the name of the channel
             pixel_data - a 2D or 2D * n-channel array of values
             mode - MODE_GRAYSCALE
             color - if present, render this image using values between 0,0,0
                     and this color. Default is white
             colormap - if present, use this color map for scaling
             alpha - when compositing an image on top of another, use this
                     alpha value times the color.
             normalization - one of the NORMALIZE_ constants
             vmin - the floor of the image range in raw mode - default is 0
             vmax - the ceiling of the image range in raw mode - default is 1
    '''
    def __init__(self, name, pixel_data, 
                 mode = None,
                 color = (1.0, 1.0, 1.0),
                 colormap = None,
                 alpha = 1,
                 normalization = None,
                 vmin = 0,
                 vmax = 1):
        self.name = name
        self.pixel_data = pixel_data
        self.__mode = mode
        self.color = color
        self.__colormap = colormap
        self.alpha = alpha
        self.normalization = normalization
        self.vmin = vmin
        self.vmax = vmax
        
    def set_mode(self, mode):
        self.__mode = mode
    
    def get_mode(self):
        if self.pixel_data.ndim == 3:
            return MODE_RGB
        elif self.__mode is None:
            return MODE_GRAYSCALE
        else:
            return self.__mode
        
    mode = property(get_mode, set_mode)
    
    def set_colormap(self, colormap):
        self.__colormap = colormap
        
    def get_colormap(self, rcparams=None):
        return self.__colormap or \
               (rcparams or matplotlib.rcParams)['image.cmap']
    
    colormap = property(
        get_colormap, set_colormap, 
        doc = "The name of the colormap to use to render the image")

class ObjectsData(object):
    '''The data needed to display objects

    name - the name of the objects
    labels - a sequence of label matrices
    outline_color - render outlines in this color. Default is 
    primary color.
    line_width - the width of the line in pixels for outlines
    colormap - render overlaid labels using this colormap
    alpha - render overlaid labels using this alpha value.
    mode - the display mode: outlines, lines, overlay or hide
    scramble - True (default) to scramble the colors. False to use
                         the labels to pick from the colormap.
    '''
    def __init__(self, name, labels, 
                 outline_color = None,
                 line_width = None,
                 colormap = None,
                 alpha = None,
                 mode = None,
                 scramble = True):
        self.name = name
        self.__labels = labels
        self.__outline_color = outline_color
        self.__line_width = line_width
        self.colormap = colormap
        self.alpha = alpha
        self.mode = mode
        self.scramble = scramble
        self.__outlines = None
        self.__points = None
        
    def get_labels(self):
        return self.__labels
    
    def set_labels(self, labels):
        self.__labels = labels
        self.__outlines = None
        self.__points = None
    
    labels = property(get_labels, set_labels)
    
    def get_outline_color(self):
        return self.__outline_color
    
    def set_outline_color(self, color):
        self.__outline_color = color
        if self.__points is not None:
            self.__points.set_color(color)
    outline_color = property(get_outline_color, set_outline_color)
    
    def get_line_width(self):
        return self.__line_width
    
    def set_line_width(self, line_width):
        self.__line_width = line_width
        self.__outlines = None
        if self.__points is not None:
            self.__points.set_linewidth(line_width)
    line_width = property(get_line_width, set_line_width)
    
    @property
    def outlines(self):
        '''Get a mask of all the points on the border of objects'''
        if self.__outlines == None:
            for i, labels in enumerate(self.labels):
                if i == 0:
                    self.__outlines = outline(labels) != 0
                else:
                    self.__outlines = self.__outlines | (outline(labels) != 0)
            if self.line_width > 1:
                hw = float(self.line_width) / 2
                d = distance_transform_edt(~ self.__outlines)
                dti, dtj = np.where((d < hw+.5) & ~self.__outlines)
                self.__outlines = self.__outlines.astype(np.float32)
                self.__outlines[dti, dtj] = np.minimum(1, hw + .5 - d[dti, dtj])
                
        return self.__outlines.astype(np.float32)
    
    @property
    def points(self):
        '''Return an artist for drawing the points'''
        if self.__points == None:
            self.__points = CPOutlineArtist(
                self.name, self.labels, linewidth = self.line_width,
                color = self.outline_color)
        return self.__points
        
class MaskData(object):
    '''The data needed to display masks
    
    name - name of the mask
    mask - the binary mask
    mode - the display mode: outline, lines or overlay
    fg_color - color of outline or line or overlay
    bg_color - background color if overlay mode (default = black)
    fg_alpha - alpha of foreground
    bg_alpha - alpha of background
    
    So, if you want the masked-out portion to be black, this would
    be your MaskData:
    
    MaskData(name, mask, MODE_OVERLAY, (0, 0, 0), (0, 0, 0), 0, 1)
    '''
    def __init__(self, name, mask,
                 mode = None,
                 fg_color = (1, 1, 1),
                 bg_color = (0, 0, 0),
                 fg_alpha = 0,
                 bg_alpha = 1):
        self.name = name
        self.mask = mask
        self.mode = mode
        self.fg_color = fg_color
        self.bg_color = bg_color
        self.fg_alpha = fg_alpha
        self.bg_alpha = bg_alpha

class CPImageArtist(matplotlib.artist.Artist):
    '''An artist that displays multiple images and objects
    
    The image artist maintains each image and object set separately as
    well as separate rendering colors, interpolation, intensity normalization
    and display styles.
    
    The keyword arguments:
    
    images - a sequence of ImageData objects to be composited. 
             
    objects - a sequence of ObjectsData to be composited. 
              
                         
    masks - a sequence of masks
            name - the name of the mask
            mask - the binary matrix for the mask
    '''
    def __init__(self, images = None, objects = None, masks = None, 
                 interpolation = None):
        '''Initialize the artist with the images and objects'''
        super(CPImageArtist, self).__init__()
        self.__images = images or []
        self.__objects = objects or []
        self.__masks = masks or []
        self.__interpolation = interpolation
        self.filterrad = 4.0
        
    def set_interpolation(self, interpolation):
        self.__interpolation = interpolation
        
    def get_interpolation(self, rcparams=None):
        return self.__interpolation or \
               (rcparams or matplotlib.rcParams)['image.interpolation']
    
    interpolation = property(
        get_interpolation, set_interpolation,
        "The interpolation to use when stretching intensities")
    
    def get_border_count(self):
        '''# of pixels needed for interpolation'''
        if self.interpolation == INTERPOLATION_NEAREST:
            return 1
        elif self.interpolation == INTERPOLATION_BICUBIC:
            return 2
        else:
            return 3
    @property
    def mp_interpolation(self):
        '''Matplotlib-based interpolation constant'''
        if self.interpolation == INTERPOLATION_BICUBIC:
            return matplotlib.image.BICUBIC
        elif self.interpolation == INTERPOLATION_BILINEAR:
            return matplotlib.image.BILINEAR
        return matplotlib.image.NEAREST

    def draw(self, renderer):
        magnification = renderer.get_image_magnification()
        shape = [0, 0]
        for image in self.__images:
            for i in range(2):
                shape[i] = max(shape[i], image.pixel_data.shape[i])
        border = self.get_border_count()
            
        vl = self.axes.viewLim
        view_xmin = int(max(0, min(vl.x0, vl.x1) - self.filterrad))
        view_ymin = int(max(0, min(vl.y0, vl.y1) - self.filterrad))
        view_xmax = int(min(shape[1], max(vl.x0, vl.x1) + self.filterrad))
        view_ymax = int(min(shape[0], max(vl.y0, vl.y1) + self.filterrad))
        flip_ud = vl.y0 > vl.y1
        flip_lr = vl.x0 > vl.x1
        if shape[1] <= view_xmin or shape[1] <= - view_xmin or view_xmax <= 0:
            return
        if shape[0] <= view_ymin or shape[0] <= - view_ymin or view_ymax <= 0:
            return
        
        target = np.zeros(
            (view_ymax - view_ymin, view_xmax - view_xmin, 3), np.float32)
        color_alpha = np.array((0.0, 0.0, 0.0))
        alpha = 0
        def get_tile_and_target(pixel_data):
            '''Return the visible tile of the image and a view of the target'''
            if pixel_data.shape[1] <= abs(view_xmin) or \
               pixel_data.shape[0] <= abs(view_ymin):
                continue
            xmin = max(0, view_xmin)
            ymin = max(0, view_ymin)
            xmax = min(view_xmax, pixel_data.shape[1])
            ymax = min(view_ymax, pixel_data.shape[0])
            pixel_data = pixel_data[ymin:ymax, xmin:xmax]
            if flip_ud:
                pixel_data = np.flipud(pixel_data)
            if flip_lr:
                pixel_data = np.fliplr(pixel_data)
            target_view = target[:(ymax - view_ymin), :(xmax - view_xmin), :]
            return pixel_data, target_view
            
        for image in self.__images:
            assert isinstance(image, ImageData)
            pixel_data, target_view = get_tile_and_target(image.pixel_data)
                
            if image.normalization in (NORMALIZE_LINEAR, NORMALIZE_LOG):
                pd_max = np.max(pixel_data)
                pd_min = np.min(pixel_data)
                if pd_min == pd_max:
                    pixel_data = np.zeros(pixel_data.shape, np.float32)
                else:
                    pixel_data = (pixel_data - pd_min) / (pd_max - pd_min)
            else:
                pixel_data = pixel_data.copy()
                pixel_data[pixel_data < image.vmin] = image.vmin
                pixel_data[pixel_data > image.vmax] = image.vmax
            if image.normalization == NORMALIZE_LOG:
                log_eps = np.log(1.0/256)
                log_one_plus_eps = np.log(257.0 / 256)
                pixel_data = (np.log(pixel_data + 1.0/256) - log_eps) / \
                    (log_one_plus_eps - log_eps)
                
            nextalpha = alpha + image.alpha - alpha * image.alpha
            if image.mode == MODE_COLORIZE or image.mode == MODE_GRAYSCALE:
                # The idea here is that the color is the alpha for each of
                # the three channels.
                if image.mode == MODE_COLORIZE:
                    imalpha = np.array(image.color) * image.alpha / \
                        np.sum(image.color)
                else:
                    imalpha = np.array([image.alpha] * 3)
                    
                target_view[:, :, :] = (
                    target_view * color_alpha[np.newaxis, np.newaxis, :] + 
                    pixel_data[:, :, np.newaxis] *
                    imalpha[np.newaxis, np.newaxis, :]) / nextalpha
                color_alpha = color_alpha + imalpha - color_alpha * imalpha
            else:
                if image.mode == MODE_COLORMAP:
                    sm = matplotlib.cm.ScalarMappable(cmap = image.colormap)
                    if image.normalization == NORMALIZE_RAW:
                        sm.set_clim((image.vmin, image.vmax))
                    pixel_data = sm.to_rgba(pixel_data)[:, :, :3]
                target_view[:, :, :] = (
                    target_view * alpha + pixel_data * image.alpha) / nextalpha
            alpha = nextalpha
        
        for objects in self.__objects:
            assert isinstance(objects, ObjectsData)
            if objects.mode == MODE_LINES:
                objects.points.draw(renderer)
                continue
            if objects.mode == MODE_OUTLINES:
                mask, target_view = get_tile_and_target(objects.outlines)
                oalpha = np.array(objects.outline_color) * outline_.alpha / \
                    np.sum(outline.color)
                target_view[:, :, :] = (
                    target_view * color_alpha[np.newaxis, np.newaxis, :] +
                    objects.outlines * oalpha[np.newaxis, np.newaxis, :]) / \
                nextalpha
                
                
        im = matplotlib.image.fromarray(target, 0)
        im.is_grayscale = False
        im.set_interpolation(self.mp_interpolation)
        fc = matplotlib.rcParams['axes.facecolor']
        bg = matplotlib.colors.colorConverter.to_rgba(fc, 0)
        im.set_bg( *bg)
        
        # image input dimensions
        im.reset_matrix()

        # the viewport translation in the X direction
        tx = view_xmin - min(vl.x0, vl.x1) - .5
        #
        # the viewport translation in the Y direction
        # which is from the bottom of the screen
        #
        if self.axes.viewLim.height < 0:
            ty = (self.axes.viewLim.y0 - view_ymin) + .5
        else:
            ty = view_ymin - self.axes.viewLim.y0 - .5
        im.apply_translation(tx, ty)
        l, b, r, t = self.axes.bbox.extents
        if b > t:
            t, b = b, t
        widthDisplay = (r - l + 1) * magnification
        heightDisplay = (t - b + 1) * magnification

        # resize viewport to display
        sx = widthDisplay / self.axes.viewLim.width
        sy = abs(heightDisplay  / self.axes.viewLim.height)
        im.apply_scaling(sx, sy)
        im.resize(widthDisplay, heightDisplay,
                  norm=1, radius = self.filterrad)
        bbox = self.axes.bbox.frozen()
        
        # Two ways to do this, try by version
        mplib_version = matplotlib.__version__.split(".")
        if mplib_version[0] == '0':
            renderer.draw_image(l, b, im, bbox)
        else:
            gc = renderer.new_gc()
            gc.set_clip_rectangle(bbox)
            renderer.draw_image(gc, l, b, im)
            

class CPOutlineArtist(matplotlib.collections.LineCollection):
    '''An artist that is a plot of the outline around an object
    
    This class is here so that we can add and remove artists for certain
    outlines.
    '''
    def __init__(self, name, labels, *args, **kwargs):
        '''Draw outlines for objects
        
        name - the name of the outline
        
        labels - a sequence of labels matrices
        
        kwargs - further arguments for Line2D
        '''
        # get_outline_pts has its faults:
        # * it doesn't do holes
        # * it only does one of two disconnected objects
        #
        # We get around the second failing by resegmenting with
        # connected components and combining the original and new segmentation
        #
        lines = []
        for l in labels:
            new_labels, counts = label(l != 0, np.ones((3, 3), bool))
            if counts == 0:
                continue
            l = l.astype(np.uint64) * counts + new_labels
            unique, idx = np.unique(l.flatten(), return_inverse=True)
            if unique[0] == 0:
                my_range = np.arange(len(unique))
            else:
                my_range = np.arange(1, len(unique))
            idx.shape = l.shape
            pts, offs, counts = get_outline_pts(idx, my_range)
            pts = pts[:, ::-1] # matplotlib x, y reversed from i,j
            for off, count in zip(offs, counts):
                lines.append(np.vstack((pts[off:off+count], pts[off:off+1])))
        matplotlib.collections.LineCollection.__init__(
            self, lines, *args, **kwargs)
        
    def get_outline_name(self):
        return self.__outline_name


if __name__ == "__main__":
    import javabridge
    import bioformats
    import wx
    import sys
    matplotlib.use('WXAgg')
    from wx.lib.inspection import InspectionTool
    import matplotlib.pyplot
    
    javabridge.start_vm(class_path=bioformats.JARS)
    try:
        app = wx.PySimpleApp()
        figure = matplotlib.figure.Figure()
        images = []
        for i, arg in enumerate(sys.argv[1:]):
            img = bioformats.load_image(arg)
            images.append(
                ImageData("Image %d" % (i+1), img,
                          alpha = 1.0 / (len(sys.argv) - 1)))
        artist = CPImageArtist(images = images)
        figure = matplotlib.pyplot.figure()
        ax = figure.add_axes((0.05, 0.05, .9, .9))
        ax.set_xlim(0, images[0].pixel_data.shape[1])
        ax.set_ylim(0, images[0].pixel_data.shape[0])
        ax.add_artist(artist)
        inspector = InspectionTool()
        my_locals = dict([(k, v) for k, v in globals().items() if k.isupper()])
        my_locals['images'] = images
        my_locals['draw'] = matplotlib.pyplot.draw
        inspector.Init(locals = my_locals)
        inspector.Show()
        matplotlib.pyplot.draw()
        matplotlib.pyplot.show()
        app.MainLoop()
        
    finally:
        javabridge.kill_vm()