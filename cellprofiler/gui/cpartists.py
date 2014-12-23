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
import matplotlib.collections
import numpy as np
from scipy.ndimage import distance_transform_edt, label

from cellprofiler.cpmath.cpmorphology import get_outline_pts
from cellprofiler.cpmath.outline import outline
from cellprofiler.gui.cpfigure_tools import renumber_labels_for_display

'''Render the image in shades of gray'''
MODE_GRAYSCALE = "grayscale"
'''Render the image in shades of a color'''
MODE_COLORIZE = "colorize"
'''Render the image using a color map'''
MODE_COLORMAP = "colormap"
'''Render the image as RGB'''
MODE_RGB = "rgb"
'''Do not display'''
MODE_HIDE = "hide"

NORMALIZE_RAW = "raw"
NORMALIZE_LOG = "log"
NORMALIZE_LINEAR = "linear"

INTERPOLATION_NEAREST = "nearest"
INTERPOLATION_BILINEAR = "bilinear"
INTERPOLATION_BICUBIC = "bicubic"

MODE_OUTLINES = "outlines"
MODE_LINES = "lines"
MODE_OVERLAY = "overlay"
MODE_INVERTED = "inverted"

class ColorMixin(object):
    '''A mixin for burying the color, colormap and alpha in hidden attributes
    '''
    def __init__(self):
        self._alpha = 1
        self._color = None
        self._colormap = None
        
    def _set_alpha(self, alpha):
        self._alpha = alpha
        self._on_alpha_changed()
        
    def _get_alpha(self):
        return self._alpha
    
    def get_alpha(self):
        return self._get_alpha()
    
    def set_alpha(self, alpha):
        self._set_alpha(alpha)
    
    alpha = property(get_alpha, set_alpha)
    
    def _on_alpha_changed(self):
        '''Called when the alpha value is modified, default = do nothing'''
        pass
    
    def _get_color(self):
        '''Get the color - default is the matplotlib foreground color'''
        if self._color is None:
            color = matplotlib.rcParams.get('patch.facecolor', 'b')
        else:
            color = self._color
        return matplotlib.colors.colorConverter.to_rgb(color)
    
    def _set_color(self, color):
        '''Set the color'''
        self._color = color
        self._on_color_changed()
    
    def get_color(self):
        return ColorMixin._get_color(self)
    
    def set_color(self, color):
        ColorMixin._set_color(self, color)
        
    color = property(get_color, set_color)
    
    def _on_color_changed(self):
        '''Called when the color changed, default = do nothing'''
        pass
    
    def _get_colormap(self):
        '''Override to get the colormap'''
        if self._colormap == None:
            return matplotlib.rcParams.get('image.cmap', 'jet')
        return self._colormap
    
    def _set_colormap(self, colormap):
        '''Override to set the colormap'''
        self._colormap = colormap
        self._on_colormap_changed()
        
    def get_colormap(self):
        return self._get_colormap()
    
    def set_colormap(self, colormap):
        self._set_colormap(colormap)
        
    colormap = property(get_colormap, set_colormap)
        
    def _on_colormap_changed(self):
        '''Called when the colormap changed, default = do nothing'''
        pass
    
    @property
    def using_alpha(self):
        '''True if this data's configuration will use the alpha setting'''
        return self._using_alpha()
    
    def _using_alpha(self):
        '''Override if we ever don't use the alpha setting'''
        return True
    
    @property
    def using_color(self):
        '''True if this data's configuration will use the color setting'''
        return self._using_color()
    
    def _using_color(self):
        '''Override this to tell the world that the data will use the color setting'''
        return False
    
    @property
    def using_colormap(self):
        '''True if this data's configuration will use the colormap setting'''
        return self._using_colormap()
    
    def _using_colormap(self):
        '''Override this to tell the world that the data will use the colormap setting'''
        return False
    
class ImageData(ColorMixin):
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
                 alpha = None,
                 normalization = None,
                 vmin = 0,
                 vmax = 1):
        self.name = name
        self.pixel_data = pixel_data
        self.__mode = mode
        self.color = color
        self.colormap = colormap
        if alpha is not None:
            self.alpha = alpha
        self.normalization = normalization
        self.vmin = vmin
        self.vmax = vmax
        
    def set_mode(self, mode):
        self.__mode = mode
    
    def get_mode(self):
        if self.__mode == MODE_HIDE:
            return MODE_HIDE
        if self.pixel_data.ndim == 3:
            return MODE_RGB
        elif self.__mode is None:
            return MODE_GRAYSCALE
        else:
            return self.__mode
        
    mode = property(get_mode, set_mode)
    
    def _using_color(self):
        return self.mode == MODE_COLORIZE
    
    def _using_colormap(self):
        return self.mode == MODE_COLORMAP
        
class OutlinesMixin(ColorMixin):
    '''Provides rendering of "labels" as outlines
    
    Needs self.labels to return a sequence of labels matrices and needs
    self._flush_outlines to be called when parameters change.
    '''
    def __init__(self, outline_color, line_width):
        super(OutlinesMixin, self).__init__()
        self._flush_outlines()
        self.color = outline_color
        self._line_width = line_width
        
    def _flush_outlines(self):
        self._outlines = None
        self._points = None
        
    def _on_color_changed(self):
        if self._points is not None:
            self._points.set_color(self.color)
            
    def get_line_width(self):
        return self._line_width
    
    def set_line_width(self, line_width):
        self._line_width = line_width
        self._outlines = None
        if self._points is not None:
            self._points.set_linewidth(line_width)
    line_width = property(get_line_width, set_line_width)
    
    @property
    def outlines(self):
        '''Get a mask of all the points on the border of objects'''
        if self._outlines == None:
            for i, labels in enumerate(self.labels):
                if i == 0:
                    self._outlines = outline(labels) != 0
                else:
                    self._outlines = self._outlines | (outline(labels) != 0)
            if self.line_width > 1:
                hw = float(self.line_width) / 2
                d = distance_transform_edt(~ self._outlines)
                dti, dtj = np.where((d < hw+.5) & ~self._outlines)
                self._outlines = self._outlines.astype(np.float32)
                self._outlines[dti, dtj] = np.minimum(1, hw + .5 - d[dti, dtj])
      
        return self._outlines.astype(np.float32)
    
    @property
    def points(self):
        '''Return an artist for drawing the points'''
        if self._points == None:
            self._points = CPOutlineArtist(
                self.name, self.labels, linewidth = self.line_width,
                color = self.color)
        return self._points
    
class ObjectsData(OutlinesMixin):
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
        super(ObjectsData, self).__init__(outline_color, line_width)
        self.name = name
        self.__labels = labels
        self.colormap = colormap
        if alpha is not None:
            self.alpha = alpha
        self.mode = mode
        self.scramble = scramble
        self.__overlay = None
        
    def get_labels(self):
        return self.__labels
    
    def set_labels(self, labels):
        self.__labels = labels
        self._flush_outlines()
        self.__overlay = None
    
    labels = property(get_labels, set_labels)
    
    def _on_colormap_changed(self):
        super(ObjectsData, self)._on_colormap_changed()
        self.__overlay = None
        
    def _using_color(self):
        return self.mode in (MODE_LINES, MODE_OUTLINES)
        
    def _using_colormap(self):
        return self.mode == MODE_OVERLAY
        
    @property
    def overlay(self):
        '''Return a color image of the segmentation as an overlay
        '''
        if self.__overlay is not None:
            return self.__overlay
        sm = matplotlib.cm.ScalarMappable(cmap = self.colormap)
        sm.set_clim(vmin=1, vmax=np.max([np.max(l) for l in self.labels])+1)

        img = None
        lmin = 0
        for l in self.labels:
            if self.scramble:
                lmin = np.min(l[l!=0])
            l[l!=0] = renumber_labels_for_display(l)[l!=0]+lmin
            lmin = np.max(l)
            if img is None:
                img = sm.to_rgba(l)
                img[l==0, :] = 0
            else:
                img[l!=0, :] = sm.to_rgba(l[l!=0])
        self.__overlay = img
        return img
        
class MaskData(OutlinesMixin):
    '''The data needed to display masks
    
    name - name of the mask
    mask - the binary mask
    mode - the display mode: outline, lines, overlay or inverted
    color - color of outline or line or overlay
    alpha - alpha of the outline or overlay
    
    MODE_OVERLAY colors the part of the mask that covers the part of the
    image that the user cares about. Ironically, the user almost certainly
    wants to see that part and MODE_INVERTED with an alpha of 1 masks the
    part that the user does not want to see and ignores the part they do.
    '''
    def __init__(self, name, mask,
                 mode = None,
                 color = None,
                 line_width = None,
                 alpha = None):
        super(MaskData, self).__init__(color, line_width)
        self.name = name
        self.mask = mask
        self.mode = mode
        if alpha is not None:
            self.alpha = alpha
        
    @property
    def labels(self):
        '''Return the mask as a sequence of labels matrices'''
        return [self.mask.astype(np.uint8)]
    
    def _using_color(self):
        return True
    
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
    
    MI_IMAGES = "Images"
    MI_OBJECTS = "Objects"
    MI_MASKS = "Masks"
    MI_INTERPOLATION = "Interpolation"
    MI_NEAREST_NEIGHBOR = "Nearest neighbor"
    MI_BILINEAR = "Bilinear"
    MI_BICUBIC = "Bicubic"
    MI_RAW = "Raw"
    MI_LINEAR = "Normalized"
    MI_LOG = "Log normalized"
    MI_LINES = "Lines"
    MI_OUTLINES = "Outlines"
    MI_OVERLAY = "Overlay"
    MI_MODE = "Mode"
    MI_ALPHA = "Alpha"
    MI_COLOR = "Color"
    MI_GRAYSCALE = "Grayscale"
    MI_MASK = "Mask"
    MI_INVERTED = "Inverted mask"
    MI_COLORMAP = "Color map"
    MI_NORMALIZATION = "Intensity normalization"
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
    
    def add(self, data):
        '''Add an image, objects or mask to the artist
        
        data - ImageData, ObjectsData or MaskData to be added
        '''
        assert isinstance(data, (ImageData, ObjectsData, MaskData))
        if isinstance(data, ImageData):
            self.__images.append(data)
        elif isinstance(data, ObjectsData):
            self.__objects.append(data)
        else:
            self.__masks.append(data)
            
    def remove(self, data):
        '''Remove an image, object or mask from the artist
        
        data - an ImageData, ObjectData or MaskData previously
               added (via constructor or add)
        '''
        assert isinstance(data, (ImageData, ObjectsData, MaskData))
        if isinstance(data, ImageData):
            self.__images.remove(data)
        elif isinstance(data, ObjectsData):
            self.__objects.remove(data)
        else:
            self.__masks.remove(data)
            
    def remove_image_by_name(self, name):
        '''Remove an image via the name given to it in its data'''
        for data in self.__images:
            if data.name == name:
                return self.remove(data)
        else:
            raise ValueError("Could not find image named %s" % name)
        
    def remove_objects_by_name(self, name):
        '''Remove objects via their name given to it in its data'''
        for data in self.__objects:
            if data.name == name:
                return self.remove(data)
        else:
            raise ValueError("Could not find objects named %s" % name)

    def remove_mask_by_name(self, name):
        '''Remove a mask via the name given to it in its data'''
        for data in self.__masks:
            if data.name == name:
                return self.remove(data)
        else:
            raise ValueError("Could not find mask named %s" % name)
        
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
            if image.mode == MODE_HIDE:
                continue
            for i in range(2):
                shape[i] = max(shape[i], image.pixel_data.shape[i])
        if any([x==0 for x in shape]):
            return
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
        
        # First 3 color indices are intensities
        # Second 3 are per-color alpha values
        
        target = np.zeros(
            (view_ymax - view_ymin, view_xmax - view_xmin, 6), np.float32)
        def get_tile_and_target(pixel_data):
            '''Return the visible tile of the image and a view of the target'''
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
            if image.mode == MODE_HIDE:
                continue
            if image.pixel_data.shape[1] <= abs(view_xmin) or \
               image.pixel_data.shape[0] <= abs(view_ymin):
                continue
            pixel_data, target_view = get_tile_and_target(image.pixel_data)
            tv_alpha = target_view[:, :, 3:]
            tv_image = target_view[:, :, :3]
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
                
            if image.mode == MODE_COLORIZE or image.mode == MODE_GRAYSCALE:
                # The idea here is that the color is the alpha for each of
                # the three channels.
                if image.mode == MODE_COLORIZE:
                    imalpha = np.array(image.color) * image.alpha / \
                        np.sum(image.color)
                else:
                    imalpha = np.array([image.alpha] * 3)
                pixel_data = pixel_data[:, :, np.newaxis]
                imalpha = imalpha[np.newaxis, np.newaxis, :]    
            else:
                if image.mode == MODE_COLORMAP:
                    sm = matplotlib.cm.ScalarMappable(cmap = image.colormap)
                    if image.normalization == NORMALIZE_RAW:
                        sm.set_clim((image.vmin, image.vmax))
                    pixel_data = sm.to_rgba(pixel_data)[:, :, :3]
                imalpha = image.alpha
            tv_image[:] = \
                tv_image * tv_alpha * (1 - imalpha) + pixel_data * imalpha
            tv_alpha[:] = \
                tv_alpha + imalpha - tv_alpha * imalpha
            tv_image[tv_alpha != 0] /= tv_alpha[tv_alpha != 0]
        
        for om in list(self.__objects) + list(self.__masks):
            assert isinstance(om, OutlinesMixin)
            if om.mode in (MODE_LINES, MODE_HIDE):
                continue
            if om.mode == MODE_OUTLINES:
                oshape = om.outlines.shape
                if oshape[1] <= abs(view_xmin) or \
                   oshape[0] <= abs(view_ymin):
                    continue
                mask, target_view = get_tile_and_target(om.outlines)
                tv_alpha = target_view[:, :, 3:]
                tv_image = target_view[:, :, :3]
                oalpha = (mask.astype(float) * om.alpha)[:, :, np.newaxis]
                ocolor = \
                    np.array(om.color)[np.newaxis, np.newaxis, :]
            elif isinstance(om, ObjectsData) and om.mode == MODE_OVERLAY:
                oshape = om.outlines.shape
                if oshape[1] <= abs(view_xmin) or \
                   oshape[0] <= abs(view_ymin):
                    continue
                ocolor, target_view = get_tile_and_target(
                    om.overlay[:, :, :3])
                oalpha = om.overlay[:, :, 3]* om.alpha
                oalpha = oalpha[:, :, np.newaxis]
            elif isinstance(om, MaskData) and \
                 om.mode in (MODE_OVERLAY, MODE_INVERTED):
                mask = om.mask
                if om.mode == MODE_INVERTED:
                    mask = ~mask
                mask = mask[:, :, np.newaxis]
                color = np.array(om.color, np.float32)[np.newaxis, np.newaxis, :]
                ocolor = mask * color
                oalpha = mask * om.alpha
            else:
                continue
            tv_image[:] = tv_image * tv_alpha * (1 - oalpha) + ocolor * oalpha
            tv_alpha[:] = tv_alpha + oalpha - tv_alpha * oalpha
            tv_image[tv_alpha != 0] /= tv_alpha[tv_alpha != 0]
       
        target = target[:, :, :3]
        np.clip(target, 0, 1, target)
        im = matplotlib.image.fromarray(target[:, :, :3], 0)
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
        for om in list(self.__objects) + list(self.__masks):
            assert isinstance(om, OutlinesMixin)
            if om.mode == MODE_LINES:
                om.points.set_axes(self.axes)
                om.points.set_transform(self.axes.transData)
                om.points.set_clip_path(self.axes.patch)
                om.points.draw(renderer)
    
    def add_to_menu(self, target, menu):
        '''Add to a context menu for a WX ui
        
        target - target window that will receive menu events.
        '''
        import wx
        assert isinstance(menu, wx.Menu)
        interpolation_menu = wx.Menu()
        assert isinstance(menu, wx.Menu)
        menu.AppendSeparator()
        menu.AppendSubMenu(interpolation_menu, self.MI_INTERPOLATION)
        for label, state in (
            (self.MI_NEAREST_NEIGHBOR, INTERPOLATION_NEAREST),
            (self.MI_BILINEAR, INTERPOLATION_BILINEAR),
            (self.MI_BICUBIC, INTERPOLATION_BICUBIC)):
            my_id = wx.NewId()
            submenu_item = interpolation_menu.AppendRadioItem(my_id, label)
            target.Bind(
                wx.EVT_MENU, 
                (lambda event, target=state:
                 self.on_interpolation_menu_event(event, target)),
                id=my_id)
            target.Bind(
                wx.EVT_UPDATE_UI,
                (lambda event, target=state: 
                 self.on_interpolation_update_event(event, target)),
                id = my_id)
            if state == self.interpolation:
                submenu_item.Check(True)
        menu.AppendSeparator()
            
    def on_interpolation_menu_event(self, event, target):
        self.interpolation = target
        self.refresh()
            
    def on_interpolation_update_event(self, event, target):
        import wx
        assert isinstance(event, wx.UpdateUIEvent)
        event.Check(self.interpolation == target)
        
    def on_update_menu(self, event, menu):
        import wx
        assert isinstance(menu, wx.Menu)
        menu_items = list(menu.GetMenuItems())
        breaks = ((self.MI_IMAGES, self.__images),
                  (self.MI_OBJECTS, self.__objects),
                  (self.MI_MASKS, self.__masks))
        for start, item in enumerate(menu_items):
            assert isinstance(item, wx.MenuItem)
            if item.Label == self.MI_INTERPOLATION:
                break
        else:
            return
        window = self.__get_window_from_event(event)
        idx = start + 1
        if menu_items[idx].IsSeparator():
            idx += 1
        label_fmt = "--- %s ---"
        for key, sequence in breaks:
            label = label_fmt % key
            if idx >= len(menu_items) or menu_items[idx].Text != label:
                item = menu.Insert(idx, wx.NewId(), label)
                item.Enable(False)
                menu_items.insert(idx, item)
            idx += 1
            #
            # Pair data items with menu items
            #
            for data in sequence:
                name = data.name
                if idx == len(menu_items) or\
                   menu_items[idx].Text.startswith("---") or\
                   menu_items[idx].IsSeparator():
                    sub_menu = wx.Menu()
                    my_id =  wx.NewId()
                    sub_menu_item = menu.InsertMenu(
                        idx, my_id, name, sub_menu)
                    if data.mode == MODE_HIDE:
                        sub_menu_item.Enable(False)
                    menu_items.insert(idx, sub_menu_item)
                    self.__initialize_sub_menu(event, sub_menu, data)
                    def on_update_ui(event, sub_menu = sub_menu, data=data):
                        self.__update_sub_menu(event, sub_menu, data)
                    window.Bind(
                        wx.EVT_UPDATE_UI, on_update_ui, id = my_id)
                    idx += 1
                else:
                    self.__update_sub_menu(
                        menu_items[idx], menu_items[idx].GetMenu(), data)
                    idx += 1
            #
            # Remove excess menu items
            #
            while len(menu_items) < idx and menu_items[idx].IsEnabled():
                menu.RemoveItem(item)
                del menu_items[idx]
    
    def __initialize_sub_menu(self, event, sub_menu, data):
        import wx
        assert isinstance(sub_menu, wx.Menu)
        if isinstance(data, ImageData):
            self.__initialize_image_sub_menu(event, sub_menu, data)
        elif isinstance(data, ObjectsData):
            self.__initialize_objects_sub_menu(event, sub_menu, data)
        elif isinstance(data, MaskData):
            self.__initialize_mask_sub_menu(event, sub_menu, data)
            
    def __initialize_image_sub_menu(self, event, sub_menu, data):
        import wx
        item = sub_menu.Append(wx.NewId(), self.MI_NORMALIZATION)
        item.Enable(False)
        window = self.__get_window_from_event(event)
        for label, target in (
            (self.MI_RAW, NORMALIZE_RAW),
            (self.MI_LINEAR, NORMALIZE_LINEAR),
            (self.MI_LOG, NORMALIZE_LOG)):
            my_id = wx.NewId()
            sub_menu.AppendRadioItem(my_id, label)
            window.Bind(
                wx.EVT_MENU,
                (lambda event, data = data, target=target:
                 self.__on_set_normalization(data, target)),
                id = my_id)
            window.Bind(
                wx.EVT_UPDATE_UI,
                (lambda event, data = data, target=target:
                 self.__on_update_normalization(event, data, target)),
                id = my_id)
        sub_menu.AppendSeparator()
        my_id = wx.NewId()
        item = sub_menu.Append(my_id, self.MI_MODE)
        item.Enable(False)
        for label, target in (
            (self.MI_COLOR, MODE_COLORIZE),
            (self.MI_GRAYSCALE, MODE_GRAYSCALE),
            (self.MI_COLORMAP, MODE_COLORMAP)):
            def update_mode(event_or_item, data=data, target=target):
                if data.mode == MODE_RGB:
                    event_or_item.Enable(False)
                else:
                    event_or_item.Enable(True)
                    event_or_item.Check(data.mode == target)
            def on_mode(event, data=data, target=target):
                data.mode = target
                self.refresh()
            my_id = wx.NewId()
            item = sub_menu.AppendRadioItem(my_id, label)
            update_mode(item)
            window.Bind(wx.EVT_MENU, on_mode, id=my_id)
            window.Bind(wx.EVT_UPDATE_UI, update_mode, id=my_id)
        sub_menu.AppendSeparator()
        self.__add_color_item(
            event, sub_menu, data, 
            "Set image color", "Set image colormap")
        self.__add_alpha_item(event, sub_menu, data, "Set image transparency")
           
    def __on_set_normalization(self, data, target):
        assert isinstance(data, ImageData)
        data.normalization = target
        self.refresh()

    def __add_color_item(self, event, sub_menu, data, color_msg, colormap_msg):
        import wx
        assert isinstance(data, ColorMixin)
        my_id = wx.NewId()
        item = sub_menu.Append(my_id, self.MI_COLOR)
        window = self.__get_window_from_event(event)
            
        def on_color(event, 
                     data = data,
                     color_msg = color_msg,
                     colormap_msg = colormap_msg):
            if data.using_color:
                self.__on_color_dlg(event, color_msg, data)
            elif data.using_colormap:
                self.__on_colormap_dlg(event, colormap_msg, data)
                
        def on_update(event_or_item, data=data):
            assert isinstance(data, ColorMixin)
            event_or_item.Enable(data.using_color or data.using_colormap)
        window.Bind(
            wx.EVT_MENU, on_color, id = my_id)
        window.Bind(
            wx.EVT_UPDATE_UI, on_update, id= my_id)
        on_update(item)
        
    def __add_alpha_item(self, event, sub_menu, data, msg):
        import wx
        my_id = wx.NewId()
        item = sub_menu.Append(my_id, self.MI_ALPHA)
        window = self.__get_window_from_event(event)
        def set_alpha(alpha, data=data):
            data.alpha = alpha
            self.refresh()
            
        def on_alpha(event, data=data, msg=msg):
            self.__on_alpha_dlg(event, msg, data)
            
        def update_alpha(event_or_item, data=data):
            assert isinstance(data, ColorMixin)
            event_or_item.Enable(data.using_alpha)
        
        window.Bind(wx.EVT_MENU, on_alpha, id=my_id)
        window.Bind(wx.EVT_UPDATE_UI, update_alpha, id= my_id)
        update_alpha(item)
        
    def refresh(self):
        if self.figure is not None and self.figure.canvas is not None:
            self.figure.canvas.draw_idle()
        
    def __get_window_from_event(self, event):
        import wx
        o = event.EventObject
        if isinstance(o, wx.Menu):
            return o.GetInvokingWindow()
        return o
    
    def __on_update_normalization(self, event, data, target):
        assert isinstance(data, ImageData)
        event.Check(data.normalization == target)
        
    def __on_update_image_color_item(self, event, data):
        event.Enable(data.mode != MODE_RGB)
        
    def __initialize_objects_sub_menu(self, event, sub_menu, data):
        import wx
        assert isinstance(data, ObjectsData)
        assert isinstance(sub_menu, wx.Menu)
        item = sub_menu.Append(wx.NewId(), "Display mode")
        item.Enable(False)
        window = self.__get_window_from_event(event)
        for label, mode in ((self.MI_LINES, MODE_LINES),
                            (self.MI_OUTLINES, MODE_OUTLINES),
                            (self.MI_OVERLAY, MODE_OVERLAY)):
            my_id = wx.NewId()
            sub_menu.AppendRadioItem(my_id, label)
            window.Bind(
                wx.EVT_MENU,
                (lambda event, data = data, mode=mode:
                 self.__on_set_objects_mode(event, data, mode)),
                id = my_id)
            window.Bind(
                wx.EVT_UPDATE_UI,
                (lambda event, data = data, mode=mode:
                 self.__on_update_objects_mode(event, data, mode)),
                id=my_id)
        sub_menu.AppendSeparator()
        self.__add_color_item(
            event, sub_menu, data, "Set objects color", "Set objects colormap")
        self.__add_alpha_item(event, sub_menu, data, "Set objects' transparency")
            
    def __on_set_objects_mode(self, event, data, mode):
        data.mode = mode
        self.refresh()
        
    def __on_update_objects_mode(self, event, data, mode):
        event.Check(mode == data.mode)
        
    def __initialize_mask_sub_menu(self, event, sub_menu, data):
        import wx
        assert isinstance(data, MaskData)
        assert isinstance(sub_menu, wx.Menu)
        item = sub_menu.Append(wx.NewId(), self.MI_MODE)
        item.Enable(False)
        window = self.__get_window_from_event(event)
        for label, target in (
            (self.MI_LINES, MODE_LINES),
            (self.MI_OUTLINES, MODE_OUTLINES),
            (self.MI_OVERLAY, MODE_OVERLAY),
            (self.MI_INVERTED, MODE_INVERTED)):
            set_fn = lambda event, data=data, mode=target:\
                self.on_mask_mode(event, data, target)
            my_id = wx.NewId()
            item = sub_menu.AppendRadioItem(my_id, label)
            self.__on_update_mask_mode(item, data, target)
            window.Bind(wx.EVT_MENU, set_fn, id=my_id)
            window.Bind(
                wx.EVT_UPDATE_UI, 
                (lambda event, data=data, target=target:
                 self.__on_update_mask_mode(event, data, target)),
                id = my_id)
        sub_menu.AppendSeparator()
        self.__add_color_item(event, sub_menu, data, "Set mask color", None)
        self.__add_alpha_item(event, sub_menu, data, "Set mask transparency")
        
    def __on_mask_mode(self, event, data, mode):
        data.mode = mode
        self.refresh()
        
    def __on_update_mask_mode(self, event_or_item, data, target):
        '''Update the menu item or UpdateUIEvent's check status
        
        event_or_item - either an UpdateUIEvent or MenuItem or other
              thing that has a Check method
        data - a MaskData whose mode will be checked
        target - the target state
        
        Either checks or unchecks the item or event, depending on whether
        the data and target matches.
        '''
        event_or_item.Check(target == data.mode)
                
    def __on_color_dlg(self, event, msg, data):
        import wx
        assert isinstance(data, ColorMixin)
        color_data = wx.ColourData()
        orig_color = data.color
        r, g, b = [ int(x*255) for x in data.color]
        color_data.SetColour(wx.Colour(r, g, b))
        window = self.__get_window_from_event(event)
        with wx.ColourDialog(window, color_data) as dlg:
            assert isinstance(dlg, wx.ColourDialog)
            dlg.Title = msg
            if dlg.ShowModal() == wx.ID_OK:
                color_data = dlg.GetColourData()
                data.color = (tuple([
                    float(x) / 255 for x in color_data.Colour]))
                self.refresh()

    def __on_colormap_dlg(self, event, msg, data):
        import wx
        assert isinstance(data, ColorMixin)
        old_colormap = data.colormap
        window = self.__get_window_from_event(event)
        with wx.Dialog(window) as dlg:
            assert isinstance(dlg, wx.Dialog)
            dlg.Title = msg
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            choices = sorted(
                [x for x in matplotlib.cm.datad if not x.endswith("_r")])
            choice = wx.Choice(
                dlg, choices = choices)
            choice.SetStringSelection(old_colormap)
            dlg.Sizer.Add(choice, 0, wx.EXPAND | wx.ALL, 10)
            button_sizer = wx.StdDialogButtonSizer()
            dlg.Sizer.Add(button_sizer, 0, wx.EXPAND)
            button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
            button_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
            button_sizer.Realize()
            def on_choice(event, data=data):
                data.colormap = choice.GetStringSelection()
                self.refresh()
            choice.Bind(wx.EVT_CHOICE, on_choice)
            dlg.Fit()
            if dlg.ShowModal() != wx.ID_OK:
                data.colormap = old_colormap
                self.refresh()
            
    def __on_alpha_dlg(self, event, msg, data):
        import wx
        assert isinstance(data, ColorMixin)
        old_alpha = data.alpha
        window = self.__get_window_from_event(event)
        with wx.Dialog(window) as dlg:
            assert isinstance(dlg, wx.Dialog)
            dlg.Title = msg
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            slider = wx.Slider(
                dlg, value = int(old_alpha * 255),
                minValue = 0,
                maxValue = 255,
                style = wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_LABELS)
            slider.SetMinSize((180, slider.GetMinHeight()))
            dlg.Sizer.Add(slider, 0, wx.EXPAND | wx.ALL, 10)
            button_sizer = wx.StdDialogButtonSizer()
            dlg.Sizer.Add(button_sizer, 0, wx.EXPAND)
            button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
            button_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
            button_sizer.Realize()
            def on_slider(event, data=data):
                data.alpha = float(slider.Value) / 255
                self.refresh()
            slider.Bind(wx.EVT_SLIDER, on_slider)
            dlg.Fit()
            if dlg.ShowModal() != wx.ID_OK:
                data = old_alpha
                self.refresh()
            
            
    def __update_sub_menu(self, event, sub_menu, data):
        event.Enable(data.mode != MODE_HIDE)
        event.Text = data.name

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
    from scipy.ndimage import label
    from cellprofiler.cpmath.otsu import otsu
    
    javabridge.start_vm(class_path=bioformats.JARS)
    try:
        app = wx.PySimpleApp()
        figure = matplotlib.figure.Figure()
        images = []
        objects = []
        masks = []
        for i, arg in enumerate(sys.argv[1:]):
            img = bioformats.load_image(arg)
            images.append(
                ImageData("Image %d" % (i+1), img,
                          alpha = 1.0 / (len(sys.argv) - 1),
                          mode = MODE_COLORIZE))
            thresh = otsu(img)
            l, _ = label(img >= thresh, np.ones((3,3), bool))
            outline_color = tuple([int(idx == i) for idx in range(3)])
            objects.append(ObjectsData(
                "Objects %d" % (i+1), [l], 
                outline_color = outline_color,
                mode = MODE_LINES))
            ii = np.linspace(-1, 1, num = img.shape[0])[:, np.newaxis]
            jj = np.linspace(-1, 1, num = img.shape[1])[np.newaxis, :]
            mask = (ii ** (2*i+2) + jj ** (2*i+2)) ** (1.0 / (2*i+2)) < .75
            masks.append(MaskData("Mask %d" % (i+1), mask, 
                                  mode = MODE_LINES,
                                  color = outline_color))
            
        
        artist = CPImageArtist(images = images, objects=objects, masks = masks)
        figure = matplotlib.pyplot.figure()
        ax = figure.add_axes((0.05, 0.05, .9, .9))
        assert isinstance(ax, matplotlib.axes.Axes)
        ax.set_aspect('equal')
        ax.set_xlim(0, images[0].pixel_data.shape[1])
        ax.set_ylim(0, images[0].pixel_data.shape[0])
        ax.add_artist(artist)
        inspector = InspectionTool()
        my_locals = dict([(k, v) for k, v in globals().items() if k.isupper()])
        my_locals['images'] = images
        my_locals['objects'] = objects
        my_locals['masks'] = masks
        for fmt, sequence in (("i%d", images),
                              ("o%d", objects),
                              ("m%d", masks)):
            for i, v in enumerate(sequence):
                my_locals[fmt % (i+1)] = v
        my_locals['draw'] = matplotlib.pyplot.draw
        inspector.Init(locals = my_locals)
        inspector.Show()
        matplotlib.pyplot.draw()
        frame = matplotlib.pyplot.gcf().canvas.GetTopLevelParent()
        menu_bar = wx.MenuBar()
        menu = wx.Menu()
        sub_menu = wx.Menu()
        item = menu.AppendSubMenu(sub_menu, "Subplot")
        menu_bar.Append(menu, "Subplots")
        frame.SetMenuBar(menu_bar)
        artist.add_to_menu(frame, item)
        matplotlib.pyplot.show()
        
    finally:
        javabridge.kill_vm()