'''<b>Flip and rotate</b>: Flips (mirror image) and/or rotates an image.
<hr>

Features that can be measured by this module:
<ul> <li>Rotation - the angle of rotation</li></ul>
'''

__version__="$Revision$"

import matplotlib
import matplotlib.cm
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
import numpy as np
import scipy.ndimage as scind
import wx

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps

FLIP_NONE = 'Do not flip'
FLIP_LEFT_TO_RIGHT = 'Left to right'
FLIP_TOP_TO_BOTTOM = 'Top to bottom'
FLIP_BOTH = 'Left to right and top to bottom'
FLIP_ALL = [FLIP_NONE, FLIP_LEFT_TO_RIGHT, FLIP_TOP_TO_BOTTOM, FLIP_BOTH]

ROTATE_NONE = 'Do not rotate'
ROTATE_ANGLE = 'Enter angle'
ROTATE_COORDINATES = 'Enter coordinates'
ROTATE_MOUSE = 'Use mouse'
ROTATE_ALL = [ROTATE_NONE, ROTATE_ANGLE, ROTATE_COORDINATES, ROTATE_MOUSE]

IO_INDIVIDUALLY = 'Individually'
IO_ONCE = 'Only Once'
IO_ALL = [IO_INDIVIDUALLY, IO_ONCE]

C_HORIZONTALLY = 'horizontally'
C_VERTICALLY = 'vertically'
C_ALL = [C_HORIZONTALLY, C_VERTICALLY]

D_ANGLE = 'angle'

'''Rotation measurement category'''
M_ROTATION_CATEGORY = "Rotation"
'''Rotation measurement format (+ image name)'''
M_ROTATION_F = "%s_%%s"% M_ROTATION_CATEGORY

class FlipAndRotate(cpm.CPModule):
 
    category = 'Image Processing'
    variable_revision_number = 2
    
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
            "What did you call the image you want to transform?", "None")
        self.output_name = cps.ImageNameProvider(
            "What do you want to call the transformed image?", 
            "FlippedOrigBlue")
        self.flip_choice = cps.Choice("How do you want to flip the image?",
                                      FLIP_ALL)
        self.rotate_choice = cps.Choice("How do you want to rotate the image?",
                                        ROTATE_ALL, doc='''
             <ul> <li> Angle - you can provide the numerical angle by which the 
             image should be rotated.</li>
             <li> Coordinates - you can provide the X,Y pixel locations of 
             two points in the image which should be aligned horizontally or 
             vertically.</li> 
             <li> Mouse - CellProfiler will pause so you can select the 
             rotation interactively. You can grab the image by 
             clicking down with the left mouse button and rotate the image by 
             dragging with the mouse. Press the "Done" button on the image 
             after rotating the image appropriately.</li>
             </ul>''')
        
        self.wants_crop = cps.Binary(
            "Would you like to crop away the rotated edges?", True, doc=
             '''When an image is rotated, there will be black space at the 
             corners/edges unless you choose to crop away the incomplete rows 
             and columns of the image. This cropping will produce an image that 
             is not the exact same size as the original, which may affect 
             downstream modules.''')
                
        self.how_often = cps.Choice(
            "Do you want to determine the amount of rotation for each image "
            "individually as you cycle through, or do you want to define it "
            "only once (on the first image) and then apply it to all images?",
            IO_ALL)
        self.first_pixel = cps.Coordinates(
            "What are the coordinates of the top or left pixel?", (0,0))
        self.second_pixel = cps.Coordinates(
            "What are the coordinates of the bottom or right pixel?", (0,100))
        self.horiz_or_vert = cps.Choice(
            "Are the points horizontally or vertically aligned?",
            C_ALL)
        self.angle = cps.Float(
            "By what angle would you like to rotate the image "
            "(in degrees, positive = counterclockwise and "
            "negative = clockwise)?", 0)
    
    def settings(self):
        return [self.image_name, self.output_name, self.flip_choice,
                self.rotate_choice, self.wants_crop, self.how_often,
                self.first_pixel, self.second_pixel, self.horiz_or_vert,
                self.angle]
    
    def visible_settings(self):
        result = [self.image_name, self.output_name, self.flip_choice,
                  self.rotate_choice]
        if self.rotate_choice == ROTATE_NONE:
            pass
        elif self.rotate_choice == ROTATE_ANGLE:
            result += [self.wants_crop, self.angle]
        elif self.rotate_choice == ROTATE_COORDINATES:
            result += [self.wants_crop, self.first_pixel, self.second_pixel,
                       self.horiz_or_vert]
        elif self.rotate_choice == ROTATE_MOUSE:
            result += [self.wants_crop, self.how_often]
        else:
            raise NotImplementedError("Unimplemented rotation choice: %s"%
                                      self.rotate_choice.value)
        return result
    
    def backwards_compatibilize(self,setting_values,variable_revision_number,
                               module_name,from_matlab):
        if from_matlab and variable_revision_number == 1:
            if setting_values[2] == cps.YES:
                if setting_values[3] == cps.YES:
                    flip_choice = FLIP_BOTH
                else:
                    flip_choice = FLIP_LEFT_TO_RIGHT
            elif setting_values[3] == cps.YES:
                flip_choice = FLIP_TOP_TO_BOTTOM
            else:
                flip_choice = FLIP_NONE
            setting_values = [
                setting_values[0],       # image_name
                setting_values[1],       # output_name
                flip_choice,
                setting_values[4],       # rotate_choice
                setting_values[5],       # wants crop
                setting_values[6],       # how often
                setting_values[8],       # first_pixel
                setting_values[9],       # second_pixel
                setting_values[7],       # horiz_or_vert
                setting_values[10]]      # angle
            from_matlab = False
            variable_revision_number = 1
        if (not from_matlab) and variable_revision_number == 1:
            # Text for ROTATE_MOUSE changed from "mouse" to "Use mouse"
            if setting_values[3] == "Mouse":
                setting_values[3] = ROTATE_MOUSE
            elif setting_values[3] == "None":
                setting_values[3] = ROTATE_NONE
            elif setting_values[3] == "Coordinates":
                setting_values[3] = ROTATE_COORDINATES
            elif setting_values[3] == "Angle":
                setting_values[3] = ROTATE_ANGLE
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

    def prepare_group(self, pipeline, image_set_list, grouping,
                      image_numbers):
        '''Initialize the angle if appropriate'''
        if self.rotate_choice == ROTATE_MOUSE and self.how_often == IO_ONCE:
            self.get_dictionary(image_set_list)[D_ANGLE] = None
        
    def run(self, workspace):
        image_set = workspace.image_set
        assert isinstance(image_set, cpi.ImageSet)
        image = image_set.get_image(self.image_name.value)
        pixel_data = image.pixel_data.copy()
        mask = image.mask
        
        if self.flip_choice != FLIP_NONE:
            if self.flip_choice == FLIP_LEFT_TO_RIGHT:
                i,j = np.mgrid[0:pixel_data.shape[0], 
                               pixel_data.shape[1]-1:-1:-1]
            elif self.flip_choice == FLIP_TOP_TO_BOTTOM:
                i,j = np.mgrid[pixel_data.shape[0]-1:-1:-1,
                               0:pixel_data.shape[1]]
            elif self.flip_choice == FLIP_BOTH:
                i,j = np.mgrid[pixel_data.shape[0]-1:-1:-1,
                               pixel_data.shape[1]-1:-1:-1]
            else:
                raise NotImplementedError("Unknown flipping operation: %s" %
                                          self.flip_choice.value)
            mask = mask[i,j]
            if pixel_data.ndim == 2:
                pixel_data = pixel_data[i,j]
            else:
                pixel_data = pixel_data[i,j,:]
                
        if self.rotate_choice != ROTATE_NONE:
            if self.rotate_choice == ROTATE_ANGLE:
                angle = self.angle.value * np.pi / 180.0
            elif self.rotate_choice == ROTATE_COORDINATES:
                xdiff = self.second_pixel.x - self.first_pixel.x
                ydiff = self.second_pixel.y - self.first_pixel.y
                if self.horiz_or_vert == C_VERTICALLY:
                    angle = np.arctan2(ydiff, xdiff)
                elif self.horiz_or_vert == C_HORIZONTALLY:
                    angle = -np.arctan2(xdiff, ydiff)
                else:
                    raise NotImplementedError("Unknown axis: %s" %
                                              self.horiz_or_vert.value)
            elif self.rotate_choice == ROTATE_MOUSE:
                angle = self.angle_from_mouse(workspace, pixel_data)
            else:
                raise NotImplementedError("Unknown rotation method: %s" %
                                          self.rotate_choice.value)
            transform = np.array([[np.cos(angle),-np.sin(angle)],
                                  [np.sin(angle),np.cos(angle)]])
            # Make it rotate about the center
            offset = affine_offset(pixel_data.shape, transform)
            mask = scind.affine_transform(mask.astype(float), 
                                          transform, offset) > .5
            crop = scind.affine_transform(np.ones(pixel_data.shape[:2]),
                                          transform, offset) > .5
            mask = mask & crop
            
            if pixel_data.ndim == 2:
                pixel_data = scind.affine_transform(pixel_data, transform, 
                                                    offset)
            else:
                for k in range(pixel_data.shape[2]):
                    pixel_data[:,:,k] = scind.affine_transform(pixel_data[:,:,k],
                                                               transform,
                                                               offset)
            if self.wants_crop.value:
                i,j = pixel_data.shape[0:2]
                i_crop = np.ceil(np.abs((i*np.sin(angle) - j*np.cos(angle)) * 
                                        np.cos(angle) * np.sin(angle) /
                                        (np.sin(angle)**2-np.cos(angle)**2)))
                j_crop = np.ceil(np.abs((j*np.sin(angle) - i*np.cos(angle)) * 
                                        np.cos(angle)*np.sin(angle) /
                                        (np.sin(angle)**2-np.cos(angle)**2)));
                ii = np.index_exp[i_crop:i-i_crop,j_crop:j-j_crop]
                crop = np.zeros(pixel_data.shape, bool)
                crop[ii] = True
                mask = mask[ii]
                pixel_data = pixel_data[ii]
            else:
                crop = None
        else:
            crop = None
            angle = 0
        output_image = cpi.Image(pixel_data, mask, crop, image)
        image_set.add(self.output_name.value, output_image)
        workspace.measurements.add_image_measurement(
            M_ROTATION_F % self.output_name.value, angle * 180.0 / np.pi)
        
        if workspace.frame is not None:
            figure = workspace.create_or_find_figure(subplots=(2,1))
            if pixel_data.ndim == 2:
                figure.subplot_imshow_grayscale(0,0, image.pixel_data,
                                                title = self.image_name.value)
                figure.subplot_imshow_grayscale(1,0, output_image.pixel_data,
                                                title = self.output_name.value)
            else:
                figure.subplot_imshow_color(0,0, image.pixel_data,
                                            title = self.image_name.value)
                figure.subplot_imshow_color(1,0, output_image.pixel_data,
                                            title = self.output_name.value)
    def angle_from_mouse(self, workspace, pixel_data):
        '''Run a UI that gets an angle from the user'''
        if self.how_often == IO_ONCE:
            d = self.get_dictionary(workspace.image_set_list)
            if d.has_key(D_ANGLE):
                return d[D_ANGLE]
        
        if pixel_data.ndim == 2:
            # make a color matrix for consistency
            pixel_data = np.dstack((pixel_data,pixel_data,pixel_data))
        pd_min = np.min(pixel_data)
        pd_max = np.max(pixel_data)
        if pd_min == pd_max:
            pixel_data[:,:,:] = 0
        else:
            pixel_data = ((pixel_data - pd_min) * 255.0 / (pd_max - pd_min))
        #
        # Make a dialog box that contains the image
        #
        dialog = wx.Dialog(workspace.frame,
                           title = "Rotate image")
        sizer = wx.BoxSizer(wx.VERTICAL)
        dialog.SetSizer(sizer)
        sizer.Add(wx.StaticText(dialog,label = "Drag image to rotate, hit OK to continue"),
                  0,wx.ALIGN_CENTER_HORIZONTAL)
        canvas = wx.StaticBitmap(dialog)
        canvas.SetDoubleBuffered(True)
        canvas.BackgroundColour = wx.Colour(0,0,0,wx.ALPHA_TRANSPARENT)
        sizer.Add(canvas, 0,
                  wx.ALIGN_CENTER_HORIZONTAL|
                  wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)
        angle = [ 0 ]
        def imshow():
            transform = np.array([[np.cos(angle[0]),-np.sin(angle[0])],
                                  [np.sin(angle[0]),np.cos(angle[0])]])
            # Make it rotate about the center
            offset = affine_offset(pixel_data.shape, transform)
            x = np.dstack((scind.affine_transform(pixel_data[:,:,0], transform,
                                                  offset, order=0),
                           scind.affine_transform(pixel_data[:,:,1], transform, 
                                                  offset, order=0),
                           scind.affine_transform(pixel_data[:,:,2], transform, 
                                                  offset, order=0)))
            buff = x.astype(np.uint8).tostring()
            bitmap = wx.BitmapFromBuffer(x.shape[0],
                                         x.shape[1],
                                         buff)
            canvas.SetBitmap(bitmap)
        imshow()
        #
        # Install handlers for mouse down, mouse move and mouse up
        #
        dragging = [False]
        initial_angle = [0]
        hand_cursor = wx.StockCursor(wx.CURSOR_HAND)
        arrow_cursor = wx.StockCursor(wx.CURSOR_ARROW)
        def get_angle(event):
            center = np.array(canvas.Size) / 2
            point = np.array(event.GetPositionTuple())
            offset = point - center
            return np.arctan2(offset[1],offset[0])
        
        def on_mouse_down(event):
            canvas.Cursor = hand_cursor
            dragging[0] = True
            initial_angle[0] = get_angle(event) - angle[0]
            canvas.CaptureMouse()
        wx.EVT_LEFT_DOWN(canvas, on_mouse_down)
        def on_mouse_up(event):
            if dragging[0]:
                canvas.ReleaseMouse()
                dragging[0] = False
                canvas.Cursor = arrow_cursor
        wx.EVT_LEFT_UP(canvas, on_mouse_up)
        def on_mouse_lost(event):
            dragging[0] = False
            canvas.Cursor = arrow_cursor
        wx.EVT_MOUSE_CAPTURE_LOST(canvas, on_mouse_lost)
        def on_mouse_move(event):
            if dragging[0]:
                angle[0] = get_angle(event) - initial_angle[0]
                imshow()
                canvas.Refresh(eraseBackground=False)
        wx.EVT_MOTION(canvas, on_mouse_move)
        #
        # Put the OK and Cancel buttons on the bottom
        #
        btnsizer = wx.StdDialogButtonSizer()
        
        btn = wx.Button(dialog, wx.ID_OK)
        btn.SetDefault()
        btnsizer.AddButton(btn)

        btn = wx.Button(dialog, wx.ID_CANCEL)
        btnsizer.AddButton(btn)
        btnsizer.Realize()

        sizer.Add(btnsizer, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5)
        dialog.Fit()
        result = dialog.ShowModal()
        if result == wx.ID_OK:
            return angle[0]
    
    def get_measurement_columns(self, pipeline):
        return [(cpmeas.IMAGE, M_ROTATION_F % self.output_name.value,
                 cpmeas.COLTYPE_FLOAT)]
    
    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return [ M_ROTATION_CATEGORY ]
        return []
    
    def get_measurements(self, pipeline, object_name, category):
        if object_name != cpmeas.IMAGE or category != M_ROTATION_CATEGORY:
            return []
        return [self.output_name.value]
        
def affine_offset(shape, transform):
    '''Calculate an offset given an array's shape and an affine transform
    
    shape - the shape of the array to be transformed
    transform - the transform to be performed
    
    Return an offset for scipy.ndimage.affine_transform that does not
    transform the location of the center of the image (the image rotates
    or is flipped about the center).
    '''
    c = (np.array(shape[:2])-1).astype(float)/2.0
    return -np.dot(transform - np.identity(2), c)
    