# coding=utf-8

"""
FlipAndRotate
=============

**FlipAndRotate** flips (mirror image) and/or rotates an image

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Rotation:* Angle of rotation for the input image.
"""

import numpy as np
import scipy.ndimage as scind

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO

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
M_ROTATION_F = "%s_%%s" % M_ROTATION_CATEGORY


class FlipAndRotate(cpm.Module):
    category = 'Image Processing'
    variable_revision_number = 2
    module_name = 'FlipAndRotate'

    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
            "Select the input image",
            cps.NONE,
            doc="Choose the image you want to flip or rotate."
        )

        self.output_name = cps.ImageNameProvider(
            "Name the output image",
            "FlippedOrigBlue",
            doc="Provide a name for the transformed image."
        )

        self.flip_choice = cps.Choice(
                "Select method to flip image",
                FLIP_ALL, doc="""\
Select how the image is to be flipped.""")

        self.rotate_choice = cps.Choice(
                "Select method to rotate image",
                ROTATE_ALL, doc="""\
-  *%(ROTATE_NONE)s:* Leave the image unrotated. This should be used if
   you want to flip the image only.
-  *%(ROTATE_ANGLE)s:* Provide the numerical angle by which the image
   should be rotated.
-  *%(ROTATE_COORDINATES)s:* Provide the X,Y pixel locations of two
   points in the image that should be aligned horizontally or
   vertically.
-  *%(ROTATE_MOUSE)s:* CellProfiler will pause so you can select the
   rotation interactively. When prompted during the analysis run, grab
   the image by clicking the left mouse button, rotate the image by
   dragging with the mouse, then release the mouse button. Press the
   *Done* button on the image after rotating the image appropriately.
""" % globals())

        self.wants_crop = cps.Binary(
                "Crop away the rotated edges?", True, doc="""\
*(Used only when rotating images)*

When an image is rotated, there will be black space at the
corners/edges; select *%(YES)s* to crop away the incomplete rows and
columns of the image, or select *%(NO)s* to leave it as-is.

This cropping will produce an image that is not exactly the same size as
the original, which may affect downstream modules.
""" % globals())

        self.how_often = cps.Choice("Calculate rotation",
                                    IO_ALL, doc="""\
*(Used only when using “%(ROTATE_MOUSE)s” to rotate images)*

Select the cycle(s) at which the calculation is requested and
calculated.
-  *%(IO_INDIVIDUALLY)s:* Determine the amount of rotation for each image individually, e.g., for each cycle.
-  *%(IO_ONCE)s:* Define the rotation only once (on the first image), then apply it to all images.
""" % globals())

        self.first_pixel = cps.Coordinates(
            "Enter coordinates of the top or left pixel",
            (0, 0),
            doc="""\
*(Used only when using {ROTATE_COORDINATES} to rotate images)*

After rotation, if the specified points are aligned horizontally, this point on the image will be positioned to the
left of the other point. If the specified points are aligned vertically, this point of the image will be positioned
above the other point.
""".format(**{
                "ROTATE_COORDINATES": ROTATE_COORDINATES
            })
        )

        self.second_pixel = cps.Coordinates(
            "Enter the coordinates of the bottom or right pixel",
            (0, 100),
            doc="""\
*(Used only when using {ROTATE_COORDINATES} to rotate images)*

After rotation, if the specified points are aligned horizontally, this point on the image will be positioned to the
right of the other point. If the specified points are aligned vertically, this point of the image will be positioned
below the other point.
""".format(**{
                "ROTATE_COORDINATES": ROTATE_COORDINATES
            })
        )

        self.horiz_or_vert = cps.Choice(
                "Select how the specified points should be aligned",
                C_ALL, doc="""\
*(Used only when using “%(ROTATE_COORDINATES)s” to rotate images)*

Specify whether you would like the coordinate points that you entered to
be horizontally or vertically aligned after the rotation is complete.""" % globals())

        self.angle = cps.Float(
                "Enter angle of rotation", 0, doc="""\
*(Used only when using “%(ROTATE_ANGLE)s” to rotate images)*

Enter the angle you would like to rotate the image. This setting is in
degrees, with positive angles corresponding to counterclockwise and
negative as clockwise.""" % globals())

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
            raise NotImplementedError("Unimplemented rotation choice: %s" %
                                      self.rotate_choice.value)
        return result

    def prepare_group(self, workspace, grouping,
                      image_numbers):
        '''Initialize the angle if appropriate'''
        if self.rotate_choice == ROTATE_MOUSE and self.how_often == IO_ONCE:
            self.get_dictionary(workspace.image_set_list)[D_ANGLE] = None

    def run(self, workspace):
        image_set = workspace.image_set
        image = image_set.get_image(self.image_name.value)
        pixel_data = image.pixel_data.copy()
        mask = image.mask

        if self.flip_choice != FLIP_NONE:
            if self.flip_choice == FLIP_LEFT_TO_RIGHT:
                i, j = np.mgrid[0:pixel_data.shape[0],
                       pixel_data.shape[1] - 1:-1:-1]
            elif self.flip_choice == FLIP_TOP_TO_BOTTOM:
                i, j = np.mgrid[pixel_data.shape[0] - 1:-1:-1,
                       0:pixel_data.shape[1]]
            elif self.flip_choice == FLIP_BOTH:
                i, j = np.mgrid[pixel_data.shape[0] - 1:-1:-1,
                       pixel_data.shape[1] - 1:-1:-1]
            else:
                raise NotImplementedError("Unknown flipping operation: %s" %
                                          self.flip_choice.value)
            mask = mask[i, j]
            if pixel_data.ndim == 2:
                pixel_data = pixel_data[i, j]
            else:
                pixel_data = pixel_data[i, j, :]

        if self.rotate_choice != ROTATE_NONE:
            if self.rotate_choice == ROTATE_ANGLE:
                angle = self.angle.value
            elif self.rotate_choice == ROTATE_COORDINATES:
                xdiff = self.second_pixel.x - self.first_pixel.x
                ydiff = self.second_pixel.y - self.first_pixel.y
                if self.horiz_or_vert == C_VERTICALLY:
                    angle = -np.arctan2(ydiff, xdiff) * 180.0 / np.pi
                elif self.horiz_or_vert == C_HORIZONTALLY:
                    angle = np.arctan2(xdiff, ydiff) * 180.0 / np.pi
                else:
                    raise NotImplementedError("Unknown axis: %s" %
                                              self.horiz_or_vert.value)
            elif self.rotate_choice == ROTATE_MOUSE:
                d = self.get_dictionary()
                if (self.how_often == IO_ONCE and d.has_key(D_ANGLE) and
                            d[D_ANGLE] is not None):
                    angle = d[D_ANGLE]
                else:
                    angle = workspace.interaction_request(
                            self, pixel_data)
                if self.how_often == IO_ONCE:
                    d[D_ANGLE] = angle
            else:
                raise NotImplementedError("Unknown rotation method: %s" %
                                          self.rotate_choice.value)
            rangle = angle * np.pi / 180.0
            mask = scind.rotate(mask.astype(float), angle,
                                reshape=True) > .50
            crop = scind.rotate(np.ones(pixel_data.shape[:2]), angle,
                                reshape=True) > .50
            mask = mask & crop
            pixel_data = scind.rotate(pixel_data, angle, reshape=True)
            if self.wants_crop.value:
                #
                # We want to find the largest rectangle that fits inside
                # the crop. The cumulative sum in the i and j direction gives
                # the length of the rectangle in each direction and
                # multiplying them gives you the area.
                #
                # The left and right halves are symmetric, so we compute
                # on just two of the quadrants.
                #
                half = (np.array(crop.shape) / 2).astype(int)
                #
                # Operate on the lower right
                #
                quartercrop = crop[half[0]:, half[1]:]
                ci = np.cumsum(quartercrop, 0)
                cj = np.cumsum(quartercrop, 1)
                carea_d = ci * cj
                carea_d[quartercrop == 0] = 0
                #
                # Operate on the upper right by flipping I
                #
                quartercrop = crop[crop.shape[0] - half[0] - 1::-1, half[1]:]
                ci = np.cumsum(quartercrop, 0)
                cj = np.cumsum(quartercrop, 1)
                carea_u = ci * cj
                carea_u[quartercrop == 0] = 0
                carea = carea_d + carea_u
                max_carea = np.max(carea)
                max_area = np.argwhere(carea == max_carea)[0] + half
                min_i = max(crop.shape[0] - max_area[0] - 1, 0)
                max_i = max_area[0] + 1
                min_j = max(crop.shape[1] - max_area[1] - 1, 0)
                max_j = max_area[1] + 1
                ii = np.index_exp[min_i:max_i, min_j:max_j]
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
                M_ROTATION_F % self.output_name.value, angle)

        vmin = min(np.min(image.pixel_data),
                   np.min(output_image.pixel_data[output_image.mask]))
        vmax = max(np.max(image.pixel_data),
                   np.max(output_image.pixel_data[output_image.mask]))
        workspace.display_data.image_pixel_data = image.pixel_data
        workspace.display_data.output_image_pixel_data = output_image.pixel_data
        workspace.display_data.vmin = vmin
        workspace.display_data.vmax = vmax

    def display(self, workspace, figure):
        image_pixel_data = workspace.display_data.image_pixel_data
        output_image_pixel_data = workspace.display_data.output_image_pixel_data
        vmin = workspace.display_data.vmin
        vmax = workspace.display_data.vmax
        figure.set_subplots((2, 1))
        if vmin == vmax:
            vmin = 0
            vmax = 1
        if output_image_pixel_data.ndim == 2:
            figure.subplot_imshow_grayscale(
                    0, 0, image_pixel_data, title=self.image_name.value,
                    vmin=vmin, vmax=vmax, normalize=False)
            figure.subplot_imshow_grayscale(
                    1, 0, output_image_pixel_data, title=self.output_name.value,
                    vmin=vmin, vmax=vmax, normalize=False,
                    sharexy=figure.subplot(0, 0))
        else:
            figure.subplot_imshow(0, 0, image_pixel_data,
                                  title=self.image_name.value,
                                  normalize=False,
                                  vmin=vmin,
                                  vmax=vmax)
            figure.subplot_imshow(1, 0, output_image_pixel_data,
                                  title=self.output_name.value,
                                  normalize=False,
                                  vmin=vmin,
                                  vmax=vmax,
                                  sharexy=figure.subplot(0, 0))

    def handle_interaction(self, pixel_data):
        '''Run a UI that gets an angle from the user'''
        import wx

        if pixel_data.ndim == 2:
            # make a color matrix for consistency
            pixel_data = np.dstack((pixel_data, pixel_data, pixel_data))
        pd_min = np.min(pixel_data)
        pd_max = np.max(pixel_data)
        if pd_min == pd_max:
            pixel_data[:, :, :] = 0
        else:
            pixel_data = ((pixel_data - pd_min) * 255.0 / (pd_max - pd_min))
        #
        # Make a 100 x 100 image so it's manageable
        #
        isize = 200
        i, j, k = np.mgrid[0:isize,
                  0:int(isize * pixel_data.shape[1] / pixel_data.shape[0]),
                  0:3].astype(float)
        i *= float(pixel_data.shape[0]) / float(isize)
        j *= float(pixel_data.shape[0]) / float(isize)
        pixel_data = scind.map_coordinates(pixel_data, (i, j, k))
        #
        # Make a dialog box that contains the image
        #
        dialog = wx.Dialog(None,
                           title="Rotate image")
        sizer = wx.BoxSizer(wx.VERTICAL)
        dialog.SetSizer(sizer)
        sizer.Add(wx.StaticText(dialog, label="Drag image to rotate, hit OK to continue"),
                  0, wx.ALIGN_CENTER_HORIZONTAL)
        canvas = wx.StaticBitmap(dialog)
        canvas.SetDoubleBuffered(True)
        sizer.Add(canvas, 0,
                  wx.ALIGN_CENTER_HORIZONTAL |
                  wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        angle = [0]
        angle_text = wx.StaticText(dialog, label="Angle: %d" % angle[0])
        sizer.Add(angle_text, 0, wx.ALIGN_CENTER_HORIZONTAL)

        def imshow():
            angle_text.Label = "Angle: %d" % int(angle[0])
            angle_text.Refresh()
            my_angle = -angle[0] * np.pi / 180.0
            transform = np.array([[np.cos(my_angle), -np.sin(my_angle)],
                                  [np.sin(my_angle), np.cos(my_angle)]])
            # Make it rotate about the center
            offset = affine_offset(pixel_data.shape, transform)
            x = np.dstack((scind.affine_transform(pixel_data[:, :, 0], transform,
                                                  offset, order=0),
                           scind.affine_transform(pixel_data[:, :, 1], transform,
                                                  offset, order=0),
                           scind.affine_transform(pixel_data[:, :, 2], transform,
                                                  offset, order=0)))
            buff = x.astype(np.uint8).tostring()
            bitmap = wx.BitmapFromBuffer(x.shape[1],
                                         x.shape[0],
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
            return -np.arctan2(offset[1], offset[0]) * 180.0 / np.pi

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

        sizer.Add(btnsizer, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
        dialog.Fit()
        result = dialog.ShowModal()
        dialog.Destroy()
        if result == wx.ID_OK:
            return angle[0]
        raise ValueError("Canceled by user in FlipAndRotate")

    def get_measurement_columns(self, pipeline):
        return [(cpmeas.IMAGE, M_ROTATION_F % self.output_name.value,
                 cpmeas.COLTYPE_FLOAT)]

    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return [M_ROTATION_CATEGORY]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name != cpmeas.IMAGE or category != M_ROTATION_CATEGORY:
            return []
        return [self.output_name.value]

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if (from_matlab and variable_revision_number == 2 and
                    module_name == "Rotate"):
            image_name, output_name, rotate_method, crop_edges, \
            individual_or_once, horiz_or_vert, pixel1, pixel2, \
            angle = setting_values
            if rotate_method == "Coordinates":
                rotate_method = ROTATE_COORDINATES
            elif rotate_method == "Mouse":
                rotate_method = ROTATE_MOUSE
            else:
                rotate_method = ROTATE_ANGLE
            if horiz_or_vert == "horizontally":
                horiz_or_vert = C_HORIZONTALLY
            else:
                horiz_or_vert = C_VERTICALLY
            if individual_or_once == "Only once":
                individual_or_once = IO_ONCE
            else:
                individual_or_once = IO_INDIVIDUALLY
            setting_values = [image_name, output_name, FLIP_NONE,
                              rotate_method, crop_edges,
                              individual_or_once, pixel1, pixel2,
                              horiz_or_vert, angle]
            variable_revision_number = 2
            from_matlab = False
            module_name = self.module_name
        if (from_matlab and variable_revision_number == 1 and
                    module_name == "Flip"):
            image_name, output_name, left_to_right, top_to_bottom = \
                setting_values
            if left_to_right == cps.YES:
                if top_to_bottom == cps.YES:
                    flip_choice = FLIP_BOTH
                else:
                    flip_choice = FLIP_LEFT_TO_RIGHT
            elif top_to_bottom == cps.YES:
                flip_choice = FLIP_TOP_TO_BOTTOM
            else:
                flip_choice = FLIP_NONE
            setting_values = [image_name, output_name, flip_choice,
                              ROTATE_NONE, cps.NO, "10,10",
                              "100,100", C_VERTICALLY, "0"]
            from_matlab = False
            module_name = self.module_name
            variable_revision_number = 2
        if (from_matlab and variable_revision_number == 1 and
                    module_name == self.module_name):
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
                setting_values[0],  # image_name
                setting_values[1],  # output_name
                flip_choice,
                setting_values[4],  # rotate_choice
                setting_values[5],  # wants crop
                setting_values[6],  # how often
                setting_values[8],  # first_pixel
                setting_values[9],  # second_pixel
                setting_values[7],  # horiz_or_vert
                setting_values[10]]  # angle
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


def affine_offset(shape, transform):
    '''Calculate an offset given an array's shape and an affine transform

    shape - the shape of the array to be transformed
    transform - the transform to be performed

    Return an offset for scipy.ndimage.affine_transform that does not
    transform the location of the center of the image (the image rotates
    or is flipped about the center).
    '''
    c = (np.array(shape[:2]) - 1).astype(float) / 2.0
    return -np.dot(transform - np.identity(2), c)
