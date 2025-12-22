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

import numpy
import scipy.ndimage
from enum import Enum
from cellprofiler_core.constants.measurement import IMAGE, COLTYPE_FLOAT
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import Coordinates
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import ImageName, Float
from cellprofiler_library.modules._flipandrotate import flip_and_rotate, flip_image, rotate_image
from cellprofiler_library.opts.flipandrotate import RotateMethod, D_ANGLE, M_ROTATION_CATEGORY, M_ROTATION_F, FLIP_ALL, ROTATE_ALL, C_ALL


class RotationCycle(str, Enum):
    INDIVIDUALLY = "Individually"
    ONCE = "Only Once"

IO_ALL = [RotationCycle.INDIVIDUALLY, RotationCycle.ONCE]

class FlipAndRotate(Module):
    category = "Image Processing"
    variable_revision_number = 2
    module_name = "FlipAndRotate"

    def create_settings(self):
        self.image_name = ImageSubscriber(
            "Select the input image",
            "None",
            doc="Choose the image you want to flip or rotate.",
        )

        self.output_name = ImageName(
            "Name the output image",
            "FlippedOrigBlue",
            doc="Provide a name for the transformed image.",
        )

        self.flip_choice = Choice(
            "Select method to flip image",
            FLIP_ALL,
            doc="""\
Select how the image is to be flipped.""",
        )

        self.rotate_choice = Choice(
            "Select method to rotate image",
            ROTATE_ALL,
            doc="""\
-  *{ROTATE_NONE}:* Leave the image unrotated. This should be used if
   you want to flip the image only.
-  *{ROTATE_ANGLE}:* Provide the numerical angle by which the image
   should be rotated.
-  *{ROTATE_COORDINATES}:* Provide the X,Y pixel locations of two
   points in the image that should be aligned horizontally or
   vertically.
-  *{ROTATE_MOUSE}:* CellProfiler will pause so you can select the
   rotation interactively. When prompted during the analysis run, grab
   the image by clicking the left mouse button, rotate the image by
   dragging with the mouse, then release the mouse button. Press the
   *Done* button on the image after rotating the image appropriately.
""".format(
    **{
        "ROTATE_NONE": RotateMethod.NONE.value,
        "ROTATE_ANGLE": RotateMethod.ANGLE.value,
        "ROTATE_COORDINATES": RotateMethod.COORDINATES.value,
        "ROTATE_MOUSE": RotateMethod.MOUSE.value,
    }
),
        )

        self.wants_crop = Binary(
            "Crop away the rotated edges?",
            True,
            doc="""\
*(Used only when rotating images)*

When an image is rotated, there will be black space at the
corners/edges; select *Yes* to crop away the incomplete rows and
columns of the image, or select *No* to leave it as-is.

This cropping will produce an image that is not exactly the same size as
the original, which may affect downstream modules.
"""
            % globals(),
        )

        self.how_often = Choice(
            "Calculate rotation",
            IO_ALL,
            doc="""\
*(Used only when using “{ROTATE_MOUSE}” to rotate images)*

Select the cycle(s) at which the calculation is requested and
calculated.
-  *{IO_INDIVIDUALLY}:* Determine the amount of rotation for each image individually, e.g., for each cycle.
-  *{IO_ONCE}:* Define the rotation only once (on the first image), then apply it to all images.
""".format(
    **{
        "ROTATE_MOUSE": RotateMethod.MOUSE.value,
        "IO_INDIVIDUALLY": RotationCycle.INDIVIDUALLY.value,
        "IO_ONCE": RotationCycle.ONCE.value,
    }
),
        )

        self.first_pixel = Coordinates(
            "Enter coordinates of the top or left pixel",
            (0, 0),
            doc="""\
*(Used only when using {ROTATE_COORDINATES} to rotate images)*

After rotation, if the specified points are aligned horizontally, this point on the image will be positioned to the
left of the other point. If the specified points are aligned vertically, this point of the image will be positioned
above the other point.
""".format(
                **{"ROTATE_COORDINATES": RotateMethod.COORDINATES.value}
            ),
        )

        self.second_pixel = Coordinates(
            "Enter the coordinates of the bottom or right pixel",
            (0, 100),
            doc="""\
*(Used only when using {ROTATE_COORDINATES} to rotate images)*

After rotation, if the specified points are aligned horizontally, this point on the image will be positioned to the
right of the other point. If the specified points are aligned vertically, this point of the image will be positioned
below the other point.
""".format(
                **{"ROTATE_COORDINATES": RotateMethod.COORDINATES.value}
            ),
        )

        self.horiz_or_vert = Choice(
            "Select how the specified points should be aligned",
            C_ALL,
            doc="""\
*(Used only when using “{ROTATE_COORDINATES}” to rotate images)*

Specify whether you would like the coordinate points that you entered to
be horizontally or vertically aligned after the rotation is complete.""".format(
    **{"ROTATE_COORDINATES": RotateMethod.COORDINATES.value}
),
        )

        self.angle = Float(
            "Enter angle of rotation",
            0,
            doc="""\
*(Used only when using “{ROTATE_ANGLE}” to rotate images)*

Enter the angle you would like to rotate the image. This setting is in
degrees, with positive angles corresponding to counterclockwise and
negative as clockwise.""".format(
    **{"ROTATE_ANGLE": RotateMethod.ANGLE.value}
),
        )

    def settings(self):
        return [
            self.image_name,
            self.output_name,
            self.flip_choice,
            self.rotate_choice,
            self.wants_crop,
            self.how_often,
            self.first_pixel,
            self.second_pixel,
            self.horiz_or_vert,
            self.angle,
        ]

    def visible_settings(self):
        result = [
            self.image_name,
            self.output_name,
            self.flip_choice,
            self.rotate_choice,
        ]
        if self.rotate_choice == RotateMethod.NONE:
            pass
        elif self.rotate_choice == RotateMethod.ANGLE:
            result += [self.wants_crop, self.angle]
        elif self.rotate_choice == RotateMethod.COORDINATES:
            result += [
                self.wants_crop,
                self.first_pixel,
                self.second_pixel,
                self.horiz_or_vert,
            ]
        elif self.rotate_choice == RotateMethod.MOUSE:
            result += [self.wants_crop, self.how_often]
        else:
            raise NotImplementedError(
                "Unimplemented rotation choice: %s" % self.rotate_choice.value
            )
        return result

    def prepare_group(self, workspace, grouping, image_numbers):
        """Initialize the angle if appropriate"""
        if self.rotate_choice == RotateMethod.MOUSE and self.how_often == RotationCycle.ONCE:
            self.get_dictionary(workspace.image_set_list)[D_ANGLE] = None

    def run(self, workspace):
        image_set = workspace.image_set
        image = image_set.get_image(self.image_name.value)
        pixel_data = image.pixel_data.copy()
        mask = image.mask


        ######
        rotate_angle = self.angle.value
        state_dict_for_mouse_mode = self.get_dictionary()
        mouse_mode_cycle = self.how_often.value
        
        if self.rotate_choice == RotateMethod.MOUSE:
            # perform flip and rotate separately
            pixel_data, mask = flip_image(pixel_data, mask, self.flip_choice.value)
            # state_dict_for_mouse_mode = self.get_dictionary()
            assert state_dict_for_mouse_mode is not None, "state_dict_for_mouse_mode must be provided for rotate_choice == RotateMethod.MOUSE"
            assert mouse_mode_cycle is not None, "mouse_mode_cycle must be provided for rotate_choice == RotateMethod.MOUSE"
            if (
                mouse_mode_cycle == RotationCycle.ONCE
                and D_ANGLE in state_dict_for_mouse_mode
                and state_dict_for_mouse_mode[D_ANGLE] is not None
            ):
                angle = state_dict_for_mouse_mode[D_ANGLE]
            else:
                angle = workspace.interaction_request(
                self, pixel_data, workspace.measurements.image_set_number
            )
            if mouse_mode_cycle == RotationCycle.ONCE:
                state_dict_for_mouse_mode[D_ANGLE] = angle
            pixel_data, mask, crop, angle = rotate_image(
                pixel_data,
                mask,
                RotateMethod.ANGLE,
                angle,
                None,
                None,
                None,
                wants_crop=self.wants_crop.value,
            )

        else:
            pixel_data, mask, crop, angle = flip_and_rotate(
                pixel_data, 
                mask, 
                self.flip_choice.value, 
                self.rotate_choice.value, 
                rotate_angle, 
                (self.first_pixel.x, self.first_pixel.y),
                (self.second_pixel.x, self.second_pixel.y), 
                self.horiz_or_vert,
                wants_crop=self.wants_crop.value,
            )
        output_image = Image(pixel_data, mask, crop, image)
        image_set.add(self.output_name.value, output_image)
        workspace.measurements.add_image_measurement(
            M_ROTATION_F % self.output_name.value, angle
        )

        vmin = min(
            numpy.min(image.pixel_data),
            numpy.min(output_image.pixel_data[output_image.mask]),
        )
        vmax = max(
            numpy.max(image.pixel_data),
            numpy.max(output_image.pixel_data[output_image.mask]),
        )
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
                0,
                0,
                image_pixel_data,
                title=self.image_name.value,
                vmin=vmin,
                vmax=vmax,
                normalize=False,
            )
            figure.subplot_imshow_grayscale(
                1,
                0,
                output_image_pixel_data,
                title=self.output_name.value,
                vmin=vmin,
                vmax=vmax,
                normalize=False,
                sharexy=figure.subplot(0, 0),
            )
        else:
            figure.subplot_imshow(
                0,
                0,
                image_pixel_data,
                title=self.image_name.value,
                normalize=False,
                vmin=vmin,
                vmax=vmax,
            )
            figure.subplot_imshow(
                1,
                0,
                output_image_pixel_data,
                title=self.output_name.value,
                normalize=False,
                vmin=vmin,
                vmax=vmax,
                sharexy=figure.subplot(0, 0),
            )

    def handle_interaction(self, pixel_data, image_set_number):
        """Run a UI that gets an angle from the user"""
        import wx

        if pixel_data.ndim == 2:
            # make a color matrix for consistency
            pixel_data = numpy.dstack((pixel_data, pixel_data, pixel_data))
        pd_min = numpy.min(pixel_data)
        pd_max = numpy.max(pixel_data)
        if pd_min == pd_max:
            pixel_data[:, :, :] = 0
        else:
            pixel_data = (pixel_data - pd_min) * 255.0 / (pd_max - pd_min)
        #
        # Make a 100 x 100 image so it's manageable
        #
        isize = 200
        i, j, k = numpy.mgrid[
            0:isize, 0 : int(isize * pixel_data.shape[1] / pixel_data.shape[0]), 0:3
        ].astype(float)
        i *= float(pixel_data.shape[0]) / float(isize)
        j *= float(pixel_data.shape[0]) / float(isize)
        pixel_data = scipy.ndimage.map_coordinates(pixel_data, (i, j, k))
        #
        # Make a dialog box that contains the image
        #
        dialog_title = "Rotate image - Cycle #%d:" % (image_set_number)
        dialog = wx.Dialog(None, title=dialog_title)
        sizer = wx.BoxSizer(wx.VERTICAL)
        dialog.SetSizer(sizer)
        sizer.Add(
            wx.StaticText(dialog, label="Drag image to rotate, hit OK to continue"),
            0,
            wx.ALIGN_CENTER_HORIZONTAL,
        )
        canvas = wx.StaticBitmap(dialog)
        canvas.SetDoubleBuffered(True)
        sizer.Add(
            canvas, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5
        )
        angle = [0]
        angle_text = wx.StaticText(dialog, label="Angle: %d" % angle[0])
        sizer.Add(angle_text, 0, wx.ALIGN_CENTER_HORIZONTAL)

        def imshow():
            angle_text.Label = "Angle: %d" % int(angle[0])
            angle_text.Refresh()
            my_angle = -angle[0] * numpy.pi / 180.0
            transform = numpy.array(
                [
                    [numpy.cos(my_angle), -numpy.sin(my_angle)],
                    [numpy.sin(my_angle), numpy.cos(my_angle)],
                ]
            )
            # Make it rotate about the center
            offset = affine_offset(pixel_data.shape, transform)
            x = numpy.dstack(
                (
                    scipy.ndimage.affine_transform(
                        pixel_data[:, :, 0], transform, offset, order=0
                    ),
                    scipy.ndimage.affine_transform(
                        pixel_data[:, :, 1], transform, offset, order=0
                    ),
                    scipy.ndimage.affine_transform(
                        pixel_data[:, :, 2], transform, offset, order=0
                    ),
                )
            )
            buff = x.astype(numpy.uint8).tostring()
            bitmap = wx.Bitmap.FromBuffer(x.shape[1], x.shape[0], buff)
            canvas.SetBitmap(bitmap)

        imshow()
        #
        # Install handlers for mouse down, mouse move and mouse up
        #
        dragging = [False]
        initial_angle = [0]
        hand_cursor = wx.Cursor(wx.CURSOR_HAND)
        arrow_cursor = wx.Cursor(wx.CURSOR_ARROW)

        def get_angle(event):
            center = numpy.array(canvas.Size) / 2
            point = numpy.array(event.GetPosition())
            offset = point - center
            return -numpy.arctan2(offset[1], offset[0]) * 180.0 / numpy.pi

        def on_mouse_down(event):
            canvas.Cursor = hand_cursor
            dragging[0] = True
            initial_angle[0] = get_angle(event) - angle[0]
            canvas.CaptureMouse()

        canvas.Bind(wx.EVT_LEFT_DOWN, on_mouse_down)

        def on_mouse_up(event):
            if dragging[0]:
                canvas.ReleaseMouse()
                dragging[0] = False
                canvas.Cursor = arrow_cursor

        canvas.Bind(wx.EVT_LEFT_UP, on_mouse_up)

        def on_mouse_lost(event):
            dragging[0] = False
            canvas.Cursor = arrow_cursor

        canvas.Bind(wx.EVT_MOUSE_CAPTURE_LOST, on_mouse_lost)

        def on_mouse_move(event):
            if dragging[0]:
                angle[0] = get_angle(event) - initial_angle[0]
                imshow()
                canvas.Refresh(eraseBackground=False)

        canvas.Bind(wx.EVT_MOTION, on_mouse_move)
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
        return [(IMAGE, M_ROTATION_F % self.output_name.value, COLTYPE_FLOAT)]

    def get_categories(self, pipeline, object_name):
        if object_name == IMAGE:
            return [M_ROTATION_CATEGORY]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name != IMAGE or category != M_ROTATION_CATEGORY:
            return []
        return [self.output_name.value]

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            # Text for ROTATE_MOUSE changed from "mouse" to "Use mouse"
            if setting_values[3] == "Mouse":
                setting_values[3] = RotateMethod.MOUSE
            elif setting_values[3] == "None":
                setting_values[3] = RotateMethod.NONE
            elif setting_values[3] == "Coordinates":
                setting_values[3] = RotateMethod.COORDINATES
            elif setting_values[3] == "Angle":
                setting_values[3] = RotateMethod.ANGLE
            variable_revision_number = 2
        return setting_values, variable_revision_number


def affine_offset(shape, transform):
    """Calculate an offset given an array's shape and an affine transform

    shape - the shape of the array to be transformed
    transform - the transform to be performed

    Return an offset for scipy.ndimage.affine_transform that does not
    transform the location of the center of the image (the image rotates
    or is flipped about the center).
    """
    c = (numpy.array(shape[:2]) - 1).astype(float) / 2.0
    return -numpy.dot(transform - numpy.identity(2), c)
