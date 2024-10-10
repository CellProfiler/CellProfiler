"""
Padding
==========

**Padding** Add padding of arbitrary value to x or y or add z planes.
Image size can be increased or maintained.
See `this tutorial <http://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html>`__ for more information. 

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

"""

import numpy
import skimage.color
from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting.text import Float, Integer
from cellprofiler_core.setting.choice import Choice


C_SAME_SIZE = "Keep original image size"
C_PAD = "Pad images"

class PadImage(ImageProcessing):
    category = "Advanced"

    module_name = "PadImage"

    variable_revision_number = 1
   
    def create_settings(self):
        super(PadImage, self).create_settings() #reload the class PadImage

        self.pad_value = Float(
            text="Value to pad with",
            value=0,
            minval=0.0,
            maxval=1.0,
            doc="""The value that will be added to the image as part of the padding, 
                    recommended to be 0 for background/ black or 1 for foreground. 
                    High values will lead to rescaling of the image""",
        )
        self.x_pad_choice = Choice(
            text="Pad X",
            choices=['No', 'Left Pad', 'Right Pad'],)
        self.x_axis = Integer(
            text="Pad X axis",
            value=0,
            minval=0,
        )
        self.y_pad_choice = Choice(
            text="Pad Y",
            choices=['No', 'Top Pad', 'Bottom Pad'],)
        self.y_axis =  Integer(
            text="Pad Y axis",
            value=0,
            minval=0,
        )
        self.z_pad_choice = Choice(
            text="Pad Z",
            choices=['No', 'Top Pad', 'Bottom Pad'],
            doc="""If the image is 2D it will not pad the Z even if you put values in this setting""")
        self.z_axis =  Integer(
            text="Pad Z axis",
            value=0,
            minval=0,
        )
        self.crop_mode = Choice(
            text="Crop mode",
            choices=[C_PAD, C_SAME_SIZE],
            doc="""\
The crop mode determines how the output images are either cropped or
padded after pad is applied. There are two choices:

-  *%(C_PAD)s:* Images are padded such that their output size is changed.
-  *%(C_SAME_SIZE)s:* Maintain the sizes of the images. This acts to shift the image
by the input number of pixels (2D) and/or z-planes (3D with z-pad). Note that %(C_SAME_SIZE)s
only works if there is a pad in a single direction per axis (e.g. pad x left OR pad x left and pad y top, NOT pad x right and pad x right)
   """
                % globals(),
        )


    def run(self, workspace):
        x_name = self.x_name.value #input image name selected from dropdown

        y_name = self.y_name.value #output that you type in the box

        x = workspace.image_set.get_image(x_name) #get the specific image

        x_data = x.pixel_data #array of pixel

        # Determine padding settings
        zpad = (0,0)
        xpad = (0,0)
        ypad = (0,0)
        if self.x_pad_choice == 'Left Pad':
            xpad = (self.x_axis.value,0)
        elif self.x_pad_choice == 'Right Pad':
            xpad = (0,self.x_axis.value)
        if self.y_pad_choice == 'Top Pad':
            ypad = (self.y_axis.value,0)
        elif self.y_pad_choice == 'Bottom Pad':
            ypad = (0,self.y_axis.value)
        if self.z_pad_choice == 'Top Pad':
            zpad = (0,self.z_axis.value)
        elif self.z_pad_choice == 'Bottom Pad':
            zpad = (self.z_axis.value,0)

        if x.volumetric:
            y_data = numpy.pad(x_data, pad_width=(zpad, ypad, xpad), constant_values=self.pad_value.value)

            mask = numpy.pad(x.mask, pad_width=(zpad, ypad, xpad), constant_values=self.pad_value.value)

            if x.has_crop_mask:
                y_cropmask = numpy.pad(
                    x.crop_mask, pad_width=(zpad, ypad, xpad), constant_values=self.pad_value.value
                )

        else:
            y_data =  numpy.pad(x_data, pad_width=(ypad, xpad), constant_values=self.pad_value.value)
            mask = numpy.pad(x.mask, pad_width=(ypad, xpad), constant_values=self.pad_value.value)

            if x.has_crop_mask:
                y_cropmask = numpy.pad(
                    x.crop_mask, pad_width=(ypad, xpad), constant_values=self.pad_value.value
                )

        # Crop the padded image
        if self.crop_mode == C_SAME_SIZE:
            # Determine crop shapes
            origshape = x_data.shape
            if self.x_pad_choice == 'Right Pad':
                xmin = self.x_axis.value
                xmax = origshape[0] + self.x_axis.value
            else:
                xmin = 0
                xmax = origshape[0]
            if self.y_pad_choice == 'Bottom Pad':
                ymin = self.y_axis.value
                ymax = origshape[1] + self.x_axis.value
            else:
                ymin = 0
                ymax = origshape[1]

            if x.volumetric:
                if self.z_pad_choice == 'Top Pad':
                    zmin = self.z_axis.value
                    zmax = origshape[2] + self.z_axis.value
                else:
                    zmin = 0
                    zmax = origshape[2]
                y_data = y_data[xmin:xmax, ymin:ymax, zmin:zmax]
                mask = mask[xmin:xmax, ymin:ymax, zmin:zmax]
                if x.has_crop_mask:
                    y_cropmask = y_cropmask[xmin:xmax, ymin:ymax, zmin:zmax]
            else:
                y_data = y_data[xmin:xmax, ymin:ymax]
                mask = mask[xmin:xmax, ymin:ymax]
                if x.has_crop_mask:
                    y_cropmask = y_cropmask[xmin:xmax, ymin:ymax]
        
        if x.has_crop_mask:
            y_cropmask = skimage.img_as_bool(y_cropmask)
        else:
            y_cropmask = None
        mask = skimage.img_as_bool(mask)
        
        # This a class packaging the image in a way that can be used by cellprofiler
        y = Image(
            y_data,
            mask=mask,
            crop_mask=y_cropmask,
            dimensions=y_data.ndim,
        ) 

        # This is adding the image to the workspace, so that it's available for the next module
        workspace.image_set.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = x.dimensions

 
    def settings(self):
        __settings__ = super(PadImage, self).settings()
        __settings__ += [self.pad_value, self.x_pad_choice, self.y_pad_choice, self.z_pad_choice, self.crop_mode]
        __settings__ += [self.x_axis, self.y_axis, self.z_axis]

        return __settings__

    def visible_settings(self):
        __settings__ = super(PadImage, self).visible_settings()
        __settings__ += [self.pad_value, self.x_pad_choice]
        if self.x_pad_choice != 'No':
            __settings__ += [self.x_axis]
        __settings__ += [self.y_pad_choice]
        if self.y_pad_choice != 'No':
            __settings__ += [self.y_axis]
        __settings__ += [self.z_pad_choice]
        if self.z_pad_choice != 'No':
            __settings__ += [self.z_axis]
        __settings__ += [self.crop_mode]

        return __settings__