"""
Padding
==========

**Padding** Add planes of value 0
See `this tutorial <http://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html>`__ for more information. 

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import numpy
import skimage.color
from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting.text import Float
from cellprofiler_core.setting.range import IntegerRange

class PadImage(ImageProcessing):
    category = "Advanced"

    module_name = "PadImage"

    variable_revision_number = 1
   
    def create_settings(self):
        super(PadImage, self).create_settings()#reload the class PadImage

        self.pad_value = Float(
            text="Value to pad with",
            value=0,
            minval=0.0,
            maxval=1.0,
            doc="""The value that will be added to the image as part of the padding, 
                    recommended to be 0 for background/ black or 1 for foreground. 
                    High values will lead to rescaling of the image""",
        )
        self.x_axis = IntegerRange(
            text="Padd x axis",
            value=(0,0),
            minval=0,
        )
        self.y_axis =  IntegerRange(
            text="Padd y axis",
            value=(0,0),
            minval=0,
        ) 
        self.z_axis =  IntegerRange(
            text="Padd z axis",
            value=(0,0),
            minval=0,
            doc="""If the image is 2D it will not pad the Z even if you put values in this setting""",
        )


    def run(self, workspace):
        x_name = self.x_name.value #input image name selected from dropdown

        y_name = self.y_name.value #output that you type in the box

        x = workspace.image_set.get_image(x_name) #get the specific image

        x_data = x.pixel_data #array of pixel

        #thisis padding the image 
        if x.volumetric:
            y_data = numpy.pad(x_data, pad_width=((self.z_axis.min,self.z_axis.max), (self.y_axis.min,self.y_axis.max), (self.x_axis.min, self.x_axis.max)), constant_values=self.pad_value.value)

            mask = numpy.pad(x.mask, pad_width=((self.z_axis.min,self.z_axis.max), (self.y_axis.min,self.y_axis.max), (self.x_axis.min, self.x_axis.max)), constant_values=self.pad_value.value)

            mask = skimage.img_as_bool(mask)

            if x.has_crop_mask:
                y_cropmask = numpy.pad(
                    x.crop_mask, pad_width=((self.z_axis.min,self.z_axis.max), (self.y_axis.min,self.y_axis.max), (self.x_axis.min, self.x_axis.max)), constant_values=self.pad_value.value
                )

                y_cropmask = skimage.img_as_bool(y_cropmask)
            else:
                y_cropmask = None


        else:
            y_data =  numpy.pad(x_data, pad_width=((self.y_axis.min,self.y_axis.max), (self.x_axis.min, self.x_axis.max)), constant_values=self.pad_value.value)
            mask = numpy.pad(x.mask, pad_width=((self.y_axis.min,self.y_axis.max), (self.x_axis.min, self.x_axis.max)), constant_values=self.pad_value.value)

            mask = skimage.img_as_bool(mask)

            if x.has_crop_mask:
                y_cropmask = numpy.pad(
                    x.crop_mask, pad_width=((self.y_axis.min,self.y_axis.max), (self.x_axis.min, self.x_axis.max)), constant_values=self.pad_value.value
                )

                y_cropmask = skimage.img_as_bool(y_cropmask)
            else:
                y_cropmask = None

        #this a class packaging the image is in a way that can be used by cellprofiler

        y = Image(
            y_data,
            mask=mask,
            crop_mask=y_cropmask,
            dimensions=y_data.ndim,
        ) 

        #this is adding the image to the workspace, so that it's available for the next module
        workspace.image_set.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = x.dimensions #dimensions is 3 because 3d image

 
    def settings(self):
        __settings__ = super(PadImage, self).settings()

        return __settings__ + [self.pad_value, self.x_axis, self.y_axis, self.z_axis]

    def visible_settings(self):
        __settings__ = super(PadImage, self).visible_settings()
#add the if here
        __settings__ += [self.pad_value, self.x_axis, self.y_axis, self.z_axis]

        return __settings__
