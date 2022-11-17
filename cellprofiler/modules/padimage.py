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
            value=1,
            minval=0.0,
            maxval=1.0,
            doc="""The value that will be added to the image as part of the padding, 
                    recommended to be 0 for background/ black or 1 for foreground""",
        )
        self.x_axis = IntegerRange(
            text="Padd x axis",
            value=(1,1),
            minval=0,
        )
        self.y_axis =  IntegerRange(
            text="Padd y axis",
            value=(1,1),
            minval=0,
        )
        self.z_axis =  IntegerRange(
            text="Padd z axis",
            value=(1,1),
            minval=0,
        )


    def run(self, workspace):
        x_name = self.x_name.value #input image name selected from dropdown

        y_name = self.y_name.value #output that you type in the box

        images = workspace.image_set #getting the data, list of images

        x = images.get_image(x_name) #get the specific image

        x_data = x.pixel_data #array of pixel

        #y_data = medialaxis(x_data, x.multichannel, x.volumetric) #output
        y_data = numpy.pad(x_data, pad_width=((self.z_axis.min,self.z_axis.max), (self.y_axis.min,self.y_axis.max), (self.x_axis.min, self.x_axis.max)), constant_values=self.pad_value.value)
        print("hello", x.dimensions)
        y = Image(dimensions=x.dimensions, image=y_data, parent_image=x)
        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = x.dimensions

    def settings(self):
        __settings__ = super(PadImage, self).settings()

        return __settings__ + [self.pad_value, self.x_axis, self.y_axis, self.z_axis]

    def visible_settings(self):
        __settings__ = super(PadImage, self).visible_settings()

        __settings__ += [self.pad_value, self.x_axis, self.y_axis, self.z_axis]

        return __settings__
