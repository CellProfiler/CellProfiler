"""
MedialAxis
==========

**MedialAxis** computes the medial axis or topological skeleton of a binary image. Rather than by sequentially
removing pixels as in **MorphologicalSkeleton**, the medial axis is computed based on the 
distance transform of the thresholded image (i.e., the distance each foreground pixel is 
from a background pixel). See `this tutorial <http://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html>`__ for more information. 

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import numpy
import skimage.color
import skimage.morphology
from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing


class MedialAxis(ImageProcessing):
    category = "Advanced"

    module_name = "MedialAxis"

    variable_revision_number = 1

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        if x.multichannel:
            x_data = skimage.color.rgb2gray(x_data)

        if x.dimensions == 3:
            y_data = numpy.zeros_like(x_data)

            for z, image in enumerate(x_data):
                y_data[z] = skimage.morphology.medial_axis(image)
        else:
            y_data = skimage.morphology.medial_axis(x_data)

        y = Image(dimensions=x.dimensions, image=y_data, parent_image=x)

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = x.dimensions
