import cellprofiler_core.object
import mahotas
import numpy
import scipy.ndimage
import skimage.color
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.transform
from cellprofiler_core.module.image_segmentation import ImageSegmentation
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Integer, Float

O_DISTANCE = "Distance"
O_MARKERS = "Markers"

__doc__ = """
Watershed
=========

**Watershed** is a segmentation algorithm. It is used to separate
different objects in an image. For more information please visit
the `scikit-image documentation`_ on the **Watershed** implementation that 
CellProfiler uses.

.. _skimage label: http://scikit-image.org/docs/dev/api/skimage.measure.html#label

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

This module has two operating modes:

-  *{O_DISTANCE}* (default): This is classical nuclei segmentation using
   watershed. Your “Input” image should be a binary image. Markers and other 
   inputs for the watershed algorithm will be automatically generated.

   -   **Footprint** defines dimentions of the window used to scan
       the input image for local maximum. The footprint can be interpreted as a region,
       window, structring element or volume that subsamples the input image. 
       The distance transform will create local maximum from a binary image that 
       will be at the centers of objects. A large footprint will suppress local maximum 
       that are close together into a single maximum, but this will require more memory 
       and time to run. Large footprint could result in a blockier segmentation. 
       A small footprint will preserve local maximum that are close together,
       but this can lead to oversegmentation. If speed and memory are issues, 
       choosing a lower footprint can be offset by downsampling the input image. 
       See `mahotas regmax`_ for more information.

   .. _mahotas regmax: http://mahotas.readthedocs.io/en/latest/api.html?highlight=regmax#mahotas.regmax

   -   **Downsample** an n-dimensional image by local averaging. If the downsampling factor 
       is 1, the image is not downsampled. To downsample more, increase the number from 1.

   
-  *{O_MARKERS}*: Similar to the IdentifySecondaryObjects in 2D, use manually 
   generated markers and supply an optional mask for watershed. Watershed works best 
   when the “Input” image has high intensity surrounding regions of interest 
   and low intensity inside regions of interest. 

   -   **Connectivity** is the maximum number of orthogonal hops to consider 
       a pixel/voxel as a neighbor. Accepted values are ranging from 1 
       to the number of dimensions.
       Two pixels are connected when they are neighbors and have the same value. 
       In 2D, they can be neighbors either in a 1- or 2-connected sense. The value 
       refers to the maximum number of orthogonal hops to consider a pixel/voxel a neighbor.
       See `skimage label`_ for more information.

       .. _scikit-image documentation: http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
       
       Note: when using marker-based **Watershed** that it is typical to use the input 
       binary image as the mask. Otherwise, if the mask is *None*, the background will be 
       interpreted as an object and **Watershed** may yield unexpected results.
       
   -   **Compactness**, use `compact watershed`_ with given compactness parameter. 
       Higher values result in more regularly-shaped watershed basins. 
       
       .. _compact watershed: http://scikit-image.org/docs/0.13.x/api/skimage.morphology.html#r395
""".format(
    **{"O_DISTANCE": O_DISTANCE, "O_MARKERS": O_MARKERS}
)


class Watershed(ImageSegmentation):
    category = "Advanced"

    module_name = "Watershed"

    variable_revision_number = 2

    def create_settings(self):
        super(Watershed, self).create_settings()

        self.operation = Choice(
            text="Generate from",
            choices=[O_DISTANCE, O_MARKERS],
            value=O_DISTANCE,
            doc="""\
Select a method of inputs for the watershed algorithm:

-  *{O_DISTANCE}* (default): This is classical nuclei segmentation using
   watershed. Your “Input” image should be a binary image. Markers and other 
   inputs for the watershed algorithm will be automatically generated.
-  *{O_MARKERS}*: Similar to the IdentifySecondaryObjects in 2D, use manually 
   generated markers and supply an optional mask for watershed. 
   Watershed works best when the “Input” image has high intensity 
   surrounding regions of interest and low intensity inside
   regions of interest. 
   
""".format(
                **{"O_DISTANCE": O_DISTANCE, "O_MARKERS": O_MARKERS}
            ),
        )

        self.markers_name = ImageSubscriber(
            "Markers",
            doc="An image marking the approximate centers of the objects for segmentation.",
        )

        self.mask_name = ImageSubscriber(
            "Mask",
            can_be_blank=True,
            doc="Optional. Only regions not blocked by the mask will be segmented.",
        )

        self.connectivity = Integer(
            doc="""\
Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. 
Accepted values are ranging from 1 to the number of dimensions.

Two pixels are connected when they are neighbors and have the same value. 
In 2D, they can be neighbors either in a 1- or 2-connected sense. The value 
refers to the maximum number of orthogonal hops to consider a pixel/voxel a neighbor.

See `skimage label`_ for more information.

.. _skimage label: http://scikit-image.org/docs/dev/api/skimage.measure.html#label
""",
            minval=1,
            text="Connectivity",
            value=1,
        )

        self.compactness = Float(
            text="Compactness",
            minval=0.0,
            value=0.0,
            doc="""\
Use `compact watershed`_ with given compactness parameter. 
Higher values result in more regularly-shaped watershed basins.


.. _compact watershed: http://scikit-image.org/docs/0.13.x/api/skimage.morphology.html#r395
""",
        )

        self.watershed_line = Binary(
            text="Separate watershed labels",
            value=False,
            doc="""\
Create a 1 pixel wide line around the watershed labels. This effectively separates
the different objects identified by the watershed algorithm, rather than allowing them 
to touch. The line has the same label as the background.
""",
        )

        self.footprint = Integer(
            doc="""\
The **Footprint** defines the dimensions of the window used to scan
the input image for local maxima. The footprint can be interpreted as a
region, window, structuring element or volume that subsamples the input
image. The distance transform will create local maxima from a binary
image that will be at the centers of objects. A large footprint will
suppress local maxima that are close together into a single maxima, but this will require
more memory and time to run. A large footprint can also result in a blockier segmentation.
A small footprint will preserve maxima that are close together, 
but this can lead to oversegmentation. If speed and memory are issues,
choosing a lower connectivity can be offset by downsampling the input image. 
See `mahotas regmax`_ for more information.

.. _mahotas regmax: http://mahotas.readthedocs.io/en/latest/api.html?highlight=regmax#mahotas.regmax
""",
            minval=1,
            text="Footprint",
            value=8,
        )

        self.downsample = Integer(
            doc="""\
Downsample an n-dimensional image by local averaging. If the downsampling factor is 1,
the image is not downsampled.
""",
            minval=1,
            text="Downsample",
            value=1,
        )

    def settings(self):
        __settings__ = super(Watershed, self).settings()

        return __settings__ + [
            self.operation,
            self.markers_name,
            self.mask_name,
            self.connectivity,
            self.compactness,
            self.footprint,
            self.downsample,
            self.watershed_line,
        ]

    def visible_settings(self):
        __settings__ = super(Watershed, self).settings()

        __settings__ = __settings__ + [self.operation]

        if self.operation.value == O_DISTANCE:
            __settings__ = __settings__ + [self.footprint, self.downsample]
        else:
            __settings__ = __settings__ + [
                self.markers_name,
                self.mask_name,
                self.connectivity,
                self.compactness,
                self.watershed_line,
            ]

        return __settings__

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        if self.operation.value == O_DISTANCE:
            original_shape = x_data.shape

            factor = self.downsample.value

            if factor > 1:
                if x.volumetric:
                    factors = (1, factor, factor)
                else:
                    factors = (factor, factor)

                x_data = skimage.transform.downscale_local_mean(x_data, factors)

            threshold = skimage.filters.threshold_otsu(x_data)

            x_data = x_data > threshold

            distance = scipy.ndimage.distance_transform_edt(x_data)

            distance = mahotas.stretch(distance)

            surface = distance.max() - distance

            if x.volumetric:
                footprint = numpy.ones(
                    (self.footprint.value, self.footprint.value, self.footprint.value)
                )
            else:
                footprint = numpy.ones((self.footprint.value, self.footprint.value))

            peaks = mahotas.regmax(distance, footprint)

            if x.volumetric:
                markers, _ = mahotas.label(peaks, numpy.ones((16, 16, 16)))
            else:
                markers, _ = mahotas.label(peaks, numpy.ones((16, 16)))

            y_data = mahotas.cwatershed(surface, markers)

            y_data = y_data * x_data

            if factor > 1:
                y_data = skimage.transform.resize(
                    y_data, original_shape, mode="edge", order=0, preserve_range=True
                )

                y_data = numpy.rint(y_data).astype(numpy.uint16)
        else:
            markers_name = self.markers_name.value

            markers = images.get_image(markers_name)

            markers_data = markers.pixel_data

            if x.multichannel:
                x_data = skimage.color.rgb2gray(x_data)

            if markers.multichannel:
                markers_data = skimage.color.rgb2gray(markers_data)

            mask_data = None

            if not self.mask_name.is_blank:
                mask_name = self.mask_name.value

                mask = images.get_image(mask_name)

                mask_data = mask.pixel_data

            y_data = skimage.morphology.watershed(
                image=x_data,
                markers=markers_data,
                mask=mask_data,
                connectivity=self.connectivity.value,
                compactness=self.compactness.value,
                watershed_line=self.watershed_line.value,
            )

        y_data = skimage.measure.label(y_data)

        objects = cellprofiler_core.object.Objects()

        objects.segmented = y_data

        objects.parent_image = x

        workspace.object_set.add_objects(objects, y_name)

        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = x.pixel_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):

        if variable_revision_number == 1:
            # Last two items were moved down to add more options for seeded watershed
            __settings__ = setting_values[:-2]

            # Add default connectivity and compactness
            __settings__ += [1, 0.0]

            # Add the rest of the settings
            __settings__ += setting_values[-2:]

            variable_revision_number = 2

        else:
            __settings__ = setting_values

        return __settings__, variable_revision_number
