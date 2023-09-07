import cellprofiler_core.object
import mahotas
from matplotlib import image
import numpy
import scipy.ndimage
import skimage.color
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.segmentation
import skimage.transform
from cellprofiler_core.module.image_segmentation import ImageSegmentation
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Integer, Float
import cellprofiler.library.functions.object_processing

from cellprofiler.library.modules import watershed

O_DISTANCE = "Distance"
O_MARKERS = "Markers"
O_LOCAL = "Local"
O_REGIONAL = "Regional"
O_SHAPE = "Shape"
O_INTENSITY = "Intensity"

default_settings = {
    "seed_method": O_LOCAL,
    "max_seeds": -1,
    "min_distance": 1,
    "min_intensity": 0.0,
    "connectivity": 1,
    "compactness": 0.0,
    "watershed_line": False,
    "gaussian_sigma": 0.0,
}

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

   -   **Footprint** defines dimensions of the window used to scan
       the input image for local maximum. The footprint can be interpreted as a region,
       window, structuring element or volume that subsamples the input image. 
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

Selecting *Advanced Settings* will split the detected objects into smaller objects based on a seeded watershed method 
that will:

    - Compute the `local maxima`_ (either through the `Euclidean distance transformation`_ of the 
      segmented objects or through the intensity values of a reference image

    - Dilate the seeds as specified

    - Use these seeds as markers for watershed

    - NOTE: This implementation is based off of the **IdentifyPrimaryObjects** declumping implementation.
      For more information, see the aforementioned module.

    .. _Euclidean distance transformation: 
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
    .. _local maxima: http://scikit-image.org/docs/dev/api/skimage.feature.html#peak-local-max 

    - The Advanced Settings declumping code was originally written by Madison Bowden as the DeclumpObjects plugin.

       
""".format(
    **{"O_DISTANCE": O_DISTANCE, "O_MARKERS": O_MARKERS}
)


class Watershed(ImageSegmentation):
    category = "Advanced"

    module_name = "Watershed"

    variable_revision_number = 4

    def create_settings(self):
        super(Watershed, self).create_settings()

        self.use_advanced = Binary(
            "Use advanced settings?",
            value=False,
            doc="""\
        The advanced settings provide additional segmentation options to improve 
        the separation of adjacent objects. If this option is not selected,
        then the watershed algorithm is applied according to the basic settings. 
""",
        )

        self.watershed_method = Choice(
            "Select watershed method",
            choices=[O_DISTANCE, O_INTENSITY, O_MARKERS],
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

        self.seed_method = Choice(
            "Select seed generation method",
            choices=[O_LOCAL, O_REGIONAL],
            value=default_settings["seed_method"],
            doc="""\
"""
        )

        self.display_maxima = Binary(
            "Display watershed seeds?",
            value=False,
            doc="""\
Select "*{YES}*" to display the seeds used for watershed.
            """.format(
                **{"YES": "Yes"}
            )
        )

        self.markers_name = ImageSubscriber(
            "Markers",
            doc="An image marking the approximate centers of the objects for segmentation.",
        )

        self.intensity_name = ImageSubscriber(
            "Intensity image",
            doc="",
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
            value=default_settings["connectivity"],
        )

        self.compactness = Float(
            text="Compactness",
            minval=0.0,
            value=default_settings["compactness"],
            doc="""\
Use `compact watershed`_ with given compactness parameter. 
Higher values result in more regularly-shaped watershed basins.


.. _compact watershed: http://scikit-image.org/docs/0.13.x/api/skimage.morphology.html#r395
""",
        )

        self.watershed_line = Binary(
            text="Separate watershed labels",
            value=default_settings["watershed_line"],
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

        self.declump_method = cellprofiler_core.setting.choice.Choice(
            text="Declump method",
            choices=[O_SHAPE, O_INTENSITY],
            value=O_SHAPE,
            doc="""\
        This setting allows you to choose the method that is used to draw the
        line between segmented objects. 

        -  *{O_SHAPE}:* Dividing lines between clumped objects are based on
           the shape of the clump. For example, when a clump contains two
           objects, the dividing line will be placed where indentations occur
           between the two objects. The intensity of the original image is
           not necessary in this case. 

           **Technical description:** The distance transform of the segmentation 
           is used to identify local maxima as seeds (i.e. the centers of the 
           individual objects), and the seeds are then used on the inverse of 
           that distance transform to determine new segmentations via watershed.

        -  *{O_INTENSITY}:* Dividing lines between clumped objects are determined
           based on the intensity of the original image. This works best if the
           dividing line between objects is dimmer than the objects themselves.

           **Technical description:** The distance transform of the segmentation 
           is used to identify local maxima as seeds (i.e. the centers of the 
           individual objects). Those seeds are then used as markers for a 
           watershed on the inverted original intensity image.
        """.format(**{
                "O_SHAPE": O_SHAPE,
                "O_INTENSITY": O_INTENSITY
            })
        )

        self.gaussian_sigma = cellprofiler_core.setting.text.Float(
            text="Segmentation distance transform smoothing factor",
            value=default_settings["gaussian_sigma"],
            doc="Sigma defines how 'smooth' the Gaussian kernel makes the image. Higher sigma means a smoother image."
        )

        self.min_dist = cellprofiler_core.setting.text.Integer(
            text="Minimum distance between seeds",
            value=default_settings["min_distance"],
            minval=0,
            doc="""\
        Minimum number of pixels separating peaks in a region of `2 * min_distance + 1 `
        (i.e. peaks are separated by at least min_distance). 
        To find the maximum number of peaks, set this value to `1`. 
        """
        )

        self.min_intensity = cellprofiler_core.setting.text.Float(
            text="Minimum absolute internal distance",
            value=default_settings["min_intensity"],
            minval=0.,
            doc="""\
        Minimum absolute intensity threshold for seed generation. Since this threshold is
        applied to the distance transformed image, this defines a minimum object
        "size". Objects smaller than this size will not contain seeds. 

        By default, the absolute threshold is the minimum value of the image.
        For distance transformed images, this value is `0` (or the background).
        """
        )

        self.exclude_border = Binary(
            "Discard objects touching the border of the image?",
            value=False,
            doc="""\ 
""",
        )

        self.max_seeds = cellprofiler_core.setting.text.Integer(
            text="Maximum number of seeds",
            value=default_settings["max_seeds"],
            doc="""\
        Maximum number of seeds to generate. Default is no limit. 
        When the number of seeds exceeds this number, seeds are chosen 
        based on largest internal distance.
        """
        )

        self.structuring_element = cellprofiler_core.setting.StructuringElement(
            text="Structuring element for seed dilation",
            doc="""\
        Structuring element to use for dilating the seeds. 
        Volumetric images will require volumetric structuring elements.
        """
        )

    def settings(self):
        __settings__ = super(Watershed, self).settings()

        return __settings__ + [
            self.use_advanced,
            self.watershed_method,
            self.seed_method,
            self.display_maxima,
            self.markers_name,
            self.intensity_name,
            self.mask_name,
            self.connectivity,
            self.compactness,
            self.footprint,
            self.downsample,
            self.watershed_line,
            self.declump_method,
            self.gaussian_sigma,
            self.min_dist,
            self.min_intensity,
            self.exclude_border,
            self.max_seeds,
            self.structuring_element,
        ]

    def visible_settings(self):
        __settings__ = [self.use_advanced]
        __settings__ += super(Watershed, self).visible_settings()
        __settings__ += [
            self.mask_name,
            self.watershed_method,   
        ]

        if self.watershed_method == O_MARKERS:
                __settings__ += [
                    self.markers_name,
                ]

        if self.use_advanced: 
            if self.watershed_method != O_MARKERS:
                __settings__ += [
                    self.seed_method,
                ]
                if self.seed_method == O_LOCAL:
                    __settings__ += [
                        self.min_dist,
                        self.min_intensity,
                        self.max_seeds,
                    ]
            __settings__ += [
                self.gaussian_sigma,
                self.connectivity,
                self.compactness,
                self.watershed_line,
                ]

        __settings__ += [
            self.exclude_border,
            self.downsample,
            self.footprint,
            self.declump_method,
        ]

        if self.watershed_method == O_INTENSITY or self.declump_method == O_INTENSITY:
            # Provide the intensity image setting
            __settings__ += [
                self.intensity_name
            ]
        
        __settings__ += [
            self.structuring_element,
            ]
        
        __settings__ += [
            self.display_maxima,
        ]

        return __settings__

    def run(self, workspace):

        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        # Set the required images
        markers_data = None
        mask_data = None
        intensity_data = None

        if self.watershed_method.value == O_MARKERS:
            # Get markers
            markers_name = self.markers_name.value
            markers = images.get_image(markers_name)
            markers_data = markers.pixel_data

            if markers.multichannel:
                markers_data = skimage.color.rgb2gray(markers_data)

        if not self.mask_name.is_blank:
            mask_name = self.mask_name.value
            mask = images.get_image(mask_name)
            mask_data = mask.pixel_data

        # Get the intensity image
        if self.watershed_method == O_INTENSITY or self.declump_method.value == O_INTENSITY:
            # Get intensity image
            intensity_image = images.get_image(self.intensity_name.value)
            intensity_data = intensity_image.pixel_data
            if intensity_image.multichannel:
                    intensity_data = skimage.color.rgb2gray(intensity_data)

        y_data, seeds = watershed(
                input_image=x_data,
                markers_image=markers_data,
                intensity_image=intensity_data,
                mask=mask_data,
                return_seeds = True,
                method=self.watershed_method.value,
                declump_method=self.declump_method.value,
                exclude_border=self.exclude_border.value,
                downsample=self.downsample.value,
                footprint=self.footprint.value,
                structuring_element=self.structuring_element.shape,
                structuring_element_size=self.structuring_element.size,                
                seed_method=self.seed_method.value if self.use_advanced else default_settings["seed_method"],
                max_seeds=self.max_seeds.value if self.use_advanced else default_settings["max_seeds"],
                min_distance=self.min_dist.value if self.use_advanced else default_settings["min_distance"],
                min_intensity=self.min_intensity.value if self.use_advanced else default_settings["min_intensity"],
                connectivity=self.connectivity.value if self.use_advanced else default_settings["connectivity"],
                compactness=self.compactness.value if self.use_advanced else default_settings["compactness"],
                watershed_line=self.watershed_line.value if self.use_advanced else default_settings["watershed_line"],
                gaussian_sigma=self.gaussian_sigma.value if self.use_advanced else default_settings["gaussian_sigma"],
                )

        objects = cellprofiler_core.object.Objects()

        objects.segmented = y_data

        objects.parent_image = x

        workspace.object_set.add_objects(objects, y_name)

        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = x.pixel_data
            workspace.display_data.x_data_name = self.x_name.value

            workspace.display_data.y_data = y_data
            workspace.display_data.y_data_name = self.y_name.value

            # Find object boundaries and combine with seeds
            object_outlines = skimage.segmentation.find_boundaries(y_data, mode="inner")
            outlines_and_seeds = seeds + object_outlines
            # Colour the boundaries based on the object label from y_data and mask out background
            workspace.display_data.outlines_and_seeds = (outlines_and_seeds > 0) * y_data

            workspace.display_data.dimensions = dimensions

    def display(self, workspace, figure):
        if self.show_window:
            if self.display_maxima:
                subplots = (2, 2)
            else:
                subplots = (2, 1)
            figure.set_subplots(
                dimensions=workspace.display_data.dimensions, subplots=subplots
                )
            cmap = figure.return_cmap()

            ax = figure.subplot_imshow_grayscale(
                0,
                0,
                workspace.display_data.x_data,
                workspace.display_data.x_data_name,
                )
            figure.subplot_imshow_labels(
                1,
                0,
                workspace.display_data.y_data,
                workspace.display_data.y_data_name,
                sharexy=ax,
                colormap=cmap,
            )
            if self.display_maxima:
                figure.subplot_imshow_labels(
                    0,
                    1,
                    workspace.display_data.outlines_and_seeds,
                    workspace.display_data.y_data_name + " object outlines and seeds",
                    sharexy=ax,
                    colormap=cmap,
                )


    def upgrade_settings(self, setting_values, variable_revision_number, module_name):

        if variable_revision_number == 1:
            # Last two items were moved down to add more options for seeded watershed
            __settings__ = setting_values[:-2]

            # Add default connectivity and compactness
            __settings__ += [1, 0.0]

            # Add the rest of the settings
            __settings__ += setting_values[-2:]

            variable_revision_number = 2

        if variable_revision_number == 2:
            # Use advanced? is a new parameter
            # first two settings are unchanged
            __settings__ = setting_values[0:2]

            # add False for "Use advanced?"
            __settings__ += [False]

            # add remainder of settings
            __settings__ += setting_values[2:]

            variable_revision_number = 3


        else:
            __settings__ = setting_values

        return __settings__, variable_revision_number
