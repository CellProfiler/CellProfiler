import skimage

import cellprofiler_core.object
from cellprofiler_core.module.image_segmentation import ImageSegmentation
from cellprofiler_core.setting import Binary, StructuringElement
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Integer, Float
from cellprofiler_library.modules import watershed

O_DISTANCE = "Distance"
O_MARKERS = "Markers"
O_LOCAL = "Local"
O_REGIONAL = "Regional"
O_SHAPE = "Shape"
O_INTENSITY = "Intensity"
O_NONE = "None"

basic_mode_defaults = {
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

**Watershed** is used to separate different objects in an image. This works by
'flooding' pixel intensity valleys (that is, areas of low intensity) from seed
objects. When the water from one flooded valley meets the water from a nearby
but different flooded valley, this is the "watershed line" and defines the
separation between two objects.


The Watershed module helps users to define what their valley and seed images
will be. The valley image is determined by the *declump* method. For shape-based
declumping, the inverted distance transform of the binary (black and white)
input image will be used. If intensity based declumping is used, the inverted
intensity will be used, meaning that areas of high pixel intensity will be set
as the bottom of valleys.


Seed objects can be calculated from the distance transform of your input binary
image by selecting the *Distance* method. This method will calculate seed
objects for pixels that are distant from the background (black pixels), which
are typically the centers of nuclei. You can also provide your own seed objects
by selecting the *Markers* watershed method. Alternatively, you can select the
*Intensity* watershed method, which will set pixel intensity maxima as seed
objects. If the *advanced mode* is enabled, you will have access to additional
settings to tweak for determining seeds. 


Good seed objects are essential for achieving accurate watershed segmentation.
Too many seed objects per valley (ie. multiple seeds for one valley) leads to
over-segmentation, whereas too few seed objects (ie. one seed object for
multiple valleys) leads to under-segmentation.


For more information please visit the `scikit-image documentation`_ on the
**Watershed** implementation that CellProfiler uses.


.. _scikit-image documentation: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed


The input image to the Watershed module must be a binary image, which can be generated using the
**Threshold** module.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

"""


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
The advanced settings provide additional options to improve calculation of seed
objects. If this option is not selected, then the watershed algorithm is applied
according to the basic settings. 
""",
        )

        self.watershed_method = Choice(
            "Select watershed method",
            choices=[O_DISTANCE, O_MARKERS, O_INTENSITY],
            value=O_DISTANCE,
            doc="""\
Select a method of inputs for the watershed algorithm:

-  *{O_DISTANCE}* (default): This is the classical object segmentation method 
    using watershed. Seed objects will be calculated from the distance transform
    of the input image.

-  *{O_MARKERS}*: Use this method if you have already calculated seed objects,
    for example from the **FindMaxima** module.

- *{O_INTENSITY}*: Use this method to calculate seeds based on intensity maxima
  of the provided intensity image.
""".format(
                **{
                    "O_DISTANCE": O_DISTANCE, 
                    "O_MARKERS": O_MARKERS, 
                    "O_INTENSITY": O_INTENSITY
                }
            ),
        )

        self.seed_method = Choice(
            "Select seed generation method",
            choices=[O_LOCAL, O_REGIONAL],
            value=basic_mode_defaults["seed_method"],
            doc="""\
-  *{O_LOCAL}*: Seed objects will be found within the footprint. One 
    seed object will be proposed within each footprint 'window'.

-  *{O_REGIONAL}*: The regional method can look for maxima slightly outside
    of the provided footprint setting. In this scenario, it can be somewhat
    automatic in finding seed objcets. However, *{O_LOCAL}* behaves identically
    at higher footprint values. Furthermore, *{O_REGIONAL}* is more
    computationally intensive to use when compared to local.
""".format(
                **{"O_LOCAL": O_LOCAL, "O_REGIONAL": O_REGIONAL}
            )
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
            doc="""\
An image marking the approximate centers, aka seeds, of objects to be
segmented.
                """,
        )

        self.intensity_name = ImageSubscriber(
            "Intensity image",
            doc="""\
Intensity image to be used for finding intensity-based seed objects and/or
declumping.

If provided, the same intensity image can be used for both finding maxima and
finding dividing lines between clumped objects. This works best if the dividing
line between objects is dimmer than the objects themselves.
                """,
        )

        self.mask_name = ImageSubscriber(
            "Mask",
            can_be_blank=True,
            doc="Optional. Only regions not blocked by the mask will be labeled.",
        )

        self.connectivity = Integer(
            doc="""\
Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
Accepted values are ranging from 1 to the number of dimensions.

Two pixels are connected when they are neighbors and have the same value. In 2D,
they can be neighbors either in a 1- or 2-connected sense. The value refers to
the maximum number of orthogonal hops to consider a pixel/voxel a neighbor.

See `skimage watershed`_ for more information.

.. _skimage watershed: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed
""",
            minval=1,
            text="Connectivity",
            value=basic_mode_defaults["connectivity"],
        )

        self.compactness = Float(
            text="Compactness",
            minval=0.0,
            value=basic_mode_defaults["compactness"],
            doc="""\
Use `compact watershed`_ with a given compactness parameter. Higher values result
in more regularly-shaped watershed basins.


.. _compact watershed: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed
""",
        )

        self.footprint = Integer(
            doc="""\
The **Footprint** defines the dimensions of the window used to scan the input
image for local maxima. The footprint can be interpreted as a region, window,
structuring element or volume that subsamples the input image. The distance
transform will create local maxima from a binary image that will be at the
centers of objects. A large footprint will suppress local maxima that are close
together into a single maxima, but this will require more memory and time to
run. A large footprint can also result in a blockier segmentation. A small
footprint will preserve maxima that are close together, but this can lead to
oversegmentation. If speed and memory are issues, choosing a lower footprint can
be offset by downsampling the input image. 


See `skimage peak_local_max`_ for more information.

.. _skimage peak_local_max: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max
""",
            minval=1,
            text="Footprint",
            value=8,
        )

        self.downsample = Integer(
            doc="""\
Downsample an n-dimensional image by local averaging. If the downsampling factor
is 1, the image is not downsampled.

Images will be resized to their original input size following watershed
segmentation.
""",
            minval=1,
            text="Downsample",
            value=1,
        )

        self.watershed_line = Binary(
            text="Separate watershed labels",
            value=basic_mode_defaults["watershed_line"],
            doc="""\
Create a 1 pixel wide line around the watershed labels. This effectively
separates the different objects identified by the watershed algorithm, rather
than allowing them to touch. The line has the same label as the background.
""",
        )

        self.declump_method = Choice(
            text="Declump method",
            choices=[O_SHAPE, O_INTENSITY, O_NONE],
            value=O_SHAPE,
            doc="""\
This setting allows you to choose the method that is used to draw the line
between segmented objects. 

-  *{O_SHAPE}:* Dividing lines between clumped objects are based on
    the shape of the clump. For example, when a clump contains two objects, the
    dividing line will be placed where indentations occur between the two
    objects. The intensity of the original image is not necessary in this case.
    **Technical description:** The distance transform of the segmentation is
    used to identify local maxima as seeds (i.e. the centers of the individual
    objects), and the seeds are then used on the inverse of that distance
    transform to determine new segmentations via watershed.


-  *{O_INTENSITY}:* Dividing lines between clumped objects are determined
    based on the intensity of the original image. This works best if the
    dividing line between objects is dimmer than the objects themselves.
    **Technical description:** The distance transform of the segmentation is
    used to identify local maxima as seeds (i.e. the centers of the individual
    objects). Those seeds are then used as markers for a watershed on the
    inverted original intensity image.
        """.format(**{
                "O_SHAPE": O_SHAPE,
                "O_INTENSITY": O_INTENSITY
            })
        )

        self.gaussian_sigma = Float(
            text="Segmentation distance transform smoothing factor",
            value=basic_mode_defaults["gaussian_sigma"],
            doc="""\
Sigma defines how 'smooth' the Gaussian kernel makes the distance transformed
input image. A higher sigma means a smoother image.
"""
        )

        self.min_distance = Integer(
            text="Minimum distance between seeds",
            value=basic_mode_defaults["min_distance"],
            minval=0,
            doc="""\
Minimum number of pixels separating peaks in a region of `2 * min_distance + 1 `
(i.e. peaks are separated by at least min_distance). To find the maximum number
of peaks, set this value to `1`. 
"""
        )

        self.min_intensity = Float(
            text="Specify the minimum intensity of a peak",
            value=basic_mode_defaults["min_intensity"],
            minval=0.,
            doc="""\
Intensity peaks below this threshold value will be excluded. Use this to ensure
that your local maxima are within objects of interest.
"""
        )

        self.exclude_border = Binary(
            "Discard objects touching the border of the image?",
            value=False,
            doc="Clear objects connected to the image border.",
        )

        self.max_seeds = Integer(
            text="Maximum number of seeds",
            value=basic_mode_defaults["max_seeds"],
            doc="""\
Maximum number of seeds to generate. Default is no limit, defined by `-1`. When
the number of seeds exceeds this number, seeds are chosen based on largest
internal distance.
        """
        )

        self.structuring_element = StructuringElement(
            text="Structuring element for seed dilation",
            doc="""\
Structuring element to use for dilating the seeds. Volumetric images will
require volumetric structuring elements.
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
            self.min_distance,
            self.min_intensity,
            self.exclude_border,
            self.max_seeds,
            self.structuring_element,
        ]

    def visible_settings(self):
        __settings__ = [self.use_advanced]
        __settings__ += super(Watershed, self).visible_settings()
        # If no declumping, there's no reason to offer watershed options
        if self.declump_method == O_NONE:
            __settings__.pop(0) # Remove the advanced option
            __settings__ += [
                self.mask_name,
                self.declump_method
                ]
            return __settings__

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
                        self.min_distance,
                        self.min_intensity,
                        self.max_seeds,
                    ]

            if self.watershed_method == O_DISTANCE or self.declump_method == O_SHAPE:
                __settings__ += [
                    self.gaussian_sigma,
                ]

            __settings__ += [
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
        if self.watershed_method == O_INTENSITY or self.declump_method == O_INTENSITY:
            intensity_image = images.get_image(self.intensity_name.value)
            intensity_data = intensity_image.pixel_data
            if intensity_image.multichannel:
                    intensity_data = skimage.color.rgb2gray(intensity_data)

        y_data, seeds = watershed(
                input_image=x_data,
                mask=mask_data,
                watershed_method=self.watershed_method.value,
                declump_method=self.declump_method.value,
                seed_method=self.seed_method.value if self.use_advanced \
                    else basic_mode_defaults["seed_method"],
                intensity_image=intensity_data,
                markers_image=markers_data,
                max_seeds=self.max_seeds.value if self.use_advanced \
                    else basic_mode_defaults["max_seeds"],
                downsample=self.downsample.value,
                min_distance=self.min_distance.value if self.use_advanced \
                    else basic_mode_defaults["min_distance"],
                min_intensity=self.min_intensity.value if self.use_advanced \
                    else basic_mode_defaults["min_intensity"],
                footprint=self.footprint.value,
                connectivity=self.connectivity.value if self.use_advanced \
                    else basic_mode_defaults["connectivity"],
                compactness=self.compactness.value if self.use_advanced \
                    else basic_mode_defaults["compactness"],
                exclude_border=self.exclude_border.value,
                watershed_line=self.watershed_line.value if self.use_advanced \
                    else basic_mode_defaults["watershed_line"],
                gaussian_sigma=self.gaussian_sigma.value if self.use_advanced \
                    else basic_mode_defaults["gaussian_sigma"],
                structuring_element=self.structuring_element.shape,
                structuring_element_size=self.structuring_element.size,
                return_seeds=True,
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

            # If declumping is None then maxima are not calculated
            if self.display_maxima and not self.declump_method == O_NONE:
                # Find object boundaries and combine with seeds
                object_outlines = skimage.segmentation.find_boundaries(y_data, mode="inner")
                outlines_and_seeds = seeds + object_outlines
                # Colour the boundaries based on the object label from y_data and mask out background
                workspace.display_data.outlines_and_seeds = (outlines_and_seeds > 0) * y_data

            workspace.display_data.dimensions = dimensions

    def display(self, workspace, figure):
        if self.show_window:
            if self.display_maxima and not self.declump_method == O_NONE:
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
            if self.display_maxima and not self.declump_method == O_NONE:
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
            new_values = setting_values[:-2]

            # add: connectivity, compactness
            new_values += [1, 0.0]

            # Add the rest of the settings
            new_values += setting_values[-2:]

            setting_values = new_values
            variable_revision_number = 2

        if variable_revision_number == 2:
            # Use advanced? is a new parameter
            # first two settings are unchanged
            new_values = setting_values[0:2]

            # add: use advanced?
            new_values += [False]

            # add remainder of settings
            new_values += setting_values[2:]

            setting_values = new_values
            variable_revision_number = 3

        if variable_revision_number == 3:
            # is "use advanced?" true?
            is_advanced = setting_values[2] == "Yes"

            new_values = setting_values[0:4]

            # add: seed method and display maxima
            new_values += [O_LOCAL, False]

            new_values += setting_values[4:5]

            # add: intensity name
            # if advanced: intensity name gets old reference image name
            new_values += [setting_values[12] if is_advanced else "None"]

            new_values += setting_values[5:11]

            if is_advanced:
                new_values += setting_values[11:12]
                new_values += setting_values[13:]
            else:
                # add declump method, gaussian sigma, min distance,
                # min intensity, exlude border, max seeds, structuring element
                new_values += [O_SHAPE, 0.0, 1, 0.0, False, -1, "Disk,1"]

            setting_values = new_values
            variable_revision_number = 4

        return setting_values, variable_revision_number
