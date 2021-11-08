import cellprofiler_core.object
import mahotas
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

O_DISTANCE = "Distance"
O_MARKERS = "Markers"
O_SHAPE = "Shape"
O_INTENSITY = "Intensity"

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

    variable_revision_number = 3

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

        self.reference_name = ImageSubscriber(
            text="Reference Image",
            doc="Image to reference for the *{O_INTENSITY}* method".format(**{"O_INTENSITY": O_INTENSITY})
        )

        self.gaussian_sigma = cellprofiler_core.setting.text.Float(
            text="Segmentation distance transform smoothing factor",
            value=1.,
            doc="Sigma defines how 'smooth' the Gaussian kernel makes the image. Higher sigma means a smoother image."
        )

        self.min_dist = cellprofiler_core.setting.text.Integer(
            text="Minimum distance between seeds",
            value=1,
            minval=0,
            doc="""\
        Minimum number of pixels separating peaks in a region of `2 * min_distance + 1 `
        (i.e. peaks are separated by at least min_distance). 
        To find the maximum number of peaks, set this value to `1`. 
        """
        )

        self.min_intensity = cellprofiler_core.setting.text.Float(
            text="Minimum absolute internal distance",
            value=0.,
            minval=0.,
            doc="""\
        Minimum absolute intensity threshold for seed generation. Since this threshold is
        applied to the distance transformed image, this defines a minimum object
        "size". Objects smaller than this size will not contain seeds. 

        By default, the absolute threshold is the minimum value of the image.
        For distance transformed images, this value is `0` (or the background).
        """
        )

        self.exclude_border = cellprofiler_core.setting.text.Integer(
            text="Pixels from border to exclude",
            value=0,
            minval=0,
            doc="Exclude seed generation from within `n` pixels of the image border."
        )

        self.max_seeds = cellprofiler_core.setting.text.Integer(
            text="Maximum number of seeds",
            value=-1,
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
            self.operation,
            self.markers_name,
            self.mask_name,
            self.connectivity,
            self.compactness,
            self.footprint,
            self.downsample,
            self.watershed_line,
            self.declump_method,
            self.reference_name,
            self.gaussian_sigma,
            self.min_dist,
            self.min_intensity,
            self.exclude_border,
            self.max_seeds,
            self.structuring_element,
        ]

    def visible_settings(self):
        __settings__ = [self.use_advanced]

        __settings__ = __settings__ + super(Watershed, self).settings()

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
        if self.use_advanced.value:
            if self.declump_method == O_SHAPE:
                __settings__ = __settings__ + [
                    self.declump_method,
                    self.gaussian_sigma,
                    self.min_dist,
                    self.min_intensity,
                    self.exclude_border,
                    self.max_seeds,
                    self.structuring_element,
                ]
            else:
                __settings__ = __settings__ + [
                    self.declump_method,
                    self.reference_name,
                    self.gaussian_sigma,
                    self.min_dist,
                    self.min_intensity,
                    self.exclude_border,
                    self.max_seeds,
                    self.structuring_element,
                ]

        return __settings__

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        # watershed algorithm for distance-based method:
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

            # resize back to the original size if downsampled
            if factor > 1:
                y_data = skimage.transform.resize(
                    y_data, original_shape, mode="edge", order=0, preserve_range=True
                )

                y_data = numpy.rint(y_data).astype(numpy.uint16)
                x_data = x.pixel_data > threshold

        # watershed algorithm for marker-based method:
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

            y_data = skimage.segmentation.watershed(
                image=x_data,
                markers=markers_data,
                mask=mask_data,
                connectivity=self.connectivity.value,
                compactness=self.compactness.value,
                watershed_line=self.watershed_line.value,
            )

        # watershed segmentation is stored in y_data variable
        if self.use_advanced.value:
            # check the dimensions of the structuring element
            strel_dim = self.structuring_element.value.ndim

            # test if the structuring element dimensions match the image dimensions
            if strel_dim != dimensions:
                raise ValueError("Structuring element does not match object dimensions: "
                                 "{} != {}".format(strel_dim, dimensions))

            # Get the segmentation distance transform for the watershed segmentation
            peak_image = scipy.ndimage.distance_transform_edt(y_data > 0)

            # shape-based method; generate a watershed ready image
            if self.declump_method.value == O_SHAPE:
                # Use the reverse of the image to get basins at peaks
                watershed_image = -peak_image
                watershed_image -= watershed_image.min()

            # intensity-based method
            else:
                # get the intensity image data
                reference_name = self.reference_name.value
                reference = images.get_image(reference_name)
                reference_data = reference.pixel_data

                # Set the image as a float and rescale to full bit depth
                watershed_image = skimage.img_as_float(reference_data, force_copy=True)
                watershed_image -= watershed_image.min()
                watershed_image = 1 - watershed_image

                if reference.multichannel:
                    watershed_image = skimage.color.rgb2gray(watershed_image)

            # Smooth the image
            watershed_image = skimage.filters.gaussian(watershed_image, sigma=self.gaussian_sigma.value)

            # Generate local peaks; returns a list of coords for each peak
            seed_coords = skimage.feature.peak_local_max(peak_image,
                                                   min_distance=self.min_dist.value,
                                                   threshold_rel=self.min_intensity.value,
                                                   exclude_border=self.exclude_border.value,
                                                   num_peaks=self.max_seeds.value if self.max_seeds.value != -1 else numpy.inf)

            # generate an array w/ same dimensions as the original image with all elements having value False
            seeds = numpy.zeros_like(peak_image, dtype=bool)

            # set value to True at every local peak
            seeds[tuple(seed_coords.T)] = True

            # Dilate seeds based on settings
            seeds = skimage.morphology.binary_dilation(seeds, self.structuring_element.value)

            # get the number of objects from the distance-based or marker-based watershed run above
            number_objects = skimage.measure.label(y_data, return_num=True)[1]

            seeds_dtype = (numpy.uint16 if number_objects < numpy.iinfo(numpy.uint16).max else numpy.uint32)

            # NOTE: Not my work, the comments below are courtesy of Ray
            #
            # Create a marker array where the unlabeled image has a label of
            # -(nobjects+1)
            # and every local maximum has a unique label which will become
            # the object's label. The labels are negative because that
            # makes the watershed algorithm use FIFO for the pixels which
            # yields fair boundaries when markers compete for pixels.
            #
            seeds = scipy.ndimage.label(seeds)[0]

            markers = numpy.zeros_like(seeds, dtype=seeds_dtype)
            markers[seeds > 0] = -seeds[seeds > 0]

            # Perform the watershed
            watershed_boundaries = skimage.segmentation.watershed(
                connectivity=self.connectivity.value,
                image=watershed_image,
                markers=markers,
                mask=x_data != 0
            )

            y_data = watershed_boundaries.copy()
            # Copy the location of the "background"
            zeros = numpy.where(y_data == 0)
            # Re-shift all of the labels into the positive realm
            y_data += numpy.abs(numpy.min(y_data)) + 1
            # Re-apply the background
            y_data[zeros] = 0

        # finalize and convert watershed to objects to export
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
