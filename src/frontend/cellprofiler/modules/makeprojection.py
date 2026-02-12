"""
MakeProjection
==============
**MakeProjection** combines two or more two-dimensional images of the same
field of view into a single two-dimensional image.

This module combines a set of images by performing a mathematical
operation of your choice at each pixel position; please refer to the
settings help for more information on the available operations. The
process of averaging or summing a Z-stack (3D image stack) is known as
making a projection.

This module will create a projection of all images specified in the
Input modules; most commonly you will want to use grouping to select
subsets of images to be combined into each projection. To
achieve per-folder projections (i.e., creating a single projection for each set
of images in a folder, for all input folders), make the following setting
selections:

#. In the **Images** module, drag-and-drop the parent folder containing
   the sub-folders.
#. In the **Metadata** module, enable metadata extraction and extract
   metadata from the folder name by using a regular expression to
   capture the subfolder name, e.g., ``.*[\\\\/](?P<Subfolder>.*)$``
#. In the **NamesAndTypes** module, specify the appropriate names for
   any desired channels.
#. In the **Groups** module, enable image grouping, and select the
   metadata tag representing the sub-folder name as the metadata
   category.

Keep in mind that the projection image is not immediately available in
subsequent modules because the output of this module is not complete
until all image processing cycles have completed. Therefore, the
projection should be created with a separate pipeline from your
analysis pipeline.

**MakeProjection** will not work on images that
have been loaded as 3D volumes in **NamesAndTypes** so be sure *Process
as 3D* is set to *No* in that module. For more information on loading image stacks and movies,
see *Help > Creating a Project > Loading Image Stacks and Movies*.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also the help for the **Input** modules.
"""


import numpy
from cellprofiler_core.image import AbstractImage
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import ImageName
from cellprofiler_core.setting.text.number import Float

from cellprofiler_library.modules._makeprojection import accumulate_projection, calculate_final_projection
from cellprofiler_library.opts.makeprojection import ProjectionType

P_AVERAGE = ProjectionType.AVERAGE.value
P_MAXIMUM = ProjectionType.MAXIMUM.value
P_MINIMUM = ProjectionType.MINIMUM.value
P_SUM = ProjectionType.SUM.value
P_VARIANCE = ProjectionType.VARIANCE.value
P_POWER = ProjectionType.POWER.value
P_BRIGHTFIELD = ProjectionType.BRIGHTFIELD.value
P_MASK = ProjectionType.MASK.value

P_ALL = [
    P_AVERAGE,
    P_MAXIMUM,
    P_MINIMUM,
    P_SUM,
    P_VARIANCE,
    P_POWER,
    P_BRIGHTFIELD,
    P_MASK,
]

K_PROVIDER = "Provider"


class MakeProjection(Module):
    module_name = "MakeProjection"
    category = "Image Processing"
    variable_revision_number = 2

    def create_settings(self):
        self.image_name = ImageSubscriber(
            "Select the input image",
            "None",
            doc="Select the images to be made into a projection.",
        )

        self.projection_type = Choice(
            "Type of projection",
            P_ALL,
            doc="""\
The final projection image can be created by the following methods:

-  *%(P_AVERAGE)s:* Use the average pixel intensity at each pixel
   position.
-  *%(P_MAXIMUM)s:* Use the maximum pixel value at each pixel position.
-  *%(P_MINIMUM)s:* Use the minimum pixel value at each pixel position.
-  *%(P_SUM)s:* Add the pixel values at each pixel position.
-  *%(P_VARIANCE)s:* Compute the variance at each pixel position.
   The variance method is described in Selinummi et al (2009). The
   method is designed to operate on a Z-stack of brightfield images
   taken at different focus planes. Background pixels will have
   relatively uniform illumination whereas cytoplasm pixels will have
   higher variance across the Z-stack.
-  *%(P_POWER)s:* Compute the power at a given frequency at each pixel
   position.
   The power method is experimental. The method computes the power at a
   given frequency through the Z-stack. It might be used with a phase
   contrast image where the signal at a given pixel will vary
   sinusoidally with depth. The frequency is measured in Z-stack steps
   and pixels that vary with the given frequency will have a higher
   score than other pixels with similar variance, but different
   frequencies.
-  *%(P_BRIGHTFIELD)s:* Perform the brightfield projection at each
   pixel position.
   Artifacts such as dust appear as black spots that are most strongly
   resolved at their focal plane with gradually increasing signals
   below. The brightfield method scores these as zero since the dark
   appears in the early Z-stacks. These pixels have a high score for the
   variance method but have a reduced score when using the brightfield
   method.
-  *%(P_MASK)s:* Compute a binary image of the pixels that are masked
   in any of the input images.
   The mask method operates on any masks that might have been applied to
   the images in a group. The output is a binary image where the “1”
   pixels are those that are not masked in all of the images and the “0”
   pixels are those that are masked in one or more of the images.
   You can use the output of the mask method to mask or crop all of the
   images in a group similarly. Use the mask method to combine all of
   the masks in a group, save the image and then use **Crop**,
   **MaskImage** or **MaskObjects** in another pipeline to mask all
   images or objects in the group similarly.

References
^^^^^^^^^^

-  Selinummi J, Ruusuvuori P, Podolsky I, Ozinsky A, Gold E, et al.
   (2009) “Bright field microscopy as an alternative to whole cell
   fluorescence in automated analysis of macrophage images”, *PLoS ONE*
   4(10): e7497 `(link)`_.

.. _(link): https://doi.org/10.1371/journal.pone.0007497
"""
            % globals(),
        )

        self.projection_image_name = ImageName(
            "Name the output image",
            "ProjectionBlue",
            doc="Enter the name for the projected image.",
            provided_attributes={"aggregate_image": True, "available_on_last": True,},
        )
        self.frequency = Float(
            "Frequency",
            6.0,
            minval=1.0,
            doc="""\
*(Used only if "%(P_POWER)s" is selected as the projection method)*

This setting controls the frequency at which the power is measured. A
frequency of 2 will respond most strongly to pixels that alternate
between dark and light in successive z-stack slices. A frequency of N
will respond most strongly to pixels whose brightness cycles every N
slices."""
            % globals(),
        )

    def settings(self):
        return [
            self.image_name,
            self.projection_type,
            self.projection_image_name,
            self.frequency,
        ]

    def visible_settings(self):
        result = [self.image_name, self.projection_type, self.projection_image_name]
        if self.projection_type == P_POWER:
            result += [self.frequency]
        return result

    def prepare_group(self, workspace, grouping, image_numbers):
        """Reset the aggregate image at the start of group processing"""
        if len(image_numbers) > 0:
            provider = ImageProvider.create(
                self.projection_image_name.value,
                self.projection_type.value,
                self.frequency.value,
            )
            provider.save_state(self.get_dictionary())
        return True

    def run(self, workspace):
        provider = ImageProvider.restore_from_state(self.get_dictionary())
        workspace.image_set.add_provider(provider)
        image = workspace.image_set.get_image(self.image_name.value)
        pixels = image.pixel_data
        if not provider.has_image:
            provider.set_image(image)
        else:
            provider.accumulate_image(image)
        provider.save_state(self.get_dictionary())
        if self.show_window:
            workspace.display_data.pixels = pixels
            workspace.display_data.provider_pixels = provider.provide_image(
                workspace.image_set
            ).pixel_data

    def is_aggregation_module(self):
        """Return True because we aggregate over all images in a group"""
        return True

    def post_group(self, workspace, grouping):
        """Handle processing that takes place at the end of a group

        Add the provider to the workspace if not present. This could
        happen if the image set didn't reach this module.
        """
        image_set = workspace.image_set
        if self.projection_image_name.value not in image_set.names:
            provider = ImageProvider.restore_from_state(self.get_dictionary())
            image_set.add_provider(provider)

    def display(self, workspace, figure):
        pixels = workspace.display_data.pixels
        provider_pixels = workspace.display_data.provider_pixels
        figure.set_subplots((2, 1))
        if provider_pixels.ndim == 3:
            figure.subplot_imshow(0, 0, pixels, self.image_name.value)
            figure.subplot_imshow(
                1,
                0,
                provider_pixels,
                self.projection_image_name.value,
                sharexy=figure.subplot(0, 0),
            )
        else:
            figure.subplot_imshow_bw(0, 0, pixels, self.image_name.value)
            figure.subplot_imshow_bw(
                1,
                0,
                provider_pixels,
                self.projection_image_name.value,
                sharexy=figure.subplot(0, 0),
            )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            # Added frequency
            setting_values = setting_values + ["6"]
            variable_revision_number = 2
        return setting_values, variable_revision_number


class ImageProvider(AbstractImage):
    """Provide the image after averaging but before dilation and smoothing"""

    D_NAME = "name"
    D_FREQUENCY = "frequency"
    D_METHOD = "method"
    D_LIBRARY_STATE = "library_state"

    def __init__(self, name, method, frequency=6.0):
        """Construct using a parent provider that does the real work

        name - name of the image provided
        """
        super(ImageProvider, self).__init__()
        self._name = name
        self.method = ProjectionType(method)
        self.frequency = frequency
        self.library_state = {}
        self._cached_image = None

    @staticmethod
    def create(name, how_to_accumulate, frequency=6):
        """Factory method to create the appropriate ImageProvider."""
        return ImageProvider(name, how_to_accumulate, frequency)

    def save_state(self, d):
        """Save the provider state to a dictionary

        d - store state in this dictionary
        """
        d[self.D_NAME] = self._name
        d[self.D_FREQUENCY] = self.frequency
        d[self.D_METHOD] = self.method.value
        d[self.D_LIBRARY_STATE] = self.library_state

    @staticmethod
    def restore_from_state(d):
        """Create a provider from the state stored in the dictionary

        d - dictionary from call to save_state

        returns a new ImageProvider built from the saved state
        """
        name = d[ImageProvider.D_NAME]
        frequency = d[ImageProvider.D_FREQUENCY]
        method = d[ImageProvider.D_METHOD]
        library_state = d.get(ImageProvider.D_LIBRARY_STATE, {})
        
        provider = ImageProvider.create(name, method, frequency)
        provider.library_state = library_state
        return provider

    def reset(self):
        """Reset accumulator at start of groups"""
        self.library_state = {}
        self._cached_image = None

    @property
    def has_image(self):
        return len(self.library_state) > 0

    @property
    def count(self):
        return self.library_state.get("image_count")

    def set_image(self, image):
        self._cached_image = None
        self.library_state = {}
        self.accumulate_image(image)

    def accumulate_image(self, image):
        self._cached_image = None
        
        pixels = image.pixel_data
        mask = image.mask if image.has_mask else None
        
        self.library_state = accumulate_projection(
            pixels, 
            mask, 
            self.library_state, 
            self.method, 
            self.frequency
        )

    def provide_image(self, image_set):
        """Return the final projected image."""
        if self._cached_image is not None:
            return self._cached_image
            
        pixels, mask = calculate_final_projection(self.library_state, self.method)
        
        if numpy.all(mask):
            self._cached_image = Image(pixels)
        else:
            self._cached_image = Image(pixels, mask=mask)
            
        return self._cached_image

    def get_name(self):
        """Return the name of the output image."""
        return self._name

    def release_memory(self):
        """Don't discard the image at end of image set"""
        pass
