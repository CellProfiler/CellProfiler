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

P_AVERAGE = "Average"
P_MAXIMUM = "Maximum"
P_MINIMUM = "Minimum"
P_SUM = "Sum"
P_VARIANCE = "Variance"
P_POWER = "Power"
P_BRIGHTFIELD = "Brightfield"
P_MASK = "Mask"
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
    D_IMAGE = "image"
    D_HOW_TO_ACCUMULATE = "howtoaccumulate"
    D_IMAGE_COUNT = "imagecount"
    D_VSQUARED = "vsquared"
    D_VSUM = "vsum"
    D_POWER_IMAGE = "powerimage"
    D_POWER_MASK = "powermask"
    D_STACK_NUMBER = "stacknumber"
    D_BRIGHT_MAX = "brightmax"
    D_BRIGHT_MIN = "brightmin"
    D_NORM0 = "norm0"

    def __init__(self, name, how_to_accumulate, frequency=6):
        """Construct using a parent provider that does the real work

        name - name of the image provided
        """
        super(ImageProvider, self).__init__()
        self._name = name
        self.frequency = frequency
        self._how_to_accumulate = how_to_accumulate
        self._image_count = None
        self._cached_image = None

    @staticmethod
    def create(name, how_to_accumulate, frequency=6):
        """Factory method to create the appropriate ImageProvider subclass based on the accumulation method."""
        providers = {
            P_AVERAGE: AverageProvider,
            P_MAXIMUM: MaximumProvider,
            P_MINIMUM: MinimumProvider,
            P_SUM: SumProvider,
            P_VARIANCE: VarianceProvider,
            P_POWER: PowerProvider,
            P_BRIGHTFIELD: BrightfieldProvider,
            P_MASK: MaskProvider,
        }

        provider_class = providers.get(how_to_accumulate)
        if provider_class:
            return provider_class(name, how_to_accumulate, frequency)

        raise NotImplementedError(
            "No such accumulation method: %s" % how_to_accumulate
        )

    def save_state(self, d):
        """Save the provider state to a dictionary

        d - store state in this dictionary
        """
        d[self.D_NAME] = self._name
        d[self.D_FREQUENCY] = self.frequency
        d[self.D_HOW_TO_ACCUMULATE] = self._how_to_accumulate
        d[self.D_IMAGE_COUNT] = self._image_count

    @staticmethod
    def restore_from_state(d):
        """Create a provider from the state stored in the dictionary

        d - dictionary from call to save_state

        returns a new ImageProvider built from the saved state
        """
        name = d[ImageProvider.D_NAME]
        frequency = d[ImageProvider.D_FREQUENCY]
        how_to_accumulate = d[ImageProvider.D_HOW_TO_ACCUMULATE]
        image_provider = ImageProvider.create(name, how_to_accumulate, frequency)
        image_provider.restore(d)
        return image_provider

    def restore(self, d):
        self._image_count = d[self.D_IMAGE_COUNT]

    def reset(self):
        """Reset accumulator at start of groups"""
        self._image_count = None
        self._cached_image = None

    @property
    def has_image(self):
        return self._image_count is not None

    @property
    def count(self):
        return self._image_count

    def set_image(self, image):
        self._cached_image = None
        if image.has_mask:
            self._image_count = image.mask.astype(int)
        else:
            self._image_count = numpy.ones(image.pixel_data.shape[:2], int)
        self._set_image_impl(image)

    def accumulate_image(self, image):
        self._cached_image = None
        if image.has_mask:
            self._image_count += image.mask.astype(int)
        else:
            self._image_count += 1
        self._accumulate_image_impl(image)

    def provide_image(self, image_set):
        """Return the final projected image."""
        raise NotImplementedError

    def get_name(self):
        """Return the name of the output image."""
        return self._name

    def release_memory(self):
        """Don't discard the image at end of image set"""
        pass

    def _set_image_impl(self, image):
        """Abstract method: Initialize the accumulator with the first image."""
        raise NotImplementedError

    def _wrap_image(self, pixel_data, mask):
        """Helper to wrap pixel data in an Image object, handling 2D/3D masks."""
        if numpy.all(mask):
            return Image(pixel_data)

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        return Image(pixel_data, mask=mask)

    def _accumulate_image_impl(self, image):
        """Abstract method: Accumulate a subsequent image into the provider."""
        raise NotImplementedError


class SumProvider(ImageProvider):
    """Accumulates the sum of pixel intensities."""

    def __init__(self, name, how_to_accumulate, frequency=6):
        super(SumProvider, self).__init__(name, how_to_accumulate, frequency)
        self._image = None

    def save_state(self, d):
        super(SumProvider, self).save_state(d)
        d[self.D_IMAGE] = self._image

    def restore(self, d):
        super(SumProvider, self).restore(d)
        self._image = d[self.D_IMAGE]

    def reset(self):
        super(SumProvider, self).reset()
        self._image = None

    def _set_image_impl(self, image):
        self._image = image.pixel_data.copy()
        if image.has_mask:
            self._image[~image.mask] = 0

    def _accumulate_image_impl(self, image):
        if image.has_mask:
            self._image[image.mask] += image.pixel_data[image.mask]
        else:
            self._image += image.pixel_data

    def provide_image(self, image_set):
        if self._cached_image is not None:
            return self._cached_image

        image_count = self._image_count
        mask = image_count > 0

        if numpy.any(~mask):
            cached_image = self._image.copy()
            cached_image[~mask] = 0
        else:
            cached_image = self._image

        if numpy.all(mask):
            self._cached_image = Image(cached_image)
        else:
            self._cached_image = Image(cached_image, mask=mask)
        return self._cached_image


class AverageProvider(SumProvider):
    """Accumulates the sum, then divides by count to get the average."""

    def provide_image(self, image_set):
        if self._cached_image is not None:
            return self._cached_image

        image_count = self._image_count
        if self._image.ndim == 3:
             image_count = numpy.dstack([image_count] * self._image.shape[2])
        mask = image_count > 0

        cached_image = self._image / image_count
        if cached_image.ndim == 3 and mask.ndim == 2:
            cached_image[~mask, :] = 0
        else:
            cached_image[~mask] = 0

        self._cached_image = self._wrap_image(cached_image, mask)
        return self._cached_image


class MaximumProvider(ImageProvider):
    """Keeps the maximum pixel intensity at each position."""

    def __init__(self, name, how_to_accumulate, frequency=6):
        super(MaximumProvider, self).__init__(name, how_to_accumulate, frequency)
        self._image = None

    def save_state(self, d):
        super(MaximumProvider, self).save_state(d)
        d[self.D_IMAGE] = self._image

    def restore(self, d):
        super(MaximumProvider, self).restore(d)
        self._image = d[self.D_IMAGE]

    def reset(self):
        super(MaximumProvider, self).reset()
        self._image = None

    def _set_image_impl(self, image):
        self._image = image.pixel_data.copy()
        if image.has_mask:
            self._image[~image.mask] = 0

    def _accumulate_image_impl(self, image):
        if image.has_mask:
            self._image[image.mask] = numpy.maximum(
                self._image[image.mask], image.pixel_data[image.mask]
            )
        else:
            self._image = numpy.maximum(image.pixel_data, self._image)

    def provide_image(self, image_set):
        if self._cached_image is not None:
            return self._cached_image

        image_count = self._image_count
        mask = image_count > 0

        if numpy.any(~mask):
            cached_image = self._image.copy()
            cached_image[~mask] = 0
        else:
            cached_image = self._image

        self._cached_image = self._wrap_image(cached_image, mask)
        return self._cached_image


class MinimumProvider(ImageProvider):
    """Keeps the minimum pixel intensity at each position."""

    def __init__(self, name, how_to_accumulate, frequency=6):
        super(MinimumProvider, self).__init__(name, how_to_accumulate, frequency)
        self._image = None

    def save_state(self, d):
        super(MinimumProvider, self).save_state(d)
        d[self.D_IMAGE] = self._image

    def restore(self, d):
        super(MinimumProvider, self).restore(d)
        self._image = d[self.D_IMAGE]

    def reset(self):
        super(MinimumProvider, self).reset()
        self._image = None

    def _set_image_impl(self, image):
        self._image = image.pixel_data.copy()
        if image.has_mask:
            self._image[~image.mask] = 1

    def _accumulate_image_impl(self, image):
        if image.has_mask:
            self._image[image.mask] = numpy.minimum(
                self._image[image.mask], image.pixel_data[image.mask]
            )
        else:
            self._image = numpy.minimum(image.pixel_data, self._image)

    def provide_image(self, image_set):
        if self._cached_image is not None:
            return self._cached_image

        image_count = self._image_count
        mask = image_count > 0

        if numpy.any(~mask):
            cached_image = self._image.copy()
            cached_image[~mask] = 0
        else:
            cached_image = self._image

        self._cached_image = self._wrap_image(cached_image, mask)
        return self._cached_image


class VarianceProvider(ImageProvider):
    """Calculates pixel variance across the image stack."""

    def __init__(self, name, how_to_accumulate, frequency=6):
        super(VarianceProvider, self).__init__(name, how_to_accumulate, frequency)
        self._vsum = None
        self._vsquared = None

    def save_state(self, d):
        super(VarianceProvider, self).save_state(d)
        d[self.D_VSUM] = self._vsum
        d[self.D_VSQUARED] = self._vsquared

    def restore(self, d):
        super(VarianceProvider, self).restore(d)
        self._vsum = d[self.D_VSUM]
        self._vsquared = d[self.D_VSQUARED]

    def reset(self):
        super(VarianceProvider, self).reset()
        self._vsum = None
        self._vsquared = None

    def _set_image_impl(self, image):
        self._vsum = image.pixel_data.copy()
        self._vsum[~image.mask] = 0
        self._vsquared = self._vsum.astype(numpy.float64) ** 2.0

    def _accumulate_image_impl(self, image):
        mask = image.mask
        self._vsum[mask] += image.pixel_data[mask]
        self._vsquared[mask] += image.pixel_data[mask].astype(numpy.float64) ** 2

    def provide_image(self, image_set):
        if self._cached_image is not None:
            return self._cached_image

        image_count = self._image_count
        mask_2d = image_count > 0

        if self._vsquared.ndim == 3:
            image_count = numpy.dstack([image_count] * self._vsquared.shape[2])
            
        mask = image_count > 0

        cached_image = numpy.zeros(self._vsquared.shape, numpy.float32)
        cached_image[mask] = self._vsquared[mask] / image_count[mask]
        cached_image[mask] -= self._vsum[mask] ** 2 / (image_count[mask] ** 2)

        cached_image[~mask] = 0

        self._cached_image = self._wrap_image(cached_image, mask_2d)
        return self._cached_image


class PowerProvider(ImageProvider):
    """Calculates power at a specific frequency across the stack."""

    def __init__(self, name, how_to_accumulate, frequency=6):
        super(PowerProvider, self).__init__(name, how_to_accumulate, frequency)
        self._vsum = None
        self._power_image = None
        self._power_mask = None
        self._stack_number = 0

    def save_state(self, d):
        super(PowerProvider, self).save_state(d)
        d[self.D_VSUM] = self._vsum
        d[self.D_POWER_IMAGE] = self._power_image
        d[self.D_POWER_MASK] = self._power_mask
        d[self.D_STACK_NUMBER] = self._stack_number

    def restore(self, d):
        super(PowerProvider, self).restore(d)
        self._vsum = d[self.D_VSUM]
        self._power_image = d[self.D_POWER_IMAGE]
        self._power_mask = d[self.D_POWER_MASK]
        self._stack_number = d[self.D_STACK_NUMBER]

    def reset(self):
        super(PowerProvider, self).reset()
        self._vsum = None
        self._power_image = None
        self._power_mask = None
        self._stack_number = 0

    def _set_image_impl(self, image):
        self._vsum = image.pixel_data.copy()
        self._vsum[~image.mask] = 0
        self._power_mask = self._image_count.astype(numpy.complex128).copy()
        self._power_image = image.pixel_data.astype(numpy.complex128).copy()
        self._stack_number = 1

    def _accumulate_image_impl(self, image):
        multiplier = numpy.exp(
            2j * numpy.pi * float(self._stack_number) / self.frequency
        )
        self._stack_number += 1
        mask = image.mask
        self._vsum[mask] += image.pixel_data[mask]
        self._power_image[mask] += multiplier * image.pixel_data[mask]
        self._power_mask[mask] += multiplier

    def provide_image(self, image_set):
        if self._cached_image is not None:
            return self._cached_image

        image_count = self._image_count
        mask_2d = image_count > 0

        if self._power_image.ndim == 3:
            image_count = numpy.dstack([image_count] * self._power_image.shape[2])
        mask = image_count > 0

        cached_image = numpy.zeros(image_count.shape, numpy.complex128)
        cached_image[mask] = self._power_image[mask]
        cached_image[mask] -= (
            self._vsum[mask] * self._power_mask[mask] / image_count[mask]
        )
        cached_image = (cached_image * numpy.conj(cached_image)).real.astype(
            numpy.float32
        )
        cached_image[~mask] = 0

        self._cached_image = self._wrap_image(cached_image, mask_2d)
        return self._cached_image


class BrightfieldProvider(ImageProvider):
    """Performs brightfield projection (focus metric based on max-min difference)."""

    def __init__(self, name, how_to_accumulate, frequency=6):
        super(BrightfieldProvider, self).__init__(name, how_to_accumulate, frequency)
        self._bright_max = None
        self._bright_min = None
        self._norm0 = None

    def save_state(self, d):
        super(BrightfieldProvider, self).save_state(d)
        d[self.D_BRIGHT_MAX] = self._bright_max
        d[self.D_BRIGHT_MIN] = self._bright_min
        d[self.D_NORM0] = self._norm0

    def restore(self, d):
        super(BrightfieldProvider, self).restore(d)
        self._bright_max = d[self.D_BRIGHT_MAX]
        self._bright_min = d[self.D_BRIGHT_MIN]
        self._norm0 = d[self.D_NORM0]

    def reset(self):
        super(BrightfieldProvider, self).reset()
        self._bright_max = None
        self._bright_min = None
        self._norm0 = None

    def _set_image_impl(self, image):
        self._bright_max = image.pixel_data.copy()
        self._bright_min = image.pixel_data.copy()
        self._norm0 = numpy.mean(image.pixel_data)

    def _accumulate_image_impl(self, image):
        mask = image.mask
        norm = numpy.mean(image.pixel_data)
        pixel_data = image.pixel_data * self._norm0 / norm
        max_mask = (self._bright_max < pixel_data) & mask
        min_mask = (self._bright_min > pixel_data) & mask
        self._bright_min[min_mask] = pixel_data[min_mask]
        self._bright_max[max_mask] = pixel_data[max_mask]
        self._bright_min[max_mask] = self._bright_max[max_mask]

    def provide_image(self, image_set):
        if self._cached_image is not None:
            return self._cached_image

        image_count = self._image_count
        mask_2d = image_count > 0

        if self._bright_max.ndim == 3:
            image_count = numpy.dstack([image_count] * self._bright_max.shape[2])
        mask = image_count > 0

        cached_image = numpy.zeros(image_count.shape, numpy.float32)
        cached_image[mask] = self._bright_max[mask] - self._bright_min[mask]
        cached_image[~mask] = 0

        self._cached_image = self._wrap_image(cached_image, mask_2d)
        return self._cached_image


class MaskProvider(ImageProvider):
    """Computes the intersection of all masks."""

    def __init__(self, name, how_to_accumulate, frequency=6):
        super(MaskProvider, self).__init__(name, how_to_accumulate, frequency)
        self._image = None

    def save_state(self, d):
        super(MaskProvider, self).save_state(d)
        d[self.D_IMAGE] = self._image

    def restore(self, d):
        super(MaskProvider, self).restore(d)
        self._image = d[self.D_IMAGE]

    def reset(self):
        super(MaskProvider, self).reset()
        self._image = None

    def _set_image_impl(self, image):
        self._image = image.mask

    def _accumulate_image_impl(self, image):
        self._image = self._image & image.mask

    def provide_image(self, image_set):
        if self._cached_image is not None:
            return self._cached_image

        self._cached_image = Image(self._image)
        return self._cached_image
