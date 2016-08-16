from cellprofiler.gui.help import LOADING_IMAGE_SEQ_HELP_REF

__doc__ = '''
<b>Make Projection</b> combines several two-dimensional images of
the same field of view together, either by performing a mathematical operation
upon the pixel values at each pixel position.
<hr>
This module combines a set of images by performing a mathematic operation of your
choice at each pixel position; please refer to the settings help for more information on the available
operations. The process of averaging or summing a Z-stack (3D image stack) is known as making a projection.

<p>This module will create a projection of all images specified in the Input modules. For more
information on loading image stacks and movies, see <i>%(LOADING_IMAGE_SEQ_HELP_REF)s</i>.
To achieve per-folder projections
i.e., creating a projection for each set of images in a folder, for all input folders,
make the following setting specifications:
<ol>
<li>In the <b>Images</b> module, drag-and-drop the parent folder containing the sub-folders.</li>
<li>In the <b>Metadata</b> module, enable metadata extraction and extract metadata from the folder name
by using a regular expression to capture the subfolder name, e.g., <code>.*[\\\\/](?P&lt;Subfolder&gt;.*)$</code></li>
<li>In the <b>NamesAndTypes</b> module, specify the appropriate names for any desired channels.</li>
<li>In the <b>Groups</b> module, enable image grouping, and select the metadata tag representing
the sub-folder name as the metadata category.</li>
</ol>
Keep in mind that the projection image is not immediately available in subsequent modules because the
output of this module is not complete until all image processing cycles have completed. Therefore,
the projection should be created with a dedicated pipeline.</p>

See also the help for the <b>Input</b> modules.
''' % globals()

import numpy as np

import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.setting as cps

P_AVERAGE = 'Average'
P_MAXIMUM = 'Maximum'
P_MINIMUM = 'Minimum'
P_SUM = 'Sum'
P_VARIANCE = 'Variance'
P_POWER = 'Power'
P_BRIGHTFIELD = 'Brightfield'
P_MASK = 'Mask'
P_ALL = [P_AVERAGE, P_MAXIMUM, P_MINIMUM, P_SUM, P_VARIANCE, P_POWER,
         P_BRIGHTFIELD, P_MASK]

K_PROVIDER = "Provider"


class MakeProjection(cpm.Module):
    module_name = 'MakeProjection'
    category = 'Image Processing'
    variable_revision_number = 2

    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
                'Select the input image', cps.NONE, doc='''
            Select the image to be made into a projection.''')

        self.projection_type = cps.Choice(
                'Type of projection',
                P_ALL, doc='''
            The final projection image can be created by the following methods:
            <ul>
            <li><i>%(P_AVERAGE)s:</i> Use the average pixel intensity at each pixel position.</li>
            <li><i>%(P_MAXIMUM)s:</i> Use the maximum pixel value at each pixel position.</li>
            <li><i>%(P_MINIMUM)s:</i> Use the minimum pixel value at each pixel position.</li>
            <li><i>%(P_SUM)s:</i> Add the pixel values at each pixel position.</li>
            <li><i>%(P_VARIANCE)s:</i> Compute the variance at each pixel position. <br>
            The variance  method is described in Selinummi et al (2009).
            The method is designed to operate on a z-stack of brightfield images taken
            at different focus planes. Background pixels will have relatively uniform
            illumination whereas cytoplasm pixels will have higher variance across the
            z-stack.</li>
            <li><i>%(P_POWER)s:</i> Compute the power at a given frequency at each pixel position.<br>
            The power method is experimental. The method computes the power at a given
            frequency through the z-stack. It might be used with a phase contrast image
            where the signal at a given pixel will vary sinusoidally with depth. The
            frequency is measured in z-stack steps and pixels that vary with the given
            frequency will have a higher score than other pixels with similar variance,
            but different frequencies.</li>
            <li><i>%(P_BRIGHTFIELD)s:</i> Perform the brightfield projection at each pixel position.<br>
            Artifacts such as dust appear as black spots which are most strongly resolved
            at their focal plane with gradually increasing signals below. The brightfield
            method scores these as zero since the dark appears in the early z-stacks.
            These pixels have a high score for the variance method but have a reduced
            score when using the brightfield method.</li>
            <li><i>%(P_MASK)s:</i> Compute a binary image of the pixels that are
            masked in any of the input images.<br>
            The mask method operates on any masks that might have been applied to the
            images in a group. The output is a binary image where the "1" pixels are
            those that are not masked in all of the images and the "0" pixels are those
            that are masked in one or more of the images.<br>
            You can use the output of the mask method to mask or crop all of the
            images in a group similarly. Use the mask method to combine all of the
            masks in a group, save the image and then use <b>Crop</b>, <b>MaskImage</b> or
            <b>MaskObjects</b> in another pipeline to mask all images or objects in the
            group similarly.</li>
            </ul>
            <p>
            <b>References</b>
            <ul>
            <li>Selinummi J, Ruusuvuori P, Podolsky I, Ozinsky A, Gold E, et al. (2009)
            "Bright field microscopy as an alternative to whole cell fluorescence in
            automated analysis of macrophage images", <i>PLoS ONE</i> 4(10): e7497
            <a href="http://dx.doi.org/10.1371/journal.pone.0007497">(link)</a>.</li>
            </ul>
            </p>''' % globals())

        self.projection_image_name = cps.ImageNameProvider(
                'Name the output image',
                'ProjectionBlue', doc='''
            Enter the name for the projected image.''',
                provided_attributes={cps.AGGREGATE_IMAGE_ATTRIBUTE: True,
                                     cps.AVAILABLE_ON_LAST_ATTRIBUTE: True})
        self.frequency = cps.Float(
                "Frequency", 6.0, minval=1.0, doc="""
            <i>(Used only if %(P_POWER)s is selected as the projection method)</i><br>
            This setting controls the frequency at which the power
            is measured. A frequency of 2 will respond most strongly to
            pixels that alternate between dark and light in successive
            z-stack slices. A frequency of N will respond most strongly
            to pixels whose brightness cycle every N slices.""" % globals())

    def settings(self):
        return [self.image_name, self.projection_type,
                self.projection_image_name, self.frequency]

    def visible_settings(self):
        result = [self.image_name, self.projection_type,
                  self.projection_image_name]
        if self.projection_type == P_POWER:
            result += [self.frequency]
        return result

    def prepare_group(self, workspace, grouping, image_numbers):
        '''Reset the aggregate image at the start of group processing'''
        if len(image_numbers) > 0:
            provider = ImageProvider(self.projection_image_name.value,
                                     self.projection_type.value,
                                     self.frequency.value)
            provider.save_state(self.get_dictionary())
        return True

    def run(self, workspace):
        provider = ImageProvider.restore_from_state(self.get_dictionary())
        workspace.image_set.providers.append(provider)
        image = workspace.image_set.get_image(self.image_name.value)
        pixels = image.pixel_data
        if not provider.has_image:
            provider.set_image(image)
        else:
            provider.accumulate_image(image)
        provider.save_state(self.get_dictionary())
        if self.show_window:
            workspace.display_data.pixels = pixels
            workspace.display_data.provider_pixels = \
                provider.provide_image(workspace.image_set).pixel_data

    def is_aggregation_module(self):
        '''Return True because we aggregate over all images in a group'''
        return True

    def post_group(self, workspace, grouping):
        '''Handle processing that takes place at the end of a group

        Add the provider to the workspace if not present. This could
        happen if the image set didn't reach this module.
        '''
        image_set = workspace.image_set
        if self.projection_image_name.value not in image_set.names:
            provider = ImageProvider.restore_from_state(self.get_dictionary())
            image_set.providers.append(provider)

    def display(self, workspace, figure):
        pixels = workspace.display_data.pixels
        provider_pixels = workspace.display_data.provider_pixels
        figure.set_subplots((2, 1))
        if provider_pixels.ndim == 3:
            figure.subplot_imshow(0, 0, pixels,
                                  self.image_name.value)
            figure.subplot_imshow(1, 0, provider_pixels,
                                  self.projection_image_name.value,
                                  sharexy=figure.subplot(0, 0))
        else:
            figure.subplot_imshow_bw(0, 0, pixels,
                                     self.image_name.value)
            figure.subplot_imshow_bw(1, 0, provider_pixels,
                                     self.projection_image_name.value,
                                     sharexy=figure.subplot(0, 0))

    def upgrade_settings(self, setting_values,
                         variable_revision_number,
                         module_name, from_matlab):
        if from_matlab and module_name == 'Average':
            setting_values = setting_values[:2] + P_AVERAGE
            from_matlab = False
            module_name = self.module_name
            variable_revision_number = 1
        if (from_matlab and module_name == 'MakeProjection' and
                    variable_revision_number == 3):
            setting_values = setting_values[:3]
            from_matlab = False
            variable_revision_number = 1
        if (not from_matlab) and variable_revision_number == 1:
            # Added frequency
            setting_values = setting_values + ["6"]
        return setting_values, variable_revision_number, from_matlab


class ImageProvider(cpi.AbstractImageProvider):
    """Provide the image after averaging but before dilation and smoothing"""

    def __init__(self, name, how_to_accumulate, frequency=6):
        """Construct using a parent provider that does the real work

        name - name of the image provided
        """
        super(ImageProvider, self).__init__()
        self.__name = name
        self.frequency = frequency
        self.__image = None
        self.__how_to_accumulate = how_to_accumulate
        self.__image_count = None
        self.__cached_image = None
        #
        # Variance needs image squared as float64, image sum and count
        #
        self.__vsquared = None
        self.__vsum = None
        #
        # Power needs a running sum (reuse vsum), a power image of the mask
        # and a complex-values image
        #
        self.__power_image = None
        self.__power_mask = None
        self.__stack_number = 0
        #
        # Brightfield needs a maximum and minimum image
        #
        self.__bright_max = None
        self.__bright_min = None
        self.__norm0 = None

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

    def save_state(self, d):
        '''Save the provider state to a dictionary

        d - store state in this dictionary
        '''
        d[self.D_NAME] = self.__name
        d[self.D_FREQUENCY] = self.frequency
        d[self.D_IMAGE] = self.__image
        d[self.D_HOW_TO_ACCUMULATE] = self.__how_to_accumulate
        d[self.D_IMAGE_COUNT] = self.__image_count
        d[self.D_VSQUARED] = self.__vsquared
        d[self.D_VSUM] = self.__vsum
        d[self.D_POWER_IMAGE] = self.__power_image
        d[self.D_POWER_MASK] = self.__power_mask
        d[self.D_STACK_NUMBER] = self.__stack_number
        d[self.D_BRIGHT_MIN] = self.__bright_min
        d[self.D_BRIGHT_MAX] = self.__bright_max
        d[self.D_NORM0] = self.__norm0

    @staticmethod
    def restore_from_state(d):
        '''Create a provider from the state stored in the dictionary

        d - dictionary from call to save_state

        returns a new ImageProvider built from the saved state
        '''
        name = d[ImageProvider.D_NAME]
        frequency = d[ImageProvider.D_FREQUENCY]
        how_to_accumulate = d[ImageProvider.D_HOW_TO_ACCUMULATE]
        image_provider = ImageProvider(name, how_to_accumulate, frequency)
        image_provider.__image = d[ImageProvider.D_IMAGE]
        image_provider.__image_count = d[ImageProvider.D_IMAGE_COUNT]
        image_provider.__vsquared = d[ImageProvider.D_VSQUARED]
        image_provider.__vsum = d[ImageProvider.D_VSUM]
        image_provider.__power_image = d[ImageProvider.D_POWER_IMAGE]
        image_provider.__power_mask = d[ImageProvider.D_POWER_MASK]
        image_provider.__stack_number = d[ImageProvider.D_STACK_NUMBER]
        image_provider.__bright_min = d[ImageProvider.D_BRIGHT_MIN]
        image_provider.__bright_max = d[ImageProvider.D_BRIGHT_MAX]
        image_provider.__norm0 = d[ImageProvider.D_NORM0]
        return image_provider

    def reset(self):
        '''Reset accumulator at start of groups'''
        self.__image_count = None
        self.__image = None
        self.__cached_image = None
        self.__vsquared = None
        self.__vsum = None
        self.__power_image = None
        self.__power_mask = None
        self.__stack_number = 0
        self.__bright_max = None
        self.__bright_min = None

    @property
    def has_image(self):
        return self.__image_count is not None

    @property
    def count(self):
        return self.__image_count

    def set_image(self, image):
        self.__cached_image = None
        if image.has_mask:
            self.__image_count = image.mask.astype(int)
        else:
            self.__image_count = np.ones(image.pixel_data.shape[:2], int)

        if self.__how_to_accumulate == P_VARIANCE:
            self.__vsum = image.pixel_data.copy()
            self.__vsum[~ image.mask] = 0
            self.__image_count = image.mask.astype(int)
            self.__vsquared = self.__vsum.astype(np.float64) ** 2.0
            return

        if self.__how_to_accumulate == P_POWER:
            self.__vsum = image.pixel_data.copy()
            self.__vsum[~ image.mask] = 0
            self.__image_count = image.mask.astype(int)
            #
            # e**0 = 1, so the first image is always in the real plane
            #
            self.__power_mask = self.__image_count.astype(np.complex128).copy()
            self.__power_image = image.pixel_data.astype(np.complex128).copy()
            self.__stack_number = 1
            return
        if self.__how_to_accumulate == P_BRIGHTFIELD:
            self.__bright_max = image.pixel_data.copy()
            self.__bright_min = image.pixel_data.copy()
            self.__norm0 = np.mean(image.pixel_data)
            return

        if self.__how_to_accumulate == P_MASK:
            self.__image = image.mask
            return

        self.__image = image.pixel_data.copy()
        if image.has_mask:
            nan_value = 1 if self.__how_to_accumulate == P_MINIMUM else 0
            self.__image[~image.mask] = nan_value

    def accumulate_image(self, image):
        self.__cached_image = None
        if image.has_mask:
            self.__image_count += image.mask.astype(int)
        else:
            self.__image_count += 1
        if self.__how_to_accumulate in [P_AVERAGE, P_SUM]:
            if image.has_mask:
                self.__image[image.mask] += image.pixel_data[image.mask]
            else:
                self.__image += image.pixel_data
        elif self.__how_to_accumulate == P_MAXIMUM:
            if image.has_mask:
                self.__image[image.mask] = np.maximum(self.__image[image.mask],
                                                      image.pixel_data[image.mask])
            else:
                self.__image = np.maximum(image.pixel_data, self.__image)
        elif self.__how_to_accumulate == P_MINIMUM:
            if image.has_mask:
                self.__image[image.mask] = np.minimum(self.__image[image.mask],
                                                      image.pixel_data[image.mask])
            else:
                self.__image = np.minimum(image.pixel_data, self.__image)
        elif self.__how_to_accumulate == P_VARIANCE:
            mask = image.mask
            self.__vsum[mask] += image.pixel_data[mask]
            self.__vsquared[mask] += image.pixel_data[mask].astype(np.float64) ** 2
        elif self.__how_to_accumulate == P_POWER:
            multiplier = np.exp(2J * np.pi * float(self.__stack_number) /
                                self.frequency)
            self.__stack_number += 1
            mask = image.mask
            self.__vsum[mask] += image.pixel_data[mask]
            self.__power_image[mask] += multiplier * image.pixel_data[mask]
            self.__power_mask[mask] += multiplier
        elif self.__how_to_accumulate == P_BRIGHTFIELD:
            mask = image.mask
            norm = np.mean(image.pixel_data)
            pixel_data = image.pixel_data * self.__norm0 / norm
            max_mask = ((self.__bright_max < pixel_data) & mask)
            min_mask = ((self.__bright_min > pixel_data) & mask)
            self.__bright_min[min_mask] = pixel_data[min_mask]
            self.__bright_max[max_mask] = pixel_data[max_mask]
            self.__bright_min[max_mask] = self.__bright_max[max_mask]
        elif self.__how_to_accumulate == P_MASK:
            self.__image = self.__image & image.mask
        else:
            raise NotImplementedError("No such accumulation method: %s" %
                                      self.__how_to_accumulate)

    def provide_image(self, image_set):
        image_count = self.__image_count
        mask_2d = image_count > 0
        if self.__how_to_accumulate == P_VARIANCE:
            ndim_image = self.__vsquared
        elif self.__how_to_accumulate == P_POWER:
            ndim_image = self.__power_image
        elif self.__how_to_accumulate == P_BRIGHTFIELD:
            ndim_image = self.__bright_max
        else:
            ndim_image = self.__image
        if ndim_image.ndim == 3:
            image_count = np.dstack([image_count] * ndim_image.shape[2])
        mask = image_count > 0
        if self.__cached_image is not None:
            return self.__cached_image
        if self.__how_to_accumulate == P_AVERAGE:
            cached_image = self.__image / image_count
        elif self.__how_to_accumulate == P_VARIANCE:
            cached_image = np.zeros(self.__vsquared.shape, np.float32)
            cached_image[mask] = self.__vsquared[mask] / image_count[mask]
            cached_image[mask] -= self.__vsum[mask] ** 2 / (image_count[mask] ** 2)
        elif self.__how_to_accumulate == P_POWER:
            cached_image = np.zeros(image_count.shape, np.complex128)
            cached_image[mask] = self.__power_image[mask]
            cached_image[mask] -= (self.__vsum[mask] * self.__power_mask[mask] /
                                   image_count[mask])
            cached_image = \
                (cached_image * np.conj(cached_image)).real.astype(np.float32)
        elif self.__how_to_accumulate == P_BRIGHTFIELD:
            cached_image = np.zeros(image_count.shape, np.float32)
            cached_image[mask] = self.__bright_max[mask] - self.__bright_min[mask]
        elif self.__how_to_accumulate == P_MINIMUM and np.any(~mask):
            cached_image = self.__image.copy()
            cached_image[~mask] = 0
        else:
            cached_image = self.__image
        cached_image[~mask] = 0
        if np.all(mask) or self.__how_to_accumulate == P_MASK:
            self.__cached_image = cpi.Image(cached_image)
        else:
            self.__cached_image = cpi.Image(cached_image, mask=mask_2d)
        return self.__cached_image

    def get_name(self):
        return self.__name

    def release_memory(self):
        '''Don't discard the image at end of image set'''
        pass
