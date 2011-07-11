'''<b>Make Projection</b> combines several two-dimensional images of 
the same field of view together, either by performing a mathematical operation
upon the pixel values at each pixel position.
<hr>

This module combines a set of images by performing a mathematic operation of your
choice at each pixel position; please refer to the settings help for more information on the available
operations. The process of averaging or summing a Z-stack (3D image stack) is known as making a projection.

The image is not immediately available in subsequent modules because the
output of this module is not complete until all image processing cycles have completed.

<h3>Technical notes</h3>
This module will create a projection of all images specified in <b>LoadImages</b>. 
Previously, the module <b>LoadImageDirectory</b> could be used for the same 
functionality, but on a per-folder basis; i.e., a projection would be created 
for each set of images in a folder, for all input folders. The
functionality of <b>LoadImageDirectory</b> can be achieved using image grouping with
metadata, with the following setting specifications in <b>LoadImages</b>:
<ol>
<li>Specify that all subfolders under the Default input folder are to be analyzed.</li>
<li>Extract metadata from the input image path by using a regular expression to capture
the subfolder name.</li>
<li>Enable grouping of image sets by metadata and specify the subfolder metadata token
as the field by which to group.</li>
</ol>
However, unlike <b>LoadImageDirectory</b>, this per-folder projection is also not 
immediately available in subsequent modules until all image processing cycles for 
the given subfolder have completed.

See also <b>LoadImages</b>.
'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2011
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__="$Revision$"

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps

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

class MakeProjection(cpm.CPModule):
    
    module_name = 'MakeProjection'
    category = 'Image Processing'
    variable_revision_number = 2
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
            'Select the input image','None', 
            doc = '''What did you call the images to be made into a projection?''')
        self.projection_type = cps.Choice(
            'Type of projection',
            P_ALL, doc = '''
            What kind of projection would you like to make? The final image can be created
            by the following methods:
            <ul><li><i>%(P_AVERAGE)s:</i> Use the average pixel intensity at each pixel position.</li>
            <li><i>%(P_MAXIMUM)s:</i> Use the maximum pixel value at each pixel position.</li>
            <li><i>%(P_MINIMUM)s:</i> Use the minimum pixel value at each pixel position.</li>
            <li><i>%(P_SUM)s:</i> Add the pixel values at each pixel position.</li>
            <li><i>%(P_VARIANCE)s:</i> Compute the variance at each pixel position. <br>
            The variance  method is described in "Selinummi J, Ruusuvuori P, Podolsky I, Ozinsky A, Gold E, et al. (2009), 
            "Bright Field Microscopy as an Alternative to Whole Cell Fluorescence in Automated Analysis of
            Macrophage Images", PLoS ONE 4(10): e7497 <a href="http://dx.doi.org/10.1371/journal.pone.0007497">(link)</a>.
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
            ''' % globals())
        self.projection_image_name = cps.ImageNameProvider(
            'Name the output image',
            'ProjectionBlue', 
            doc = '''What do you want to call the projected image?''',
            provided_attributes={cps.AGGREGATE_IMAGE_ATTRIBUTE: True,
                                 cps.AVAILABLE_ON_LAST_ATTRIBUTE: True } )
        self.frequency = cps.Float(
            "Frequency", 6.0, minval=1.0,
            doc = """
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

    def prepare_group(self, pipeline, image_set_list, grouping, image_numbers):
        '''Reset the aggregate image at the start of group processing'''
        if len(image_numbers) > 0:
            d = self.get_dictionary(image_set_list)
            if d.has_key(K_PROVIDER):
                provider = d[K_PROVIDER]
                provider.reset()
            else:
                provider = ImageProvider(self.projection_image_name.value,
                                         self.projection_type.value,
                                         self.frequency.value)
                d[K_PROVIDER] = provider
            for image_number in image_numbers:
                image_set = image_set_list.get_image_set(image_number-1)
                assert isinstance(image_set, cpi.ImageSet)
                image_set.providers.append(provider)
        return True
        
    def is_interactive(self):
        return False
    
    def run(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value)
        pixels = image.pixel_data
        provider = workspace.image_set.get_image_provider(self.projection_image_name.value)
        if (not provider.has_image):
            provider.set_image(image)
        else:
            provider.accumulate_image(image)
            
    def display(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value)
        provider = workspace.image_set.get_image_provider(self.projection_image_name.value)
        figure = workspace.create_or_find_figure(title="MakeProjection, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(2,1))
        provider_image = provider.provide_image(workspace.image_set)
        if provider_image.pixel_data.ndim == 3:
            figure.subplot_imshow(0, 0, image.pixel_data,
                                  self.image_name.value)
            figure.subplot_imshow(1, 0, provider_image.pixel_data,
                                  self.projection_image_name.value,
                                  sharex = figure.subplot(0,0),
                                  sharey = figure.subplot(0,0))
        else:
            figure.subplot_imshow_bw(0,0,image.pixel_data,
                                     self.image_name.value)
            figure.subplot_imshow_bw(1,0,provider_image.pixel_data,
                                     self.projection_image_name.value,
                                     sharex = figure.subplot(0,0),
                                     sharey = figure.subplot(0,0))
                
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
            setting_values = setting_values + [ "6" ]
        return setting_values, variable_revision_number, from_matlab


class ImageProvider(cpi.AbstractImageProvider):
    """Provide the image after averaging but before dilation and smoothing"""
    def __init__(self, name, how_to_accumulate, frequency = 6):
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
    
    def set_image(self, image):
        self.__cached_image = None
        if image.has_mask:
            self.__image_count = image.mask.astype(int)
        else:
            self.__image_count = np.ones(image.pixel_data.shape, int)
        
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
        if self.__how_to_accumulate in [P_AVERAGE,P_SUM]:
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
            raise NotImplementedError("No such accumulation method: %s"%
                                      self.__how_to_accumulate)
    
    def provide_image(self, image_set):
        mask = self.__image_count > 0
        if self.__cached_image is not None:
            return self.__cached_image
        if self.__how_to_accumulate == P_AVERAGE:
            cached_image = self.__image / self.__image_count
        elif self.__how_to_accumulate == P_VARIANCE:
            cached_image = np.zeros(self.__vsquared.shape, np.float32)
            cached_image[mask] = self.__vsquared[mask] / self.__image_count[mask]
            cached_image[mask] -= self.__vsum[mask]**2 / (self.__image_count[mask] ** 2)
        elif self.__how_to_accumulate == P_POWER:
            cached_image = np.zeros(self.__image_count.shape, np.complex128)
            cached_image[mask] = self.__power_image[mask]
            cached_image[mask] -= (self.__vsum[mask] * self.__power_mask[mask] /
                                   self.__image_count[mask])
            cached_image = (cached_image * np.conj(cached_image)).astype(np.float32)
        elif self.__how_to_accumulate == P_BRIGHTFIELD:
            cached_image = np.zeros(self.__image_count.shape, np.float32)
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
            self.__cached_image = cpi.Image(cached_image, mask=mask)
        return self.__cached_image

    def get_name(self):
        return self.__name
    
    def release_memory(self):
        '''Don't discard the image at end of image set'''
        pass


