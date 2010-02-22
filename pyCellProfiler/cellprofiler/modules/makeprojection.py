'''<b>Make Projection</b> combines several two-dimensional images of 
the same field of view together, either by averaging or taking the 
maximum pixel value at each pixel position
<hr>
This module combines a set of images by averaging the pixel intensities
(or taking the maximum pixel intensity) at each pixel position. The 
process of averaging a Z-stack (3-D image stack) is known as making a projection.

The image is not immediately available in subsequent modules because the
output of this module is not complete until all image processing cycles have completed.

<h2>Technical notes:</h2>
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
as the field by which to group.
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
# Copyright 2003-2010
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
P_ALL = [P_AVERAGE, P_MAXIMUM]

class MakeProjection(cpm.CPModule):
    
    module_name = 'MakeProjection'
    category = 'Image Processing'
    variable_revision_number = 1
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
            'Select the input image','None', 
            doc = '''What did you call the images to be made into a projection?''')
        self.projection_type = cps.Choice('Type of projection',
                                          P_ALL, doc = '''
                                          What kind of projection would you like to make?
                                          <ul><li>Average: Use the average pixel intensity at each pixel position
                                          to create the final image.</li>
                                          <li>Maximum: Use the maximum pixel value at each pixel position to
                                          create the final image.</li></ul>''')
        self.projection_image_name = cps.ImageNameProvider(
            'Name the output image',
            'ProjectionBlue', 
            doc = '''What do you want to call the projected image?''',
            provided_attributes={cps.AGGREGATE_IMAGE_ATTRIBUTE: True,
                                 cps.AVAILABLE_ON_LAST_ATTRIBUTE: True } )

    def settings(self):
        return [self.image_name, self.projection_type, 
                self.projection_image_name]

    def prepare_run(self, pipeline, image_set_list, frame):
        return True
    
    def prepare_group(self, pipeline, image_set_list, grouping, image_numbers):
        '''Reset the aggregate image at the start of group processing'''
        if len(image_numbers) > 0:
            provider = ImageProvider(self.projection_image_name.value,
                                     self.projection_type.value)
            for image_number in image_numbers:
                image_set = image_set_list.get_image_set(image_number-1)
                assert isinstance(image_set, cpi.ImageSet)
                image_set.providers.append(provider)
        return True
        
    def run(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value)
        pixels = image.pixel_data
        provider = workspace.image_set.get_image_provider(self.projection_image_name.value)
        if (not provider.has_image):
            provider.set_image(image)
        else:
            provider.accumulate_image(image)
        if workspace.frame is not None:
            figure = workspace.create_or_find_figure(subplots=(2,1))
            provider_image = provider.provide_image(workspace.image_set)
            if provider_image.pixel_data.ndim == 3:
                figure.subplot_imshow(0, 0, image.pixel_data,
                                      self.image_name.value)
                figure.subplot_imshow(1, 0, provider_image.pixel_data,
                                      self.projection_image_name.value)
            else:
                figure.subplot_imshow_bw(0,0,image.pixel_data,
                                         self.image_name.value)
                figure.subplot_imshow_bw(1,0,provider_image.pixel_data,
                                         self.projection_image_name.value)

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
        return setting_values, variable_revision_number, from_matlab


class ImageProvider(cpi.AbstractImageProvider):
    """Provide the image after averaging but before dilation and smoothing"""
    def __init__(self, name, how_to_accumulate):
        """Construct using a parent provider that does the real work
        
        name - name of the image provided
        """
        super(ImageProvider, self).__init__()
        self.__name = name
        self.__image = None
        self.__how_to_accumulate = how_to_accumulate
        self.__image_count = None
        self.__cached_image = None
    
    def reset(self):
        '''Reset accumulator at start of groups'''
        self.__image_count = None
        self.__image = None
        self.__cached_image = None
        
    @property
    def has_image(self):
        return self.__image is not None
    
    def set_image(self, image):
        self.__image = image.pixel_data.copy()
        if image.has_mask:
            self.__image[~image.mask] = 0
            self.__image_count = image.mask.astype(int)
        else:
            self.__image_count = np.ones(image.pixel_data.shape, int)
    
    def accumulate_image(self, image):
        if self.__how_to_accumulate == P_AVERAGE:
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
        else:
            raise NotImplementedError("No such accumulation method: %s"%
                                      self.__how_to_accumulate)
        if image.has_mask:
            self.__image_count += image.mask.astype(int)
        else:
                self.__image_count += 1
            
        self.__cached_image = None
    
    def provide_image(self, image_set):
        if self.__cached_image is not None:
            return self.__cached_image
        if self.__how_to_accumulate == P_AVERAGE:
            cached_image = self.__image / self.__image_count
        else:
            cached_image = self.__image
        mask = self.__image_count > 0
        cached_image[~mask] = 0
        if np.all(mask):
            self.__cached_image = cpi.Image(cached_image)
        else:
            self.__cached_image = cpi.Image(cached_image, mask=mask)
        return self.__cached_image

    def get_name(self):
        return self.__name
    
    def release_memory(self):
        self.reset()


