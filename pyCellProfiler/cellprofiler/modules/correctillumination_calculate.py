"""correctillumination_calculate.py - correct illumination module pt I

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

__version__="$Revision$"

import numpy as np
import scipy.ndimage as scind
import scipy.linalg
import wx

import cellprofiler.cpimage  as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.preferences as cpp
import cellprofiler.cpmath.cpmorphology as cpmm
from cellprofiler.cpmath.smooth import smooth_with_function_and_mask
from cellprofiler.cpmath.smooth import circular_gaussian_kernel
from cellprofiler.cpmath.filter import median_filter

IC_REGULAR         = "Regular"
IC_BACKGROUND      = "Background"
RE_MEDIAN          = "Median"
EA_EACH            = "Each"
EA_ALL             = "All"
SRC_LOAD_IMAGES    = "Load Images module"
SRC_PIPELINE       = "Pipeline"
SM_NONE            = "No smoothing"
SM_FIT_POLYNOMIAL  = "Fit Polynomial"
SM_MEDIAN_FILTER   = "Median Filter"
SM_GAUSSIAN_FILTER = "Gaussian Filter"
SM_TO_AVERAGE      = "Smooth to Average"

FI_AUTOMATIC       = "Automatic"
FI_OBJECT_SIZE     = "Object size"
FI_MANUALLY        = "Manually"

ROBUST_FACTOR      = .02 # For rescaling, take 2nd percentile value

class CorrectIllumination_Calculate(cpm.CPModule):
    """SHORT DESCRIPTION:
Calculates an illumination function, used to correct uneven
illumination/lighting/shading or to reduce uneven background in images.
*************************************************************************

This module calculates an illumination function which can be saved to the
hard drive for later use (you should save in .mat format using the Save
Images module), or it can be immediately applied to images later in the
pipeline (using the CorrectIllumination_Apply module). This will correct
for uneven illumination of each image.

Illumination correction is challenging and we are writing a paper on it
which should help clarify it (TR Jones, AE Carpenter, P Golland, in
preparation). In the meantime, please be patient in trying to understand
this module.

Settings:

* Regular or Background intensities?

Regular intensities:
If you have objects that are evenly dispersed across your image(s) and
cover most of the image, then you can choose Regular intensities. Regular
intensities makes the illumination function based on the intensity at
each pixel of the image (or group of images if you are in All mode) and
is most often rescaled (see below) and applied by division using
CorrectIllumination_Apply. Note that if you are in Each mode or using a
small set of images with few objects, there will be regions in the
average image that contain no objects and smoothing by median filtering
is unlikely to work well.
Note: it does not make sense to choose (Regular + no smoothing + Each)
because the illumination function would be identical to the original
image and applying it will yield a blank image. You either need to smooth
each image or you need to use All images.

Background intensities:
If you think that the background (dim points) between objects show the
same pattern of illumination as your objects of interest, you can choose
Background intensities. Background intensities finds the minimum pixel
intensities in blocks across the image (or group of images if you are in
All mode) and is most often applied by subtraction using the
CorrectIllumination_Apply module.
Note: if you will be using the Subtract option in the
CorrectIllumination_Apply module, you almost certainly do NOT want to
Rescale! See below!!

* Each or All?
Enter Each to calculate an illumination function for each image
individually, or enter All to calculate the illumination function from
all images at each pixel location. All is more robust, but depends on the
assumption that the illumination patterns are consistent across all the
images in the set and that the objects of interest are randomly
positioned within each image. Applying illumination correction on each
image individually may make intensity measures not directly comparable
across different images.

* Dilation:
For some applications, the incoming images are binary and each object
should be dilated with a gaussian filter in the final averaged
(projection) image. This is for a sophisticated method of illumination
correction where model objects are produced.

* Smoothing Method:
If requested, the resulting image is smoothed. See the help for the
Smooth module for more details. If you are using Each mode, this is
almost certainly necessary. If you have few objects in each image or a
small image set, you may want to smooth. The goal is to smooth to the
point where the illumination function resembles a believable pattern.
That is, if it is a lamp illumination problem you are trying to correct,
you would apply smoothing until you obtain a fairly smooth pattern
without sharp bright or dim regions.  Note that smoothing is a
time-consuming process, and fitting a polynomial is fastest but does not
allow a very tight fit as compared to the slower median and gaussian 
filtering methods. We typically recommend median vs. gaussian because median 
is less sensitive to outliers, although the results are also slightly 
less smooth and the fact that images are in the range of 0 to 1 means that
outliers typically will not dominate too strongly anyway. A less commonly
used option is to *completely* smooth the entire image by choosing
"Smooth to average", which will create a flat, smooth image where every
pixel of the image is the average of what the illumination function would
otherwise have been.

* Approximate width of objects:
For certain smoothing methods, this will be used to calculate an adequate
filter size. If you don't know the width of your objects, you can use the
ShowOrHidePixelData image tool to find out or leave the word 'Automatic'
to calculate a smoothing filter simply based on the size of the image.


Rescaling:
The illumination function can be rescaled so that the pixel intensities
are all equal to or greater than one. This is recommended if you plan to
use the division option in CorrectIllumination_Apply so that the
corrected images are in the range 0 to 1. It is NOT recommended if you
plan to use the Subtract option in CorrectIllumination_Apply! Note that
as a result of the illumination function being rescaled from 1 to
infinity, if there is substantial variation across the field of view, the
rescaling of each image might be dramatic, causing the corrected images
to be very dark.

See also Average, CorrectIllumination_Apply, and Smooth modules.

    """

    variable_revision_number = 1
    category = "Image Processing"
    
    def create_settings(self):
        """Create the setting variables
        """
        self.module_name = "CorrectIllumination_Calculate"
        self.image_name = cps.ImageNameSubscriber("What did you call the images to be used to calculate the illumination function?","None")
        self.illumination_image_name = cps.ImageNameProvider("What do you want to call the illumination function?","IllumBlue")
        self.intensity_choice = cps.Choice("Do you want to calculate using regular intensities or background intensities?",
                                           [IC_REGULAR, IC_BACKGROUND],
                                           IC_REGULAR)
        self.dilate_objects = cps.Binary("Do you want to dilate objects in the final averaged image?",False)
        self.object_dilation_radius = cps.Integer("Enter the radius (roughly equal to the original radius of the objects).",1,0)
        self.block_size = cps.Integer("Enter the block size, which should be large enough that every square block of pixels is likely to contain some background pixels, where no objects are located.",60,1)
        self.rescale_option = cps.Choice("""Do you want to rescale the illumination function so that the pixel intensities are all equal to or greater than one (Y or N)? This is recommended if you plan to use the division option in CorrectIllumination_Apply so that the resulting images will be in the range 0 to 1. The "Median" option chooses the median value in the image to rescale so that division increases some values and decreases others.""",
                                         [cps.YES, cps.NO, RE_MEDIAN])
        self.each_or_all = cps.Choice("Enter Each to calculate an illumination function for Each image individually (in which case, choose Pipeline mode in the next box) or All to calculate an illumination function based on All the specified images to be corrected. See the help for details.",
                                      [EA_EACH,EA_ALL])
        self.smoothing_method = cps.Choice("Enter the smoothing method you would like to use, if any.",
                                           [SM_NONE, SM_FIT_POLYNOMIAL, 
                                            SM_MEDIAN_FILTER, 
                                            SM_GAUSSIAN_FILTER,
                                            SM_TO_AVERAGE])
        self.automatic_object_width = cps.Choice("Calculate the smoothing filter size automatically, relative to the width of artifacts to be smoothed or use a manually entered value?",
                                                 [FI_AUTOMATIC, FI_OBJECT_SIZE, FI_MANUALLY])
        self.object_width = cps.Integer("What is the approximate width of the artifacts to be smoothed (in pixels)?",10)
        self.size_of_smoothing_filter = cps.Integer("What is the size of the smoothing filter (in pixels)?",10)
        self.save_average_image = cps.Binary("Do you want to save the averaged image  (prior to dilation or smoothing)? (This is an image produced during the calculations - it is typically not needed for downstream modules)",False)
        self.average_image_name = cps.ImageNameProvider("What is the name of the averaged image?","IllumBlueAvg")
        self.save_dilated_image = cps.Binary("Do you want to save the image after dilation but prior to smoothing? (This is an image produced during the calculations - it is typically not needed for downstream modules)", False)
        self.dilated_image_name = cps.ImageNameProvider("What is the name of the dilated image?","IllumBlueDilated")

    def settings(self):
        return [ self.image_name, self.illumination_image_name,
                self.intensity_choice, self.dilate_objects,
                self.object_dilation_radius, self.block_size, 
                self.rescale_option, self.each_or_all, self.smoothing_method,
                self.automatic_object_width, self.object_width,
                self.size_of_smoothing_filter, self.save_average_image,
                self.average_image_name, self.save_dilated_image,
                self.dilated_image_name]

    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        """Adjust the setting values of old versions
        
        setting_values - sequence of strings that are the values for our settings
        variable_revision_number - settings were saved by module with this
                                   variable revision number
        module_name - name of module that did the saving
        from_matlab - True if it was the Matlab version that did the saving
        
        returns upgraded setting values, upgraded variable revision number
                and from_matlab flag
        
        Matlab variable revision numbers 6 and 7 supported.
        pyCellProfiler variable revision number 1 supported.
        """
        
        if from_matlab and variable_revision_number == 6:
            # Smoothing could be sum of squares or square of sums in 6,
            # could be Gaussian in 7 - arbitrarily, I've translated
            # the obsolete ones to be Gaussian
            new_setting_values = list(setting_values)
            if new_setting_values[8] in ("Sum of Squares","Square of Sum"):
                new_setting_values[8] = SM_GAUSSIAN_FILTER
            setting_values = new_setting_values
            
        if from_matlab and variable_revision_number == 7:
            # Convert Matlab variable order to ours
            new_setting_values = list(setting_values[:3])
            #
            # If object_dilation_radius is 0, then set self.dilate_objects
            # to false, otherwise true
            #
            if setting_values[3] == "0":
                new_setting_values.append(cps.NO)
            else:
                new_setting_values.append(cps.YES)
            #
            # We determine whether the input image is loaded from a file
            # or generated by the pipeline. In Matlab, setting # 8 (our 7)
            # made the user answer this question.
            #
            new_setting_values.extend(setting_values[3:7])
            new_setting_values.append(setting_values[8])
            #
            # set self.automatic_object_width based on settings 9 (the old
            # ObjectWidth setting) and 10 (the old SizeOfSmoothingFilter)
            #
            if setting_values[9] == FI_AUTOMATIC:
                new_setting_values.extend([FI_AUTOMATIC,"10","10"])
            elif (setting_values[10] == cps.DO_NOT_USE or
                  setting_values[10] == "/"):
                new_setting_values.extend([FI_OBJECT_SIZE,setting_values[9],"10"])
            else:
                new_setting_values.extend([FI_MANUALLY, setting_values[9],
                                           setting_values[10]])
            #
            # The optional output images: were "Do not use" if the user
            # didn't want them. Now it's two settings each.
            #
            for setting, name in zip(setting_values[11:],
                                     ("IllumBlueAvg","IllumBlueDilated")):
                if setting == cps.DO_NOT_USE:
                    new_setting_values.extend([cps.NO, name])
                else:
                    new_setting_values.extend([cps.YES, setting])
            setting_values = new_setting_values
            variable_revision_number = 1
            from_matlab = False
        
        return setting_values, variable_revision_number, from_matlab

    def visible_settings(self):
        """The settings as seen by the UI
        
        """
        result = [ self.image_name, self.illumination_image_name,
                  self.intensity_choice]
        if self.intensity_choice == IC_REGULAR:
            result += [self.dilate_objects]
            if self.dilate_objects.value:
                result += [ self.object_dilation_radius]
        else:
            result += [self.block_size]
        
        result += [ self.rescale_option, self.each_or_all,
                    self.smoothing_method]
        if self.smoothing_method in (SM_GAUSSIAN_FILTER, SM_MEDIAN_FILTER):
            result += [self.automatic_object_width]
            if self.automatic_object_width == FI_OBJECT_SIZE:
                result += [self.object_width]
            elif self.automatic_object_width == FI_MANUALLY:
                result += [self.size_of_smoothing_filter]
        result += [self.save_average_image]
        if self.save_average_image.value:
            result += [self.average_image_name]
        result += [self.save_dilated_image]
        if self.save_dilated_image.value:
            result += [self.dilated_image_name]
        return result

    def prepare_run(self, pipeline, image_set_list, frame):
        """Prepare the image set list for a run
        
        Calculate the illumination correction image for all images in
        the image set list if the image is loaded from a file and the
        user wants image correction over all images.
        """
        
        if self.each_or_all == EA_ALL:
            output_image_provider =\
                CorrectIlluminationImageProvider(self.illumination_image_name,
                                                 self)
            image_set_list.add_provider_to_all_image_sets(output_image_provider)
            if self.save_average_image.value:
                ap = CorrectIlluminationAvgImageProvider(self.average_image_name,
                                                         output_image_provider)
                image_set_list.add_provider_to_all_image_sets(ap)
            if self.save_dilated_image.value:
                dp = CorrectIlluminationDilatedImageProvider(self.average_image_name,
                                                             output_image_provider)
                image_set_list.add_provider_to_all_image_sets(dp)
            if self.is_source_loaded(pipeline):
                if frame != None:
                    progress_dialog = wx.ProgressDialog("#%d: CorrectIllumination_Calculate for %s"%(self.module_num, self.image_name),
                                                        "CorrectIllumination_Calculate is averaging %d images while preparing for run"%(image_set_list.count()),
                                                        image_set_list.count(),
                                                        frame,
                                                        wx.PD_APP_MODAL |
                                                        wx.PD_AUTO_HIDE |
                                                        wx.PD_CAN_ABORT)
 
                for i in range(image_set_list.count()):
                    image_set = image_set_list.get_image_set(i)
                    image     = image_set.get_image(self.image_name,
                                                    must_be_grayscale=True,
                                                    cache = False)
                    output_image_provider.add_image(image)
                    if frame != None:
                        should_continue, skip = progress_dialog.Update(i+1)
                        if not should_continue:
                            progress_dialog.EndModal(0)
                            return False
        return True
        
    def run(self, workspace):
        orig_image = workspace.image_set.get_image(self.image_name,
                                                   must_be_grayscale=True)
        pixels = orig_image.pixel_data
        if self.each_or_all == EA_ALL:
            output_image_provider = \
                workspace.image_set.get_image_provider(self.illumination_image_name)
            if not self.is_source_loaded(workspace.pipeline):
                #
                # We are accumulating a pipeline image. Add this image set's
                # image to the output image provider.
                # 
                output_image_provider.add_image(orig_image)
            if workspace.frame != None:
                avg_image = output_image_provider.provide_avg_image()
                dilated_image = output_image_provider.provide_dilated_image()
                output_image = output_image_provider.provide_image(workspace.image_set)
        else:
            avg_image       = self.preprocess_image_for_averaging(orig_image)
            dilated_image   = self.apply_dilation(avg_image, orig_image)
            smoothed_image  = self.apply_smoothing(dilated_image, orig_image)
            output_image    = self.apply_scaling(smoothed_image, orig_image)
            if self.save_average_image.value:
                workspace.image_set.add(self.average_image_name.value,
                                         avg_image)
            if self.save_dilated_image.value:
                workspace.image_set.add(self.dilated_image_name.value, 
                                        dilated_image)
            workspace.image_set.add(self.illumination_image_name.value,
                                    output_image)
        
        if workspace.frame != None:
            self.display(workspace, avg_image, dilated_image, output_image)
    
    def display(self, workspace, avg_image, dilated_image, output_image):
        figure = workspace.create_or_find_figure(subplots=(2,2))
        figure.subplot_imshow_grayscale(0, 0, avg_image.pixel_data, 
                                        "Averaged image")
        figure.subplot_imshow_grayscale(0, 1, output_image.pixel_data,
                                        "Final illumination function")
        figure.subplot_imshow_grayscale(1, 0, dilated_image.pixel_data,
                                        "Dilated image")
        statistics = [["Min value", round(np.min(output_image.pixel_data),2)],
                      ["Max value", round(np.max(output_image.pixel_data),2)],
                      ["Calculation type", self.intensity_choice.value]
                      ]
        if self.rescale_option == IC_REGULAR:
            statistics.append(["Radius",self.object_dilation_radius.value])
        else:
            statistics.append(["Block size",self.block_size.value])
        statistics.append(["Rescaling?", self.rescale_option.value])
        statistics.append(["Each or all?", self.each_or_all.value])
        statistics.append(["Smoothing method", self.smoothing_method.value])
        statistics.append(["Smoothing filter size",
                           round(self.smoothing_filter_size(output_image.pixel_data.size),2)])
        figure.subplot_table(1, 1, statistics, ratio=[.6,.4])

    def is_source_loaded(self, pipeline):
        """True if the image_name is provided by an image file loader"""
        for module in pipeline.modules():
            for setting in module.settings():
                if (isinstance(setting, cps.FileImageNameProvider) and
                    setting.value == self.image_name.value):
                    return True
        return False
    
    def apply_dilation(self, image, orig_image=None):
        """Return an image that is dilated according to the settings
        
        image - an instance of cpimage.Image
        
        returns another instance of cpimage.Image
        """
        if self.dilate_objects.value:
            #
            # This filter is designed to spread the boundaries of cells
            # and this "dilates" the cells
            #
            kernel = circular_gaussian_kernel(self.object_dilation_radius.value,
                                              self.object_dilation_radius.value*3)
            def fn(image):
                return scind.convolve(image, kernel, mode='constant', cval=0)
            dilated_pixels = smooth_with_function_and_mask(image.pixel_data,
                                                           fn, image.mask)
            return cpi.Image(dilated_pixels, parent_image = orig_image)
        else:
            return image
            
    def smoothing_filter_size(self, image_shape):
        """Return the smoothing filter size based on the settings and image size
        
        """
        if self.automatic_object_width == FI_MANUALLY:
            # Convert from full-width at half-maximum to standard deviation
            # (or so says CPsmooth.m)
            return self.size_of_smoothing_filter.value
        elif self.automatic_object_width == FI_OBJECT_SIZE:
            return self.object_width.value * 2.35 / 3.5
        elif self.automatic_object_width == FI_AUTOMATIC:
            return min(30, float(np.max(image_shape))/40.0)
    
    def preprocess_image_for_averaging(self, orig_image):
        """Create a version of the image appropriate for averaging
        
        """
        pixels = orig_image.pixel_data
        if self.intensity_choice == IC_REGULAR:
            if orig_image.has_mask:
                pixels[np.logical_not(orig_image.mask)] = 0
                avg_image = cpi.Image(pixels, parent_image = orig_image)
            else:
                avg_image = orig_image
        else:
            # For background, we create a labels image using the block
            # size and find the minimum within each block.
            labels, indexes = cpmm.block(pixels.shape,
                                         (self.block_size.value,
                                          self.block_size.value))
            if orig_image.has_mask:
                labels[np.logical_not(orig_image.mask)] = -1
            minima = scind.minimum(pixels, labels, indexes)
            minima = np.array(minima)
            min_block = np.zeros(pixels.shape)
            min_block[labels != -1] = minima[labels[labels != -1]]
            avg_image = cpi.Image(min_block, parent_image = orig_image)
        return avg_image
        
    def apply_smoothing(self, image, orig_image=None):
        """Return an image that is smoothed according to the settings
        
        image - an instance of cpimage.Image containing the pixels to analyze
        orig_image - the ancestor source image or None if ambiguous
        returns another instance of cpimage.Image
        """
        if self.smoothing_method == SM_NONE:
            return image
        
        pixel_data = image.pixel_data
        sigma = self.smoothing_filter_size(pixel_data.shape) / 2.35
        if self.smoothing_method == SM_FIT_POLYNOMIAL:
            mask = np.logical_and(image.mask,pixel_data > 0)
            if not np.any(mask):
                return image
            x,y = np.mgrid[0:pixel_data.shape[0],0:pixel_data.shape[1]]
            x2 = x*x
            y2 = y*y
            xy = x*y
            o  = np.ones(pixel_data.shape)
            a = np.array([x[mask],y[mask],x2[mask],y2[mask],xy[mask],o[mask]])
            coeffs = scipy.linalg.lstsq(a.transpose(),pixel_data[mask])[0]
            output_pixels = np.sum([coeff * index for coeff, index in
                                    zip(coeffs, [x,y,x2,y2,xy,o])],0)
        elif self.smoothing_method == SM_GAUSSIAN_FILTER:
            #
            # Smoothing with the mask is good, even if there's no mask
            # because the mechanism undoes the edge effects that are introduced
            # by any choice of how to deal with border effects.
            #
            def fn(image):
                return scind.gaussian_filter(image, sigma, 
                                             mode='constant', cval=0)
            output_pixels = smooth_with_function_and_mask(pixel_data, fn,
                                                          image.mask)
        elif self.smoothing_method == SM_MEDIAN_FILTER:
            filter_sigma = max(1, int(sigma+.5))
            output_pixels = median_filter(pixel_data, image.mask, filter_sigma)
        elif self.smoothing_method == SM_TO_AVERAGE:
            if image.has_mask:
                mean = np.mean(pixel_data[image.mask])
            else:
                mean = np.mean(pixel_data)
            output_pixels = np.ones(pixel_data.shape, pixel_data.dtype) * mean
        else:
            raise ValueError("Unimplemented smoothing method: %s:"%(self.smoothing_method.value))
        output_image = cpi.Image(output_pixels, parent_image = orig_image)
        return output_image

    def apply_scaling(self, image, orig_image=None):
        """Return an image that is rescaled according to the settings
        
        image - an instance of cpimage.Image
        returns another instance of cpimage.Image
        """
        if self.rescale_option == cps.NO:
            return image
        pixel_data = image.pixel_data
        if image.has_mask:
            sorted_pixel_data = pixel_data[np.logical_and(pixel_data>0,
                                                          image.mask)]
        else:
            sorted_pixel_data = pixel_data[pixel_data > 0]
        if sorted_pixel_data.shape[0] == 0:
            return image
        sorted_pixel_data.sort()
        if self.rescale_option == cps.YES:
            idx = int(sorted_pixel_data.shape[0] * ROBUST_FACTOR)
            robust_minimum = sorted_pixel_data[idx]
            pixel_data = pixel_data.copy()
            pixel_data[pixel_data < robust_minimum] = robust_minimum
        elif self.rescale_option == cps.MEDIAN:
            idx = int(sorted_pixel_data.shape[0]/2)
            robust_minimum = sorted_pixel_data[idx]
        if robust_minimum == 0:
            return image
        output_pixels = pixel_data / robust_minimum
        output_image = cpi.Image(output_pixels, parent_image = orig_image)
        return output_image
    
class CorrectIlluminationImageProvider(cpi.AbstractImageProvider):
    """CorrectIlluminationImageProvider provides the illumination correction image
    
    This class accumulates the image data from successive images and
    calculates the illumination correction image when asked.
    """
    def __init__(self, name, module):
        super(CorrectIlluminationImageProvider,self).__init__()
        self.__name = name
        self.__module = module
        self.__dirty = False
        self.__image_sum = None
        self.__cached_image = None
        self.__cached_avg_image = None
        self.__cached_dilated_image = None
        self.__cached_mask_count = None

    def add_image(self, image):
        """Accumulate the data from the given image
        
        image - an instance of cellprofiler.cpimage.Image, including
                image data and a mask
        """
        self.__dirty = True
        pimage = self.__module.preprocess_image_for_averaging(image)
        pixel_data = pimage.pixel_data
        if self.__image_sum == None:
            self.__image_sum = np.zeros(pixel_data.shape, 
                                        pixel_data.dtype)
            self.__mask_count = np.zeros(pixel_data.shape,
                                         np.int32)
        if image.has_mask:
            mask = image.mask
            self.__image_sum[mask] = self.__image_sum[mask] + pixel_data[mask]
            self.__mask_count[mask] = self.__mask_count[mask]+1
        else:
            self.__image_sum = self.__image_sum + pixel_data
            self.__mask_count = self.__mask_count+1

    def provide_image(self, image_set):
        if self.__dirty:
            self.calculate_image()
        return self.__cached_image

    def get_name(self):
        return self.__name

    def provide_avg_image(self):
        if self.__dirty:
            self.calculate_image()
        return self.__cached_avg_image
    
    def provide_dilated_image(self):
        if self.__dirty:
            self.calculate_image()
        return self.__cached_dilated_image
    
    def calculate_image(self):
        pixel_data = np.zeros(self.__image_sum.shape,
                              self.__image_sum.dtype)
        mask = self.__mask_count > 0
        pixel_data[mask] = self.__image_sum[mask] / self.__mask_count[mask]
        self.__cached_avg_image = cpi.Image(pixel_data, mask)
        self.__cached_dilated_image =\
            self.__module.apply_dilation(self.__cached_avg_image)
        smoothed_image =\
            self.__module.apply_smoothing(self.__cached_dilated_image)
        self.__cached_image = self.__module.apply_scaling(smoothed_image)
        self.__dirty = False
    
class CorrectIlluminationAvgImageProvider(cpi.AbstractImageProvider):
    """Provide the image after averaging but before dilation and smoothing"""
    def __init__(self, name, ci_provider):
        """Construct using a parent provider that does the real work
        
        name - name of the image provided
        ci_provider - a CorrectIlluminationProvider that does the actual
                      accumulation and calculation
        """
        super(CorrectIlluminationAvgImageProvider, self).__init__()
        self.__name = name
        self.__ci_provider = ci_provider
    
    def provide_image(self, image_set):
        return self.__ci_provider.provide_avg_image()

    def get_name(self):
        return self.__name

class CorrectIlluminationDilatedImageProvider(cpi.AbstractImageProvider):
    """Provide the image after averaging but before dilation and smoothing"""
    def __init__(self, name, ci_provider):
        """Construct using a parent provider that does the real work
        
        name - name of the image provided
        ci_provider - a CorrectIlluminationProvider that does the actual
                      accumulation and calculation
        """
        super(CorrectIlluminationDilatedImageProvider, self).__init__()
        self.__name = name
        self.__ci_provider = ci_provider
    
    def provide_image(self, image_set):
        return self.__ci_provider.provide_dilated_image()

    def get_name(self):
        return self.__name
