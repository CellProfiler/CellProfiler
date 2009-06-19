"""rescaleintensity.py

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute of MIT and Harvard
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
# TODO: Verify that the mask is respected.
# TODO: Implement "Determine automatically from all"
# TODO: Implement "Text -- Divide by loaded text value"
# TODO: Test.
__version__="$Revision: 6746 $"

import numpy as np
import wx
import matplotlib.cm
import matplotlib.backends.backend_wxagg

from cellprofiler.cpmodule import CPModule
from cellprofiler import cpimage
import cellprofiler.settings as cpsetting
from cellprofiler.gui import cpfigure

METHOD_S = "Stretch 0 to 1"
METHOD_E = "Enter min/max below"
METHOD_G = "Greater than one"
METHOD_M = "Match maximum"
METHOD_T = "Text -- Divide by loaded text value"
AUTO_ALL = "Determine automatically from all of the images to be analyzed"
AUTO_EACH = "Determine automatically from each image independently"
MANUAL = "Manual"

class RescaleIntensity(CPModule):
    """Changes intensity range of an image to desired specifications.

The intensity of the incoming images are rescaled by one of several
methods. This is especially helpful for converting 12-bit images saved in
16-bit format to the correct range (see method E).

Settings:

Rescaling method:

Stretch 0 to 1: Stretch the image so that the minimum is zero and the
maximum is one.

Enter min/max below: Enter the minimum and maximum values of the
original image and the desired resulting image. Pixels are scaled from
their user-specified original range to a new user-specified range.  If
the user chooses "Determine automatically from each image
independently" then the highest and lowest pixel values will be
Automatically computed for each image by taking the maximum and
minimum pixel values in each image.  If the user chooses "Determine
automatically from all of the images to be analyzed" then the highest
and/or lowest pixel values will be automatically computed by taking
the maximum and minimum pixel values in all the images in the set.

The user also has the option of selecting the values that pixels
outside the original min/max range are set to, by entering numbers in
the "What value should pixels below/above the original intensity range
be mapped to" boxes. If you want these pixels to be set to the
highest/lowest rescaled intensity values, enter the same number in
these boxes as was entered in the highest/lowest rescaled intensity
boxes. However, using other values permits a simple form of
thresholding (e.g., setting the upper bounding value to 0 can be used
for removing bright pixels above a specified value)

To convert 12-bit images saved in 16-bit format to the correct range,
use the settings 0, 0.0625, 0, 1, 0, 1.  The value 0.0625 is
equivalent to 2^12 divided by 2^16, so it will convert a 16 bit image
containing only 12 bits of data to the proper range.

Greater than one: Rescale the image so that all pixels are equal to or
greater than one.

Match maximum: Match the maximum of one image to the maximum of
another.

Text -- Divide by loaded text value: Rescale by dividing by a value
loaded from a text file with LoadText.

See also SubtractBackground.
    """

    variable_revision_number = 1
    category = "Image Processing"

    def create_settings(self):
        self.module_name = self.__class__.__name__
        self.image_name = cpsetting.ImageNameSubscriber(
            "Which image do you want to rescale?", "None")
        self.rescaled_image_name = cpsetting.ImageNameProvider(
            "What do you want to call the rescaled image?", "RescaledBlue")
        self.method = cpsetting.Choice(
            "Which rescaling method do you want to use?",
            [METHOD_S, METHOD_E, METHOD_G, METHOD_M, METHOD_T])

        # if METHOD_E:
        self.low_orig = cpsetting.Choice(
            "What intensity from the original image should be set to the "
            "lowest value in the rescaled image?",
            [AUTO_ALL, AUTO_EACH, MANUAL])
        # if METHOD_E and low_orig == MANUAL:
        self.low_orig_manual = cpsetting.Float(
            "Manual value (range [0, 1])", 0.0, minval=0, maxval=1)
        # if METHOD_E:
        self.high_orig = cpsetting.Choice(
            "What intensity from the original image should be set to the "
            "highest value in the rescaled image?",
            [AUTO_ALL, AUTO_EACH, MANUAL])
        # if METHOD_E and high_orig == MANUAL:
        self.high_orig_manual = cpsetting.Float(
            "Manual value (range [0, 1])", 1.0, minval=0, maxval=1)
        
        # if METHOD_E:
        self.low_rescale = cpsetting.Float(
            "What value should pixels at the low end of the original "
            "intensity range be mapped to (range [0, 1])?", 0.0, 
            minval=0, maxval=1)
        # if METHOD_E:
        self.high_rescale = cpsetting.Float(
            "What value should pixels at the high end of the original "
            "intensity range be mapped to (range [0, 1])?", 1.0,
            minval=0, maxval=1)

        # if METHOD_E:
        self.low_pinned = cpsetting.Float(
            "What value should pixels below the original "
            "intensity range be mapped to (range [0, 1])?", 0.0,
            minval=0, maxval=1)
        # if METHOD_E:
        self.high_pinned = cpsetting.Float(
            "What value should pixels above the original "
            "intensity range be mapped to (range [0, 1])?", 1.0,
            minval=0, maxval=1)

        # if METHOD_M:
        self.other_image = cpsetting.ImageNameSubscriber(
            "What did you call the image whose maximum you want the rescaled "
            "image to match?", "None")

        # if METHOD_T:
        self.text_name = cpsetting.NameSubscriber(
            "What did you call the loaded text in the LoadText module?",
            "datagroup", "None")

    def visible_settings(self):
        vv = [self.image_name, self.rescaled_image_name, self.method]
        if self.method.value == METHOD_E:
            vv.append(self.low_orig)
            if self.low_orig.value == MANUAL:
                vv.append(self.low_orig_manual)
            vv.append(self.high_orig)
            if self.high_orig.value == MANUAL:
                vv.append(self.high_orig_manual)
            vv += [self.low_rescale, self.high_rescale, self.low_pinned,
                   self.high_pinned]
        elif self.method.value == METHOD_M:
            vv.append(self.other_image)
        elif self.method.value == METHOD_T:
            vv.append(self.text_name)
        return vv
    
    def settings(self):
        """Return all settings in a consistent order"""
        return [self.image_name, self.rescaled_image_name, self.method,
                self.low_orig, self.low_orig_manual, self.high_orig,
                self.high_orig_manual, self.low_rescale, self.high_rescale,
                self.low_pinned, self.high_pinned, self.other_image,
                self.text_name]
    
    def backwards_compatibilize(self, setting_values,
                                variable_revision_number, module_name,
                                from_matlab):
        if from_matlab and variable_revision_number < 4:
            raise NotImplementedError, ("TODO: Handle Matlab CP pipelines for "
                                        "RescaleIntensity with revision < 4")
        if from_matlab and variable_revision_number == 4:
            new = [setting_values[0],  # ImageName
                   setting_values[1],  # RescaledImageName
                   setting_values[3]]  # RescaleOption
            # LowestPixelOrig
            if setting_values[4] == "AA":
                new[4] = AUTO_ALL
                new[5] = 0
            elif setting_values[4] == "AE":
                new[4] = AUTO_EACH
                new[5] = 1
            else:
                new[4] = MANUAL
                new[5] = setting_values[4]
            # HighestPixelOrig
            if setting_values[5] == "AA":
                new[6] = AUTO_ALL
                new[7] = 0
            elif setting_values[5] == "AE":
                new[6] = AUTO_EACH
                new[7] = 1
            else:
                new[6] = MANUAL
                new[7] = setting_values[5]
            new += [setting_values[6:]]
            variable_revision_number = 1
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab

    def prepare_run(self, pipeline, image_set_list, frame):
        if self.method == METHOD_E:
            if self.low_orig.value == AUTO_ALL or \
                    self.high_orig.value == AUTO_ALL:
               if not pipeline.is_source_loaded(self.image_name.value):
                   raise ValueError, "Values can only be determined "
               "automatically from all images if the images are loaded "
               "directly from files (i.e., not preprocessed by other modules)."
               nimages = image_set_list.count()
               if frame != None:
                   progress_dialog = wx.ProgressDialog(
                       "#%d: RescaleIntensity for %s"%(self.module_num, 
                                                       self.image_name),
                       "RescaleIntensity is inspecting %d images to "
                       "determine values automatically"%(nimages,),
                       nimages, frame, 
                       wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT)
               for i in range(nimages):
                    image_set = image_set_list.get_image_set(i)
                    image = image_set.get_image(self.image_name, cache=False,
                                                must_be_grayscale=True)
                    if self.low_orig.value == AUTO_ALL:
                        low = image.pixel_data[image.mask].min()
                        if i == 0 or low < self.auto_low_orig:
                            self.auto_low_orig = low
                    if self.high_orig.value == AUTO_ALL:
                        high = image.pixel_data[image.mask].max()
                        if i == 0 or high > self.auto_high_orig:
                            self.auto_high_orig = high
                    if frame != None:
                        should_continue, skip = progress_dialog.Update(i+1)
                        if not should_continue:
                            progress_dialog.EndModal(0)
                            return False
        return True

        
    def run(self,workspace):
        """Run the module
        
        workspace    - the workspace contains:
            pipeline     - instance of CellProfiler.Pipeline for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - display within this frame (or None to not display)
        """
        input = workspace.image_set.get_image(self.image_name,
                                              must_be_grayscale=True)

        if self.method == METHOD_S:
            pixels = rescale_s(input.pixel_data)
        elif self.method == METHOD_E:
            if self.low_orig.value == AUTO_ALL:
                low_orig = self.auto_low_orig
            elif self.low_orig.value == AUTO_EACH:
                low_orig = input.pixel_data[input.mask].min()
            else:
                low_orig = self.low_orig_manual.value
            if self.high_orig.value == AUTO_ALL:
                high_orig = self.auto_high_orig
            elif self.high_orig.value == AUTO_EACH:
                high_orig = input.pixel_data[input.mask].max()
            else:
                high_orig = self.high_orig_manual.value
            pixels = rescale_e(input.pixel_data, low_orig, high_orig,
                               self.low_rescale.value, self.low_pinned.value,
                               self.high_rescale.value, self.high_pinned.value,
                               self.image_name.value)
        elif self.method == METHOD_G:
            pixels = rescale_g(input.pixel_data)
        elif self.method == METHOD_M:
            other = workspace.image_set.get_image(self.other_image, 
                                                  must_be_grayscale=True)
            pixels = rescale_m(input.pixel_data, other.pixel_data)
        elif self.method == METHOD_T:
            pixels = rescale_t(input.pixel_data, self.text_name.value)

        output = cpimage.Image(pixels, input.mask)
        workspace.image_set.add(self.rescaled_image_name, output)
        if workspace.display:
            figure = workspace.create_or_find_figure(subplots=(2, 1))
            figure.subplot_imshow(0, 0, input.pixel_data, 
                                  "Original image: %s"%(self.image_name,),
                                  colormap=matplotlib.cm.Greys_r,
                                  colorbar=True),
            figure.subplot_imshow(1, 0, output.pixel_data, 
                                  "Rescaled image: " + \
                                      self.rescaled_image_name.value,
                                  colormap=matplotlib.cm.Greys_r,
                                  colorbar=True)

def rescale_s(input):
    """The minimum of the image is brought to zero, whether it
     originally positive or negative.  maximum of the image is brought
     to 1."""
    tmp = input - input.min() 
    return tmp / tmp.max()

def rescale_m(input, other):
    """Rescales the image so the max equals the max of the other
    image."""
    if input.any():
        tmp = input / input.max()
    else:
        tmp = input
    return tmp * other.max()

def rescale_g(input):
    """Rescales the image so that all pixels are equal to or greater
    than one. This is done by dividing each pixel of the image by a
    scalar: the minimum pixel value anywhere in the smoothed
    image. (If the minimum value is zero, .0001 is substituted
    instead.) This rescales the image from 1 to some number. This is
    useful in cases where other images will be divided by this image,
    because it ensures that the final, divided image will be in a
    reasonable range, from zero to 1."""
    tmp = input / max(input.min(), 0.0001)
    tmp[tmp < 1] = 1
    return tmp

def rescale_t(input):
    return NotImplementedError

def rescale_e(input, low_orig, high_orig, low_rescale, low_pinned, 
              high_rescale, high_pinned, image_name):
    # (1) Scale and shift the original image to produce the rescaled
    # image.  Here, we find the linear transformation that maps the
    # user-specified old high/low values to their new high/low values.
    hi = high_orig
    HI = high_rescale;
    lo = low_orig
    LO = low_rescale;
    a = np.array([[low_orig, 1], [high_orig, 1]])
    b = np.array([low_rescale, high_rescale])
    X = np.linalg.solve(a, b)
    output = input * X[0] + X[1]

    # Make sure values close to EPS are mapped to 0 (since the matrix
    # algebra is not perfect).
    eps = np.finfo(float).eps
    output[np.logical_and(np.abs(output) > 0,
                          np.abs(output) < eps)] = 0
    
    # (2) Pixels above/below rescaled values are set to the desired
    # pinning values.
    output[output > high_rescale] = high_pinned;
    output[output < low_rescale] = low_pinned;
    
    return output
