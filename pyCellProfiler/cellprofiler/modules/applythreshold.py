"""applythreshold.py

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
# TODO: Find out why this module makes python segfaults whenever the
# user switches from grayscale to binary.
# TODO: Review new settings with Anne.
# TODO: Update docstring to match the new settings.
# TODO: Test.
__version__="$Revision: 6746 $"

import wx
import matplotlib.cm
import matplotlib.backends.backend_wxagg

from cellprofiler.cpmodule import CPModule
from cellprofiler import cpimage
import cellprofiler.settings as cpsetting
from cellprofiler.gui import cpfigure
from cellprofiler.modules.identify import Identify, O_BACKGROUND, O_ENTROPY
from cellprofiler.modules.identify import O_FOREGROUND, O_THREE_CLASS
from cellprofiler.modules.identify import O_TWO_CLASS, O_WEIGHTED_VARIANCE
from cellprofiler.cpmath.threshold import TM_METHODS, TM_MANUAL, TM_MOG, TM_OTSU
from cellprofiler.cpmath.threshold import TM_PER_OBJECT, TM_BINARY_IMAGE

from cellprofiler.cpmath.cpmorphology import strel_disk
from scipy.ndimage.morphology import binary_dilation

RETAIN = "Retain"
SHIFT = "Shift"
GRAYSCALE = "Grayscale"
BINARY = "Binary (black and white)"

class ApplyThreshold(Identify):
    """Pixel intensity below or above a certain threshold is set to zero.

Settings:

When a pixel is thresholded, its intensity value is set to zero so that
it appears black.

If you wish to threshold dim pixels, change the value for which "Pixels
below this value will be set to zero". In this case, the remaining pixels
can retain their original intensity values or are shifted dimmer to
match the threshold used.

If you wish to threshold bright pixels, change the value for which
"Pixels above this value will be set to zero". In this case, you can
expand the thresholding around them by entering the number of pixels to
expand here: This setting is useful to adjust when you are attempting to
exclude bright artifactual objects: you can first set the threshold to
exclude these bright objects, but it may also be desirable to expand the
thresholded region around those bright objects by a certain distance so
as to avoid a 'halo' effect.
"""

    variable_revision_number = 3
    category = "Image Processing"

    def create_settings(self):
        threshold_methods = [method for method in TM_METHODS
                             if method != TM_BINARY_IMAGE]
        self.module_name = self.__class__.__name__
        self.image_name = cpsetting.NameSubscriber("Which image do you want to threshold?",
                                                   "imagegroup", "None")
        self.thresholded_image_name = cpsetting.NameProvider("What do you want to call the thresholded image?",
                                                             "imagegroup", "ThreshBlue")
        self.binary = cpsetting.Choice("What kind of image would you like to produce?", [GRAYSCALE, BINARY])

        # if not binary:
        self.low = cpsetting.Binary("Set pixels below a certain intensity to zero?", False)
        self.high = cpsetting.Binary("Set pixels above a certain intensity to zero?", False)
        # if not binary and self.low:
        self.low_threshold = cpsetting.Float("Set pixels below this value to zero", 0.0, minval=0, maxval=1)
        self.shift = cpsetting.Binary("Shift the remaining pixels' intensities down by the amount of the threshold?", False)
        # if not binary and self.high:
        self.high_threshold = cpsetting.Float("Set pixels above this value to zero", 1.0, minval=0, maxval=1)
        self.dilation = cpsetting.Float("Number of pixels by which to expand the thresholding around those excluded bright pixels",
                                        0.0)

        # if binary:
        self.manual_threshold = cpsetting.Float("Set pixels below this value to zero and set pixels at least this value to one.",
                                                0.5)
        self.threshold_method = cpsetting.Choice('''Select an automatic thresholding method or choose "Manual" to enter a threshold manually.  To choose a binary image, select "Binary image".''',
                                                 threshold_methods)
        self.threshold_range = cpsetting.FloatRange('Enter the lower and upper bounds for the threshold',(0,1),0,1)
        self.threshold_correction_factor = cpsetting.Float('Threshold correction factor', 1)
        self.object_fraction = cpsetting.CustomChoice('For MoG thresholding, what is the approximate fraction of image covered by objects?',
                                                      ['0.01','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','0.99'])
        self.enclosing_objects_name = cpsetting.ObjectNameSubscriber("What is the name of the objects to be used for per-object thresholding","None")
        self.two_class_otsu = cpsetting.Choice('Does your image have two classes of intensity value or three?',
                                               [O_TWO_CLASS, O_THREE_CLASS])
        self.use_weighted_variance = cpsetting.Choice('Do you want to minimize the weighted variance or the entropy?',
                                                [O_WEIGHTED_VARIANCE, O_ENTROPY])
        self.assign_middle_to_foreground = cpsetting.Choice("Assign pixels in the middle intensity class to the foreground or the background?",
                                                      [O_FOREGROUND, O_BACKGROUND])

    def visible_settings(self):
        vv = [self.image_name, self.thresholded_image_name, self.binary]
        if self.binary.value == GRAYSCALE:
            vv.append(self.low)
            if self.low.value:
                vv.extend([self.low_threshold, self.shift])
            vv.append(self.high)
            if self.high.value:
                vv.extend([self.high_threshold, self.dilation])
        else:
            vv.append(self.threshold_method)
            if self.threshold_method == TM_MANUAL:
                vv.append(self.manual_threshold)
            else:
                vv += [self.threshold_range, self.threshold_correction_factor]
                if self.threshold_algorithm == TM_MOG:
                    vv.append(self.object_fraction)
                if self.threshold_algorithm == TM_OTSU:
                    vv += [self.two_class_otsu, self.use_weighted_variance]
                    if self.two_class_otsu == O_THREE_CLASS:
                        vv.append(self.assign_middle_to_foreground)
                if self.threshold_modifier == TM_PER_OBJECT:
                    vv.append(self.enclosing_objects_name)
        return vv
    
    def settings(self):
        """Return all  settings in a consistent order"""
        return [self.image_name, self.thresholded_image_name,
                self.binary, self.low, self.high, self.low_threshold,
                self.shift, self.high_threshold, self.dilation,
                self.threshold_method, self.manual_threshold,
                self.threshold_range, self.threshold_correction_factor,
                self.object_fraction, self.enclosing_objects_name,
                self.two_class_otsu, self.use_weighted_variance,
                self.assign_middle_to_foreground]
    
    def backwards_compatibilize(self, setting_values,
                                variable_revision_number, module_name,
                                from_matlab):
        if from_matlab and variable_revision_number < 4:
            raise NotImplementedError, ("TODO: Handle Matlab CP pipelines for "
                                        "ApplyThreshold with revision < 4")
        if from_matlab and variable_revision_number == 4:
            setting_values = [ setting_values[0],  # ImageName
                                setting_values[1],  # ThresholdedImageName
                                None,
                                None,
                                None,
                                setting_values[2],  # LowThreshold
                                setting_values[3],  # Shift
                                setting_values[4],  # HighThreshold
                                setting_values[5],  # DilationValue
                                TM_MANUAL,          # Manual thresholding
                                setting_values[6],  # BinaryChoice
                                "0,1",              # Threshold range
                                "1",                # Threshold correction factor
                                ".2",               # Object fraction
                                "None"              # Enclosing objects name
                                ]
            setting_values[2] = (BINARY if float(setting_values[10]) > 0
                                 else GRAYSCALE) # binary flag
            setting_values[3] = (cpsetting.YES if float(setting_values[5]) > 0
                                 else cpsetting.NO) # low threshold set
            setting_values[4] = (cpsetting.YES if float(setting_values[7]) > 0
                                 else cpsetting.NO) # high threshold set
            variable_revision_number = 2
            from_matlab = False
        if (not from_matlab) and variable_revision_number == 1:
            setting_values = (setting_values[:9] + 
                              [TM_MANUAL, setting_values[9], "O,1", "1",
                               ".2","None"])
            variable_revision_number = 2
        if (not from_matlab) and variable_revision_number == 2:
            # Added Otsu options
            setting_values = list(setting_values)
            setting_values += [O_TWO_CLASS, O_WEIGHTED_VARIANCE,
                               O_FOREGROUND]
            variable_revision_number = 3
            
        return setting_values, variable_revision_number, from_matlab
        
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
        pixels = input.pixel_data.copy()
        if self.binary != 'Grayscale':
            if self.threshold_modifier == TM_PER_OBJECT:
                objects = workspace.object_set.get_objects(self.enclosing_objects_name.value)
                labels = objects.segmented
            else:
                labels = None
            local_thresh,ignore = self.get_threshold(pixels,input.mask,labels)
            pixels = ((pixels > local_thresh) & input.mask).astype(float)
        else:
            if self.low.value:
                thresholded_pixels = pixels < self.low_threshold.value
                pixels[input.mask & thresholded_pixels] = 0
                if self.shift.value:
                    pixels[input.mask & ~ thresholded_pixels] -= self.low_threshold.value
            if self.high.value:
                undilated = input.mask & (pixels >= self.high_threshold.value)
                dilated = binary_dilation(undilated, strel_disk(self.dilation.value), mask=input.mask)
                pixels[dilated] = 0
        output = cpimage.Image(pixels, input.mask)
        workspace.image_set.add(self.thresholded_image_name, output)
        if workspace.display:
            figure = workspace.create_or_find_figure(subplots=(1,2))

            left = figure.subplot(0,0)
            left.clear()
            left.imshow(input.pixel_data,matplotlib.cm.Greys_r)
            left.set_title("Original image: %s"%(self.image_name,))

            right = figure.subplot(0,1)
            right.clear()
            right.imshow(output.pixel_data,matplotlib.cm.Greys_r)
            right.set_title("Thresholded image: %s"%(self.thresholded_image_name,))
