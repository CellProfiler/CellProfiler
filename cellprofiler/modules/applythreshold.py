"""<b>Apply Threshold</b> sets pixel intensities below or above a certain threshold to zero
<hr>
<b>ApplyThreshold</b> produces either a grayscale or binary image
based on a threshold which can be pre-selected or calculated automatically using one of many methods.
"""

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2014 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

from cellprofiler.cpmodule import CPModule
from cellprofiler import cpimage
import cellprofiler.settings as cps
from cellprofiler.settings import YES, NO
from identify import Identify, O_BACKGROUND, O_ENTROPY
from identify import O_FOREGROUND, O_THREE_CLASS
from identify import O_TWO_CLASS, O_WEIGHTED_VARIANCE
from identify import FF_ORIG_THRESHOLD, FF_FINAL_THRESHOLD
from identify import FF_SUM_OF_ENTROPIES, FF_WEIGHTED_VARIANCE
from identify import FI_IMAGE_SIZE, TSM_NONE
from cellprofiler.modules.identify import get_threshold_measurement_columns
from cellprofiler.cpmath.threshold import TM_METHODS, TM_MANUAL, TM_MOG, TM_OTSU
from cellprofiler.cpmath.threshold import TM_GLOBAL, TM_ADAPTIVE, TM_PER_OBJECT, TM_BINARY_IMAGE

from cellprofiler.cpmath.cpmorphology import strel_disk
from scipy.ndimage.morphology import binary_dilation

RETAIN = "Retain"
SHIFT = "Shift"
GRAYSCALE = "Grayscale"
BINARY = "Binary (black and white)"

TH_BELOW_THRESHOLD = "Below threshold"
TH_ABOVE_THRESHOLD = "Above threshold"

'''# of non-threshold settings in current revision'''
N_SETTINGS = 6

class ApplyThreshold(Identify):

    module_name = "ApplyThreshold"
    variable_revision_number = 7
    category = "Image Processing"

    def create_settings(self):
        threshold_methods = [method for method in TM_METHODS
                             if method != TM_BINARY_IMAGE]

        self.image_name = cps.ImageNameSubscriber(
            "Select the input image",doc = '''
            Choose the image to be thresholded.''')
        
        self.thresholded_image_name = cps.ImageNameProvider(
            "Name the output image",
            "ThreshBlue", doc = '''
            Enter a name for the thresholded image.''')
        
        self.binary = cps.Choice(
            "Select the output image type", [GRAYSCALE, BINARY], doc = '''
            Two types of output images can be produced:<br>
            <ul>
            <li><i>%(GRAYSCALE)s:</i> The pixels that are retained after some pixels 
            are set to zero or shifted (based on your selections for thresholding 
            options) will have their original 
            intensity values.</li>
            <li><i>%(BINARY)s:</i> The pixels that are retained after some pixels are 
            set to zero (based on your selections for thresholding options) will be 
            white and all other pixels will be black (zeroes).</li>
            </ul>'''%globals())
        
        # if not binary:
        self.low_or_high = cps.Choice(
            "Set pixels below or above the threshold to zero?",
            [TH_BELOW_THRESHOLD, TH_ABOVE_THRESHOLD], doc="""
            <i>(Used only when "%(GRAYSCALE)s" thresholding is selected)</i><br>
            For grayscale output, the dim pixels below 
            the threshold can be set to zero or the bright pixels above 
            the threshold can be set to zero.
            Choose <i>%(TH_BELOW_THRESHOLD)s</i> to threshold dim pixels and
            <i>%(TH_ABOVE_THRESHOLD)s</i> to threshold bright pixels."""%globals())
        
        # if not binary and below threshold
        
        self.shift = cps.Binary(
            "Subtract the threshold value from the remaining pixel intensities?", False, doc ='''
            <i>(Used only if the output image is %(GRAYSCALE)s and pixels below a given intensity are to be set to zero)</i><br>
            Select <i>%(YES)s</i> to shift the value of the dim pixels by the threshold value.'''%globals())
        
        # if not binary and above threshold
        
        self.dilation = cps.Float(
            "Number of pixels by which to expand the thresholding around those excluded bright pixels",
            0.0, doc = '''
            <i>(Used only if the output image is grayscale and pixels above a given intensity are to be set to zero)</i><br>
            This setting is useful when attempting to exclude bright artifactual objects: 
            first, set the threshold to exclude these bright objects; it may also be desirable to expand the
            thresholded region around those bright objects by a certain distance so as to avoid a "halo" effect.''')

        self.create_threshold_settings(threshold_methods)
        self.threshold_smoothing_choice.value = TSM_NONE
        
        
    def visible_settings(self):
        vv = [self.image_name, self.thresholded_image_name, self.binary]
        if self.binary.value == GRAYSCALE:
            vv.append(self.low_or_high)
            if self.low_or_high.value == TH_BELOW_THRESHOLD:
                vv.extend([self.shift])
            else:
                vv.extend([self.dilation])
        vv += self.get_threshold_visible_settings()
        return vv
    
    def settings(self):
        return [self.image_name, self.thresholded_image_name,
                self.binary, self.low_or_high, 
                self.shift, self.dilation] + self.get_threshold_settings()
    
    def help_settings(self):
        """Return all settings in a consistent order"""
        return [self.image_name, self.thresholded_image_name,
                self.binary, self.low_or_high, 
                self.shift, self.dilation] +\
               self.get_threshold_help_settings()
    
    def run(self,workspace):
        """Run the module
        
        workspace    - the workspace contains:
            pipeline     - instance of CellProfiler.Pipeline for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - display within this frame (or None to not display)
        """
        input = workspace.image_set.get_image(self.image_name.value,
                                              must_be_grayscale=True)
        pixels = input.pixel_data.copy()
        binary_image, local_thresh = self.threshold_image(
            self.image_name.value, workspace, wants_local_threshold=True)
        if self.binary != 'Grayscale':
            pixels = binary_image & input.mask
        else:
            if self.low_or_high == TH_BELOW_THRESHOLD:
                pixels[input.mask & ~ binary_image] = 0
                if self.shift.value:
                    pixels[input.mask & binary_image] -= \
                        local_thresh if self.threshold_modifier == TM_GLOBAL \
                        else local_thresh[input.mask & binary_image]
            elif self.low_or_high == TH_ABOVE_THRESHOLD:
                undilated = input.mask & binary_image
                dilated = binary_dilation(undilated, 
                                          strel_disk(self.dilation.value), 
                                          mask=input.mask)
                pixels[dilated] = 0
            else:
                raise NotImplementedError(
                    """Threshold setting, "%s" is not "%s" or "%s".""" %
                    (self.low_or_high.value, TH_BELOW_THRESHOLD, 
                     TH_ABOVE_THRESHOLD))
        output = cpimage.Image(pixels, parent_image=input)
        workspace.image_set.add(self.thresholded_image_name.value, output)
        if self.show_window:
            workspace.display_data.input_pixel_data = input.pixel_data
            workspace.display_data.output_pixel_data = output.pixel_data
            statistics = workspace.display_data.statistics = []
            workspace.display_data.col_labels = ("Feature", "Value")
            
            for column in self.get_measurement_columns(workspace.pipeline):
                value = workspace.measurements.get_current_image_measurement(column[1])
                statistics += [(column[1].split('_')[1], str(value))]
                
    def display(self, workspace, figure):
        figure.set_subplots((3, 1))

        figure.subplot_imshow_grayscale(0,0, workspace.display_data.input_pixel_data,
                              title = "Original image: %s" % 
                              self.image_name.value)

        figure.subplot_imshow_grayscale(1,0, workspace.display_data.output_pixel_data,
                              title = "Thresholded image: %s" %
                              self.thresholded_image_name.value, 
                              sharexy = figure.subplot(0,0))
        figure.subplot_table(
            2, 0, workspace.display_data.statistics,
            workspace.display_data.col_labels)
        
    def get_measurement_objects_name(self):
        '''Return the name of the "objects" used to name thresholding measurements
        
        In the case of ApplyThreshold, we use the image name to name the
        measurements, so the code here works, but is misnamed.
        '''
        return self.thresholded_image_name.value
    
    def get_measurement_columns(self, pipeline):
        return get_threshold_measurement_columns(self.thresholded_image_name.value)
    
    def get_categories(self, pipeline, object_name):
        return self.get_threshold_categories(pipeline, object_name)
    
    def get_measurements(self, pipeline, object_name, category):
        return self.get_threshold_measurements(pipeline, object_name, category)
    
    def get_measurement_images(self, pipeline, object_name, category, measurement):
        return self.get_threshold_measurement_objects(
            pipeline, object_name, category, measurement)
    def upgrade_settings(self, setting_values,
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
                                cps.NONE              # Enclosing objects name
                                ]
            setting_values[2] = (BINARY if float(setting_values[10]) > 0
                                 else GRAYSCALE) # binary flag
            setting_values[3] = (cps.YES if float(setting_values[5]) > 0
                                 else cps.NO) # low threshold set
            setting_values[4] = (cps.YES if float(setting_values[7]) > 0
                                 else cps.NO) # high threshold set
            variable_revision_number = 2
            from_matlab = False
        if (not from_matlab) and variable_revision_number == 1:
            setting_values = (setting_values[:9] + 
                              [TM_MANUAL, setting_values[9], "O,1", "1",
                               ".2",cps.NONE])
            variable_revision_number = 2
        if (not from_matlab) and variable_revision_number == 2:
            # Added Otsu options
            setting_values = list(setting_values)
            setting_values += [O_TWO_CLASS, O_WEIGHTED_VARIANCE,
                               O_FOREGROUND]
            variable_revision_number = 3
            
        if (not from_matlab) and variable_revision_number == 3:
            #
            # Only low or high, not both + removed manual threshold settings
            #
            if setting_values[3] == cps.YES:
                th = TH_BELOW_THRESHOLD
            else:
                th = TH_ABOVE_THRESHOLD
            if setting_values[2] == GRAYSCALE:
                # Grayscale used to have just manual thresholding
                setting_values = list(setting_values)
                setting_values[9] = TM_MANUAL
                if th == TH_BELOW_THRESHOLD:
                    # Set to old low threshold
                    setting_values[10] = setting_values[5]
                else:
                    setting_values[10] = setting_values[7]
            setting_values = [setting_values[0],  # Image name
                              setting_values[1],  # Thresholded image
                              setting_values[2],  # binary or gray
                              th,
                              setting_values[6],  # shift
                              ] +setting_values[8:]
            variable_revision_number = 4
            
        if (not from_matlab) and variable_revision_number == 4:
            # Added measurements to threshold methods
            setting_values = setting_values + [cps.NONE]
            variable_revision_number = 5
                              
        if (not from_matlab) and variable_revision_number == 5:
            # Added adaptive thresholding settings
            setting_values += [FI_IMAGE_SIZE, "10"]
            variable_revision_number = 6
            
        if (not from_matlab) and variable_revision_number == 6:
            image_name, thresholded_image_name, binary, low_or_high, \
                shift, dilation, threshold_method, manual_threshold, \
                threshold_range, threshold_correction_factor, \
                object_fraction, enclosing_objects_name, \
                two_class_otsu, use_weighted_variance, \
                assign_middle_to_foreground, thresholding_measurement = \
                setting_values[:16]
            setting_values = [
                image_name, thresholded_image_name, binary, low_or_high, 
                shift, dilation ] + self.upgrade_legacy_threshold_settings(
                    threshold_method, TSM_NONE, threshold_correction_factor, 
                    threshold_range, object_fraction, manual_threshold,
                    thresholding_measurement, cps.NONE, two_class_otsu,
                    use_weighted_variance, assign_middle_to_foreground,
                    FI_IMAGE_SIZE, "10", masking_objects=enclosing_objects_name)
            variable_revision_number = 7
        #
        # Upgrade the threshold settings
        #
        setting_values = setting_values[:N_SETTINGS] + \
            self.upgrade_threshold_settings(setting_values[N_SETTINGS:])
        return setting_values, variable_revision_number, from_matlab
        
