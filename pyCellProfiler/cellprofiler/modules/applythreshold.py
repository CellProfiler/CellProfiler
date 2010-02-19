"""<b>Apply Threshold</b> sets pixel intensities below or above a certain threshold to zero
<hr>
<b>ApplyThreshold</b> allows you to produce either a grayscale or binary image
based on a threshold which can be pre-selected or calculated automatically using one of many methods.
"""

__version__="$Revision: 6746 $"


from cellprofiler.cpmodule import CPModule
from cellprofiler import cpimage
import cellprofiler.settings as cpsetting
from cellprofiler.modules.identify import Identify, O_BACKGROUND, O_ENTROPY
from cellprofiler.modules.identify import O_FOREGROUND, O_THREE_CLASS
from cellprofiler.modules.identify import O_TWO_CLASS, O_WEIGHTED_VARIANCE
from cellprofiler.modules.identify import FF_ORIG_THRESHOLD, FF_FINAL_THRESHOLD
from cellprofiler.modules.identify import FF_SUM_OF_ENTROPIES, FF_WEIGHTED_VARIANCE
from cellprofiler.modules.identify import get_threshold_measurement_columns
from cellprofiler.cpmath.threshold import TM_METHODS, TM_MANUAL, TM_MOG, TM_OTSU
from cellprofiler.cpmath.threshold import TM_PER_OBJECT, TM_BINARY_IMAGE

from cellprofiler.cpmath.cpmorphology import strel_disk
from scipy.ndimage.morphology import binary_dilation

RETAIN = "Retain"
SHIFT = "Shift"
GRAYSCALE = "Grayscale"
BINARY = "Binary (black and white)"

TH_BELOW_THRESHOLD = "Below threshold"
TH_ABOVE_THRESHOLD = "Above threshold"

class ApplyThreshold(Identify):

    module_name = "ApplyThreshold"
    variable_revision_number = 4
    category = "Image Processing"

    def create_settings(self):
        threshold_methods = [method for method in TM_METHODS
                             if method != TM_BINARY_IMAGE]
        self.image_name = cpsetting.NameSubscriber("Select the input image",
                                "imagegroup", "None", doc = '''
                                Which image do you want to threshold?''')
        
        self.thresholded_image_name = cpsetting.NameProvider("Name the output image",
                                "imagegroup", "ThreshBlue", doc = '''
                                What do you want to call the thresholded image?''')
        
        self.binary = cpsetting.Choice("Select the output image type", [GRAYSCALE, BINARY], doc = '''
                                What kind of output image would you like to produce?<br>
                                <ul>
                                <li><i>Grayscale:</i> The pixels that are retained after some pixels are set to zero or shifted (based on your selections for thresholding options) will have their original 
                                intensity values.</li>
                                <li><i>Binary:</i> The pixels that are retained after some pixels are set to zero (based on your selections for thresholding options) will be white and all other pixels will be black (zeroes).</li>
                                </ul>''')
        # if not binary:
        self.low_or_high = cpsetting.Choice(
                                "Set pixels below or above the threshold to zero?",
                                [TH_BELOW_THRESHOLD, TH_ABOVE_THRESHOLD],
                                doc="""For grayscale output, you can either set the dim pixels below 
                                the threshold to zero or set the bright pixels above the threshold to zero.
                                Choose <i>Below threshold</i> to threshold dim pixels and
                                <i>Above threshold</i> to threshold bright pixels.""")
        
        # if not binary and below threshold
        
        self.shift = cpsetting.Binary("Subtract the threshold value from the remaining pixel intensities?", False, doc ='''
                                <i>(Used only if the image is grayscale and pixels below a given intensity are to be set to zero)</i><br>
                                Use this setting if you would like the dim pixels to be shifted in value by the amount of the threshold.''')
        
        # if not binary and above threshold
        
        self.dilation = cpsetting.Float("Number of pixels by which to expand the thresholding around those excluded bright pixels",
                                0.0, doc = '''
                                <i>(Used only if the output image is grayscale and pixels above a given intensity are to be set to zero)</i><br>
                                This setting is useful when you are attempting to exclude bright artifactual objects: 
                                first, set the threshold to exclude these bright objects; it may also be desirable to expand the
                                thresholded region around those bright objects by a certain distance so as to avoid a "halo" effect.''')

        self.create_threshold_settings(threshold_methods)
        
        self.enclosing_objects_name = cpsetting.ObjectNameSubscriber("Select the input objects","None")
        
    def visible_settings(self):
        vv = [self.image_name, self.thresholded_image_name, self.binary]
        if self.binary.value == GRAYSCALE:
            vv.append(self.low_or_high)
            if self.low_or_high.value == TH_BELOW_THRESHOLD:
                vv.extend([self.shift])
            else:
                vv.extend([self.dilation])
        vv += self.get_threshold_visible_settings()
        if self.threshold_modifier == TM_PER_OBJECT:
            vv.append(self.enclosing_objects_name)
        return vv
    
    def settings(self):
        """Return all  settings in a consistent order"""
        return [self.image_name, self.thresholded_image_name,
                self.binary, self.low_or_high, 
                self.shift, self.dilation,
                self.threshold_method, self.manual_threshold,
                self.threshold_range, self.threshold_correction_factor,
                self.object_fraction, self.enclosing_objects_name,
                self.two_class_otsu, self.use_weighted_variance,
                self.assign_middle_to_foreground]
    
    def is_interactive(self):
        return False
    
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
        if self.threshold_modifier == TM_PER_OBJECT:
            objects = workspace.object_set.get_objects(self.enclosing_objects_name.value)
            labels = objects.segmented
        else:
            labels = None
        local_thresh,global_thresh = self.get_threshold(pixels,input.mask,labels)
        if self.binary != 'Grayscale':
            pixels = (pixels > local_thresh) & input.mask
        else:
            if self.low_or_high == TH_BELOW_THRESHOLD:
                thresholded_pixels = pixels < local_thresh
                pixels[input.mask & thresholded_pixels] = 0
                if self.shift.value:
                    pixels[input.mask & ~ thresholded_pixels] -= local_thresh
            elif self.low_or_high == TH_ABOVE_THRESHOLD:
                undilated = input.mask & (pixels >= local_thresh)
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
        self.add_threshold_measurements(workspace.measurements, 
                                        input.pixel_data, input.mask, 
                                        local_thresh, global_thresh,
                                        self.thresholded_image_name.value)
        if workspace.frame is not None:
            workspace.display_data.input_pixel_data = input.pixel_data
            workspace.display_data.output_pixel_data = output.pixel_data
            statistics = workspace.display_data.statistics = [
                ("Feature","Value")]
            
            for column in self.get_measurement_columns(workspace.pipeline):
                value = workspace.measurements.get_current_image_measurement(column[1])[0]
                statistics += [(column[1].split('_')[1], str(value))]
                
    def display(self, workspace):
        figure = workspace.create_or_find_figure(subplots=(1,3))

        figure.subplot_imshow_grayscale(0,0, workspace.display_data.input_pixel_data,
                              title = "Original image: %s" % 
                              self.image_name.value)

        figure.subplot_imshow_grayscale(0,1, workspace.display_data.output_pixel_data,
                              title = "Thresholded image: %s" %
                              self.thresholded_image_name.value)
        figure.subplot_table(0,2, workspace.display_data.statistics)
        
    def get_measurement_columns(self, pipeline):
        return get_threshold_measurement_columns(self.thresholded_image_name.value)
    
    def get_categories(self, pipeline, object_name):
        return self.get_threshold_categories(pipeline, object_name)
    
    def get_measurements(self, pipeline, object_name, category):
        return self.get_threshold_measurements(pipeline, object_name, category)
    
    def get_measurement_images(self, pipeline, object_name, category, measurement):
        return self.get_threshold_measurement_images(pipeline, object_name,
                                                     category, measurement,
                                                     self.thresholded_image_name.value)
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
            
        if (not from_matlab) and variable_revision_number == 3:
            #
            # Only low or high, not both + removed manual threshold settings
            #
            if setting_values[3] == cpsetting.YES:
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
                              
        return setting_values, variable_revision_number, from_matlab
        
