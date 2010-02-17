'''<b>Calculate Image Overlap </b> calculates the overlap of two image sets
<hr>
This module calculates overlap by determining precision, recall, F-factor, false positive rate, and false 
negative rate.  One image is considered the "ground truth" (possibly the result of hand-segmentation) and the other
is the "test image"; the images are determined to overlap most completely when the test image matches the ground
truth perfectly.  The module requires binary (black and white) input, where the objects that have been segmented 
are white and the background is black.  If you segment your images in CellProfiler using <b>IdentifyPrimaryObjects</b>, 
simply use <b>ConvertObjectsToImage</b> and select <i>Binary</i> as the color type.  

If your images have been segmented using other image processing software, or you have hand-segmented them in software 
such as Photoshop, you may need to use one or more of the following:
<ul>
<li> <b>ImageMath</b>: If the objects are black and the background is white, you must invert the intensity.</li>
<li> <b>ApplyThreshold</b>: If the image is grayscale, you must make it binary. </li>
<li> <b>ColorToGray</b>: If the image is in color, you must first convert it to grayscale, and then use <b>ApplyThreshold</b> to generate a binary image. </li>
</ul>

In the test image, any foreground (white) pixels that overlap with the foreground of the ground
truth will be considered "true positives", since they are correctly labeled as foreground.  Background (black) 
pixels that overlap with the background of the ground truth image are considered "true negatives", 
since they are correctly labeled as background.  A foreground pixel that overlaps with the background will
be considered a "false negative" (since it should have been labeled as part of the foreground), 
while a background pixel that overlaps with foreground in the ground truth will be considered a "false positive"
(since it was labeled as part of the foreground, but should not be).

This module measures the following:
<ul>
<li> False positive rate: total false positive pixels / total number of actual negative pixels </li>
<li> False negative rate: total false negative pixels / total number of actual postive pixels </li>
<li> Precision: true positive pixels / (true positive pixels + false positive pixels) </li>
<li> Recall: true positive pixels/ (true positive pixels + false negative pixels) </li>
<li> F- factor: 2 x (precision x recall)/(precision + recall) </li>
</ul>

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

__version__="$Revision: 9000 $"

import numpy as np
from contrib.english import ordinal

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps

C_IMAGE_OVERLAP = "Overlap"
FTR_F_FACTOR = "Ffactor"
FTR_PRECISION = "Precision"
FTR_RECALL = "Recall"
FTR_FALSE_POS_RATE = "FalsePosRate"
FTR_FALSE_NEG_RATE = "FalseNegRate"

FTR_ALL = [FTR_F_FACTOR, FTR_PRECISION, FTR_RECALL,
           FTR_FALSE_POS_RATE, FTR_FALSE_NEG_RATE]

class CalculateImageOverlap(cpm.CPModule):
    
    category = "Image Processing"
    variable_revision_number = 1
    module_name = "CalculateImageOverlap"

    def create_settings(self):
        self.ground_truth = cps.ImageNameSubscriber("Which image do you want to use as the basis for calculating the amount of overlap? ", "None", doc = 
                                                    '''This binary (black and white) image is known as the "ground truth" image.  It can be the product of hand-outlined segmentation, or
                                                    simply the result of another segmentation algorithm you would like to test.''')
        self.test_img = cps.ImageNameSubscriber("Which image do you want to compare for overlap?", "None", doc = ''' This 
                                                binary (black and white) image is the result of some image processing algorithm (either in CellProfiler 
                                                or any image processing software) that you would like to compare with the ground truth image.''')

    
    def settings(self):
        result = [self.ground_truth, self.test_img]
        return result

    def visible_settings(self):
        result = [self.ground_truth, self.test_img]
        return result

    def is_interactive(self):
        return False
    
    def run(self, workspace):
        '''Add the image overlap measurements'''
        
        image_set = workspace.image_set
        ground_truth_image = image_set.get_image(self.ground_truth.value,
                                                 must_be_binary = True)
        test_image = image_set.get_image(self.test_img.value,
                                         must_be_binary = True)
        ground_truth_pixels = ground_truth_image.pixel_data
        ground_truth_pixels = test_image.crop_image_similarly(ground_truth_pixels)
        mask = ground_truth_image.mask
        mask = test_image.crop_image_similarly(mask)
        if test_image.has_mask:
            mask = mask & test_image.mask
        test_pixels = test_image.pixel_data
        
        false_positives = test_pixels & ~ ground_truth_pixels
        false_positives[~ mask] = False
        false_negatives = (~ test_pixels) & ground_truth_pixels
        false_negatives[~ mask] = False
        true_positives = test_pixels & ground_truth_pixels
        true_positives[ ~ mask] = False
        true_negatives = (~ test_pixels) & (~ ground_truth_pixels)
        true_negatives[~ mask] = False
        
        false_positive_count = np.sum(false_positives)
        true_positive_count = np.sum(true_positives)
        
        false_negative_count = np.sum(false_negatives)
        true_negative_count = np.sum(true_negatives)
        
        labeled_pixel_count = true_positive_count + false_positive_count
        true_count = true_positive_count + false_negative_count
        
        ##################################
        #
        # Calculate the F-Factor
        #
        # 2 * precision * recall
        # -----------------------
        # precision + recall
        #
        # precision = true positives / labeled
        # recall = true positives / true count
        #
        ###################################
        
        if labeled_pixel_count == 0:
            precision = 1.0
        else:
            precision = float(true_positive_count) / float(labeled_pixel_count)
        if true_count == 0:
            recall = 1.0
        else:
            recall = float(true_positive_count) / float(true_count)
        f_factor = 2.0 * precision * recall / (precision + recall)
        negative_count = false_positive_count + true_negative_count
        if negative_count == 0:
            false_positive_rate = 0.0
        else:
            false_positive_rate = (float(false_positive_count) / 
                                   float(negative_count))
        if true_count == 0:
            false_negative_rate = 0.0
        else:
            false_negative_rate = (float(false_negative_count) / 
                                   float(true_count))
        
        m = workspace.measurements
        m.add_image_measurement(self.measurement_name(FTR_F_FACTOR), f_factor)
        m.add_image_measurement(self.measurement_name(FTR_PRECISION),
                                precision)
        m.add_image_measurement(self.measurement_name(FTR_RECALL), recall)
        m.add_image_measurement(self.measurement_name(FTR_FALSE_POS_RATE),
                                false_positive_rate)
        m.add_image_measurement(self.measurement_name(FTR_FALSE_NEG_RATE),
                                false_negative_rate)
        if workspace.frame is not None:
            workspace.display_data.true_positives = true_positives
            workspace.display_data.true_negatives = true_negatives
            workspace.display_data.false_positives = false_positives
            workspace.display_data.false_negatives = false_negatives
            workspace.display_data.statistics = [
                ("Measurement", "Value"),
                (FTR_F_FACTOR, f_factor),
                (FTR_PRECISION, precision),
                (FTR_RECALL, recall),
                (FTR_FALSE_POS_RATE, false_positive_rate),
                (FTR_FALSE_NEG_RATE, false_negative_rate)]
            
    def display(self, workspace):
        '''Display the image confusion matrix & statistics'''
        figure = workspace.create_or_find_figure(subplots=(2,3))
        for x, y, image, label in (
            (0, 0, workspace.display_data.true_positives, "True positives"),
            (0, 1, workspace.display_data.false_positives, "False positives"),
            (1, 0, workspace.display_data.false_negatives, "False negatives"),
            (1, 1, workspace.display_data.true_negatives, "True negatives")):
            figure.subplot_imshow_bw(x, y, image, title=label)
            
        figure.subplot_table(1, 2, workspace.display_data.statistics,
                             ratio = (.5, .5))

    def measurement_name(self, feature):
        return '_'.join((C_IMAGE_OVERLAP, feature, self.test_img.value))
    
    def get_categories(self, pipeline, object_name):
        '''Return the measurement categories for an object'''
        if object_name == cpmeas.IMAGE:
            return [ C_IMAGE_OVERLAP ]
        return []
    
    def get_measurements(self, pipeline, object_name, category):
        '''Return the measurements made for a category'''
        if object_name == cpmeas.IMAGE and category == C_IMAGE_OVERLAP:
            return FTR_ALL
        return []
    
    def get_measurement_images(self, pipeline, object_name, category, 
                               measurement):
        '''Return the images that were used when making the measurement'''
        if (object_name == cpmeas.IMAGE and category == C_IMAGE_OVERLAP and
            measurement in FTR_ALL):
            return [self.test_img.value]
        return []
    
    def get_measurement_columns(self, pipeline):
        '''Return database column information for each measurement'''
        return [ (cpmeas.IMAGE,
                  '_'.join((C_IMAGE_OVERLAP, feature, self.test_img.value)),
                  cpmeas.COLTYPE_FLOAT)
                 for feature in FTR_ALL]

    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        if from_matlab:
            # Variable revision # wasn't in Matlab file
            # All settings were identical to CP 2.0 v 1
            from_matlab = False
            variable_revision_number = 1
            
        return setting_values, variable_revision_number, from_matlab
