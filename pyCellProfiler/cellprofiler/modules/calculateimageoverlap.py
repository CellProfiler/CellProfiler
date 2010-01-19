'''<b>Calculate Image Overlap </b> calculates the overlap of two image sets
<hr>
This module calculates overlap by determining precision, recall, F-factor, false positive rate, and false 
negative rate.  One image is considered the "ground truth" (possibly the result of hand-segmentation) and the other
is the "test image", so the images are considered to overlap most completely when the test image matches the ground
truth perfectly.  The module requires binary (black and white) input, where the objects that have been segmented 
are white and the background is black.  If you segment your images in CellProfiler using IdentifyPrimaryObjects, 
simply use ConvertObjectsToImage and select Binary as the color type.  

If your images have been segmented using other image processing software, or you have hand-segmented them in software 
such as Photoshop, you may need to use one or more of the following:
<ul>
<li> ImageMath : if the objects are black and the background is white, you must invert the intensity</li>
<li> ApplyThreshold : if the image is grayscale, and you must make it binary </li>
<li> ColorToGray : if the image is color, you must first convert it to grayscale, and then use ApplyThreshold to generate a binary image </li>
</ul>

In the test image, any foreground (white) pixels that overlap with the foreground of the ground
truth will be considered "true positives", since they are correctly labeled as foreground.  Background (black) 
pixels that overlap with the background of the ground truth image are considered "true negatives", 
since they are correctly labeled as background.  A foreground pixel that overlaps with the background will
be considered a "false negative" (since it should have been labeled as part of the foreground), 
while a background pixel that overlaps with foreground in the ground truth will be considered a "false positive"
(since it was labeled as part of the foreground, but should not be).

This module measures:
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
import cellprofiler.settings as cps




class CalculateImageOverlap(cpm.CPModule):
    
    category = "Image Processing"
    variable_revision_number = 1
    module_name = "CalculateImageOverlap"

    def create_settings(self):
        self.ground_truth = cps.ImageNameSubscriber("Which image do you want to use as the basis for calculating the amount of overlap? ", "None", doc = 
                                                    '''This binary (black and white) image is known as the "ground truth" image.  It can be hand-outlined segmentation, or
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


    def run(self, workspace):
        #do stuffs
        return stuff
    
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        #matlab stuffs
        return setting_values, variable_revision_number, from_matlab
