'''<b>Identify Primary Objects</b> identifies objects in an image
<hr>

This module identifies primary objects in grayscale images
containing bright objects on a dark background. Primary objects (e.g., nuclei) are those that can be found in an image without using any corresponding reference image. By contrast, secondary objects are those that are found using previously-identified primary objects as a reference.

After processing, the window for this module will
show objects outlined in three colors:
<ul>
<li>Green: Acceptable; passed all criteria</li>
<li>Red: Discarded based on size</li>
<li>Yellow: Discarded due to touching the border</li>
</ul>

The module window will also show another image where the identified 
objects are displayed with arbitrary colors: the colors themselves do not mean 
anything but simply help you distingush the various objects. You can 
change the colormap in <i>File > Preferences</i>.

Steps to prepare images for this module:
<ul>
<li>If the objects in your images are dark on a light background, you 
should invert the images using the Invert operation in the <b>ImageMath</b> module.</li>
<li>If you are working with color images, they must first be converted to
grayscale using the <b>ColorToGray</b> module.</li>
</ul>

<h4>Available measurements</h4>
<ul>
<li><i>Image features:</i>
<ul>
<li><i>Count:</i> The number of primary objects identified.</li>
<li><i>OriginalThreshold:</i> The global threshold for the image.</li>
<li><i>FinalThreshold:</i> For the global threshold methods, this value is the
same as <i>OriginalThreshold</i>. For the adaptive or per-object methods, this
value is the mean of the local thresholds.</li>
<li><i>WeightedVariance:</i> The sum of the log-transformed variances of the 
foreground and background pixels, weighted by the number of pixels in 
each distribution.</li>
<li><i>SumOfEntropies:</i> The sum of entropies computed from the foreground and
background distributions.</li>
</ul>
</li>         
<li><i>Object features:</i>
<ul>
<li><i>Location_X, Location_Y:</i> The pixel (X,Y) coordinates of the primary 
object centroids. The centroid is calculated as the center of mass of the binary 
representation of the object.</li>
</ul>
</li>
</ul>

<h3>Technical notes</h3>

<p>CellProfiler contains a modular
three-step strategy to identify objects even if they touch each other. It
is based on previously published algorithms (<i>Malpica et
al., 1997; Meyer and Beucher, 1990; Ortiz de Solorzano et al., 1999;
Wahlby, 2003; Wahlby et al., 2004</i>). Choosing different options for each
of these three steps allows CellProfiler to flexibly analyze a variety of
different types of objects. The module has many
options, which vary in terms of speed and sophistication.
More detail can be found in the Settings section below.
Here are the three steps, using an example 
where the primary objects to be identified are nuclei:
<ol>
<li>CellProfiler determines whether an object is an individual
nucleus or two or more clumped nuclei.</li>
<li>The edges of nuclei are identified, using thresholding if the object 
is a single, isolated nucleus, and using more advanced options if the 
object is actually two or more nuclei that touch each other. </li>
<li>Some identified objects are discarded or merged together if
they fail to meet certain your specified criteria. For example, partial objects
at the border of the image can
be discarded, and small objects can be discarded or merged with nearby larger 
ones. A separate module,
<b>FilterObjects</b>, can further refine the identified nuclei, if
desired, by excluding objects that are a particular size, shape,
intensity, or texture. </li>
</ol>

References
<ul>
<li>Malpica N, de Solorzano CO, Vaquero JJ, Santos, A, Vallcorba I, 
Garcia-Sagredo JM, and del Pozo F (1997). "Applying watershed
algorithms to the segmentation of clustered nuclei." <i> Cytometry</i> 28,
289-297.</li>
<li>Meyer F, and Beucher S (1990). "Morphological segmentation." <i>J Visual
Communication and Image Representation</i> 1, 21-46.</li>
<li>Ortiz de Solorzano C, Rodriguez EG, Jones A, Pinkel D, Gray JW, 
Sudar D, and Lockett SJ. (1999). "Segmentation of confocal
microscope images of cell nuclei in thick tissue sections." <i>Journal of
Microscopy-Oxford</i> 193, 212-226.</li>
<li>Wahlby C (2003) <i>Algorithms for applied digital image cytometry</i>, Ph.D.,
Uppsala University, Uppsala.</li>
<li>Wahlby C, Sintorn IM, Erlandsson F, Borgefors G, and Bengtsson E. (2004). 
"Combining intensity, edge and shape information for 2D and 3D
segmentation of cell nuclei in tissue sections." <i>J Microsc</i> 215, 67-76.</li>
</ul>

<h3>Technical notes</h3> 
The initial step of identifying local maxima is
performed on the user-controlled heavily smoothed image, the
foreground/background is  on a hard-coded slightly smoothed image,
and the dividing lines between clumped objects (watershed) are done on the
non-smoothed image.

<h3>Special note on saving images</h3> 
<p>Using the settings in this module, you can pass object outlines along to the
module <b>OverlayOutlines</b> and then save them with the <b>SaveImages</b> module. 
You can also pass along the objects themselves to the object processing module 
<b>ConvertToImage</b> and then save them with the <b>SaveImages module</b>.

See also <b>IdentifySecondaryObjects</b>, <b>IdentifyTertiaryObjects</b>, and <b>IdentifyPrimManual</b>.
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

import math
import scipy.ndimage
import scipy.sparse
import numpy as np
import re
import scipy.stats

import identify as cpmi
import cellprofiler.cpmodule
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
import cellprofiler.preferences as cpp
from cellprofiler.cpmath.otsu import otsu
from cellprofiler.cpmath.cpmorphology import fill_labeled_holes, strel_disk
from cellprofiler.cpmath.cpmorphology import binary_shrink, relabel
from cellprofiler.cpmath.cpmorphology import is_local_maximum
from cellprofiler.cpmath.filter import stretch, laplacian_of_gaussian
from cellprofiler.cpmath.watershed import fast_watershed as watershed
from cellprofiler.cpmath.propagate import propagate
from cellprofiler.cpmath.smooth import smooth_with_noise
import cellprofiler.cpmath.outline
import cellprofiler.objects
from cellprofiler.settings import AUTOMATIC
import cellprofiler.cpmath.threshold as cpthresh
from identify import FF_FINAL_THRESHOLD, FF_ORIG_THRESHOLD
from identify import FF_SUM_OF_ENTROPIES, FF_WEIGHTED_VARIANCE
from identify import draw_outline
from identify import FI_IMAGE_SIZE
from cellprofiler.gui.help import HELP_ON_MEASURING_DISTANCES

IMAGE_NAME_VAR                  = 0
OBJECT_NAME_VAR                 = 1
SIZE_RANGE_VAR                  = 2
EXCLUDE_SIZE_VAR                = 3
MERGE_CHOICE_VAR                = 4
EXCLUDE_BORDER_OBJECTS_VAR      = 5
THRESHOLD_METHOD_VAR            = 6
THRESHOLD_CORRECTION_VAR        = 7
THRESHOLD_RANGE_VAR             = 8
OBJECT_FRACTION_VAR             = 9
UNCLUMP_METHOD_VAR              = 10
UN_INTENSITY                    = "Intensity"
UN_SHAPE                        = "Shape"
UN_LOG                          = "Laplacian of Gaussian"
UN_NONE                         = "None"
WATERSHED_VAR                   = 11
WA_INTENSITY                    = "Intensity"
WA_SHAPE                        = "Shape"
WA_PROPAGATE                    = "Propagate"
WA_NONE                         = "None"
SMOOTHING_SIZE_VAR              = 12
MAXIMA_SUPPRESSION_SIZE_VAR     = 13
LOW_RES_MAXIMA_VAR              = 14
SAVE_OUTLINES_VAR               = 15
FILL_HOLES_OPTION_VAR           = 16
TEST_MODE_VAR                   = 17
AUTOMATIC_SMOOTHING_VAR         = 18
AUTOMATIC_MAXIMA_SUPPRESSION    = 19
MANUAL_THRESHOLD_VAR            = 20
BINARY_IMAGE_VAR                = 21
MEASUREMENT_THRESHOLD_VAR       = 22

LIMIT_NONE = "Continue"
LIMIT_TRUNCATE = "Truncate"
LIMIT_ERASE = "Erase"

class IdentifyPrimaryObjects(cpmi.Identify):
            
    variable_revision_number = 9
    category =  "Object Processing"
    module_name = "IdentifyPrimaryObjects"
    
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
            "Select the input image",doc="""
            What did you call the images you want to use to identify objects?""")
        
        self.object_name = cps.ObjectNameProvider(
            "Name the primary objects to be identified",
            "Nuclei",doc="""
            What do you want to call the objects identified by this module?""")
        
        self.size_range = cps.IntegerRange(
            "Typical diameter of objects, in pixel units (Min,Max)", 
            (10,40), minval=1, doc='''\
            Most options within this module use this estimate of the
            size range of the objects in order to distinguish them from noise in the
            image. For example, for some of the identification methods, the smoothing
            applied to the image is based on the minimum size of the objects. The units
            here are pixels so that it is easy to zoom in on objects and determine
            typical diameters. %(HELP_ON_MEASURING_DISTANCES)s Note that for non-round objects, the 
            diameter here is actually the "equivalent diameter", i.e.,
            the diameter of a circle with the same area as the object.'''%globals())
        
        self.exclude_size = cps.Binary(
            "Discard objects outside the diameter range?",
            True, doc='''\
            You can choose to discard objects outside the range you specified. 
            This allows you to exclude small objects (e.g., dust, noise,
            and debris) or large objects (e.g., large clumps) if desired. Objects discarded 
            based on size are outlined in red in the module's display. See also the
            <b>FilterObjects</b> module to further discard objects based on some
            other measurement.''')
        
        self.merge_objects = cps.Binary(
            "Try to merge too small objects with nearby larger objects?", 
            False, doc='''\
            Use caution when choosing <i>Yes</i>. Objects that are
            smaller than the specified minimum diameter will be merged, if possible, with
            other surrounding objects. This is helpful in cases when an object was
            incorrectly split into two objects, one of which is actually just a tiny
            piece of the larger object. However, this could be problematic if the other
            settings in the module are set poorly, producing many tiny objects; the module
            will take a very long time trying to merge the tiny objects back together again; you may
            not notice that this is the case, since it may successfully piece together the
            objects again. It is therefore a good idea to run the
            module first without merging objects to make sure the settings are
            reasonably effective.''')
        
        self.exclude_border_objects = cps.Binary(
            "Discard objects touching the border of the image?", 
            True, doc='''\
            You can choose to discard objects that touch the border of the image.
            This is useful in cases when you do not want to make measurements of
            objects that are not fully within the field of view (because, for
            example, the morphological measurements would not be accurate).  Objects 
            discarded due to border touching are outlined in yellow in the module's display.
            Note that if a per-object thresholding method is used, objects that touch the 
            border of the cropped region will be discarded with this setting
            checked.''')
        
        self.create_threshold_settings()
        
        self.unclump_method = cps.Choice(
            'Method to distinguish clumped objects', 
            [UN_INTENSITY, UN_SHAPE, UN_LOG, UN_NONE], doc="""\
            This setting allows you to choose the method that is used to segment
            objects, i.e., "declump" a large, merged object into the appropriate number of  
            objects of interest. To decide between these methods, you can run Test mode to see the results of each.
            <ul>
            <li><i>Intensity:</i> For objects that tend to have only one peak of brightness
            per object (e.g. objects that are brighter towards their interiors and 
            dimmer towards their edges), this
            option counts each intensity peak as a separate object. The objects can
            be any shape, so they need not be round and uniform in size as would be
            required for the <i>Shape</i> option. The module is more successful when
            the objects have a smooth texture. By default, the image is automatically
            blurred to attempt to achieve appropriate smoothness (see <i>Blur</i> option),
            but overriding the default value can improve the outcome on
            lumpy-textured objects. Technical description: Object centers are defined
            as local intensity maxima in the smoothed image.</li>
            <li><i>Shape:</i> For cases when there are definite indentations separating
            objects. This works best for objects that are round. The intensity
            patterns in the original image are largely irrelevant: the image is converted to
            black and white (binary) and the shape determines whether clumped
            objects will be distinguished. Therefore, the cells need not be brighter
            towards the interior as is required for the <i>Intensity</i> option. The
            declumping results of this method are affected by the thresholding
            method you choose. Technical description: The binary thresholded image is
            distance-transformed and object centers are defined as peaks in this
            image. A distance-transform gives each pixel a value equal to the distance 
            to the nearest pixel below a certain threshold, so it indicates the <i>Shape</i> of the object.</li>
            <li><i>Laplacian of Gaussian (LoG):</i> For objects that have an increasing intensity
            gradient toward their center, this option performs an LoG (Mexican hat)
            transform on the image, which accentuates pixels that are local maxima of a desired size. It
            thresholds the result and finds pixels that are both local maxima and above
            threshold. These pixels are used as the seeds for objects in the watershed.</li>
            <li><i>None:</i> If objects are well separated and bright relative to the
            background, it may be unnecessary to attempt to separate clumped objects.
            Using the very fast <i>None</i> option, a simple threshold will be used to identify
            objects. This will override any declumping method chosen in the next
            question.</li></ul>""")

        self.watershed_method = cps.Choice(
            'Method to draw dividing lines between clumped objects', 
            [WA_INTENSITY, WA_SHAPE, WA_PROPAGATE, WA_NONE], doc="""\
            This setting allows you to choose the method that is used to draw the line
            bewteen segmented objects, provided that you have chosen to declump the objects.
            To decide between these methods, you can run Test mode to see the results of each.
            <ul><li><i>Intensity:</i> Works best where the dividing lines between clumped
            objects are dimmer than the remainder of the objects. Technical description: 
            Using the previously identified local maxima as seeds, this method is a
            watershed (<i>Vincent and Soille, 1991</i>) on the intensity image.</li>
            <li><i>Shape:</i> Dividing lines between clumped objects are based on the
            shape of the clump. For example, when a clump contains two objects, the
            dividing line will be placed where indentations occur between the two
            objects. The intensity patterns in the original image are largely irrelevant: the
            cells need not be dimmer along the lines between clumped objects.
            Technical description: Using the previously identified local maxima as seeds, 
            this method is a
            watershed on the distance-transformed thresholded image.</li>
            <li><i>Propagate</i>: This method uses a propagation algorithm
            instead of a watershed. The image is ignored and the pixels are
            assigned to the objects by repeatedly adding unassigned pixels to
            the objects that are immediately adjacent to them. This method
            is suited in cases such as objects with branching extensions,
            for instance neurites, where the goal is to trace outward from
            the cell body along the branch, assigning pixels in the branch
            along the way.</li>
            <li><i>None</i>: If objects are well separated and bright relative to the
            background, it may be unnecessary to attempt to separate clumped objects.
            Using the very fast <i>None</i> option, a simple threshold will be used to identify
            objects. This will override any declumping method chosen in the previous
            question.</li></ul>""")
        
        self.automatic_smoothing = cps.Binary(
            'Automatically calculate size of smoothing filter?', 
            True, doc="""\
            <i>(Used only when distinguishing between clumped objects)</i> <br>
            This setting,
            along with the <i>Minimum allowed distance between local maxima</i> setting, affects whether objects
            close to each other are considered a single object or multiple objects.
            It does not affect the dividing lines between an object and the
            background. The size of the smoothing filter is automatically 
            calculated based on the specified minimum object diameter
            that you have entered. If you see too many objects merged that ought to be separate
            or too many objects split up that
            ought to be merged, you may want to override the automatically
            calculated value.""")
        
        self.smoothing_filter_size = cps.Integer(
            'Size of smoothing filter', 10, doc="""\
            <i>(Used only when distinguishing between clumped objects)</i> <br>
            If you see too many objects merged that ought to be separated
            (under-segmentation), the <i>Size of smoothing filter</i> value 
            should be lower. If you see too many 
            objects split up that ought to be merged (over-segmentation), the 
            value should be higher. Enter 0 to prevent any image smoothing in certain 
            cases; for example, for low resolution images with small objects 
            ( &lt; ~5 pixels in diameter).

            <p>Reducing the texture of objects by increasing the
            smoothing increases the chance that each real, distinct object has only
            one peak of intensity but also increases the chance that two distinct
            objects will be recognized as only one object. Note that increasing the
            size of the smoothing filter increases the processing time exponentially.</p>""")

        self.automatic_suppression = cps.Binary(
            'Automatically calculate minimum allowed distance between local maxima?', 
            True, doc="""\
            <i>(Used only when distinguishing between clumped objects)</i><br>
            This setting, along with the <i>Size of the smoothing filter</i> setting,
            affects whether objects close to each other are considered a single object
            or multiple objects. It does not affect the dividing lines between an object and the
            background. Local maxima that are closer together than the minimum 
            allowed distance will be suppressed (the local intensity histogram is smoothed to 
            remove the peaks within that distance). The distance can be automatically 
            calculated based on the minimum object diameter that you have entered,
            but if you see too many objects merged that ought to be separate, or
            too many objects split up that ought to be merged, you may want to override the
            automatically calculated value.""")
        
        self.maxima_suppression_size = cps.Integer(
            'Suppress local maxima that are closer than this minimum allowed distance', 
            7, doc="""\
            <i>(Used only when distinguishing between clumped objects)</i><br>
            Enter a positive integer, in pixel units. If you see too many objects 
            merged that ought to be separated (under-segmentation), the value 
            should be lower. If you see too many objects split up that ought to 
            be merged (over-segmentation), the value should be higher.
            <p>The maxima suppression distance
            should be set to be roughly equivalent to the minimum radius of a real
            object of interest. Any distinct "objects" which are found but
            are within two times this distance from each other will be assumed to be
            actually two lumpy parts of the same object, and they will be merged.</p>""")
        
        self.low_res_maxima = cps.Binary(
            'Speed up by using lower-resolution image to find local maxima?', 
            True, doc="""\
            <i>(Used only when distinguishing between clumped objects)</i><br> 
            Note that if you have entered a minimum object diameter of 10 or less, setting this option to
            <i>Yes</i> will have no effect.""")

        self.should_save_outlines = cps.Binary(
            'Retain outlines of the identified objects?', False)
        
        self.save_outlines = cps.OutlineNameProvider(
            'Name the outline image',"PrimaryOutlines", doc="""\
            <i>(Used only if outlines are to be saved)</i><br>
            You can use the outlines of the identified objects in modules downstream,
            by selecting them from any drop-down image list.""")
        
        self.fill_holes = cps.Binary(
            'Fill holes in identified objects?', True, doc="""
            Checking this box will cause holes interior to identified objects
            to be filled.""")
        
        self.test_mode = cps.Binary(
            'Run in test mode where each method for distinguishing clumped objects is compared?', False)
        
        self.wants_automatic_log_threshold = cps.Binary(
            'Automatically calculate the threshold using the Otsu method?', True)
        
        self.manual_log_threshold = cps.Float('Enter Laplacian of '
                                              'Gaussian threshold', .5, 0, 1)
        
        self.wants_automatic_log_diameter = cps.Binary(
            'Automatically calculate the size of objects '
            'for the Laplacian of Gaussian filter?', True,
            doc="""\
            <i>(Used only when applying the LoG thresholding method)</i><br>
            Check this box to use the filtering diameter range above 
            when constructing the LoG filter. Uncheck the
            box in order to enter a size that is not related to the filtering 
            size. You may want to specify a custom size if you want to filter 
            using loose criteria, but have objects that are generally of 
            similar sizes.""")
        
        self.log_diameter = cps.Float(
            'Enter LoG filter diameter', 
            5, minval=1, maxval=100,
            doc="""\
            <i>(Used only when applying the LoG thresholding method)</i><br>
            The size to use when calculating the LoG filter. The filter enhances the local maxima of objects 
            whose diameters are roughly the entered number or smaller.""")
        
        self.limit_choice = cps.Choice(
            "Handling of objects if excessive number of objects identified",
            [LIMIT_NONE, LIMIT_TRUNCATE, LIMIT_ERASE],
            doc = """This setting deals with images that are segmented
            into an unreasonable number of objects. This might happen if
            the module calculates a low threshold or if the image has
            unusual artifacts. <b>IdentifyPrimaryObjects</b> can handle
            this condition in one of three ways:
            <br><ul><li><i>%(LIMIT_NONE)s</i>: Don't check for large numbers
            of objects.</li>
            <li><i>%(LIMIT_TRUNCATE)s</i>: Limit the number of objects.
            Arbitrarily erase objects to limit the number to the maximum
            allowed.</li>
            <li><i>%(LIMIT_ERASE)s</i>: Erase all objects if the number of
            objects exceeds the maximum. This results in an image with
            no primary objects. This option is a good choice if a large
            number of objects indicates that the image should not be
            processed.</li></ul>""" % globals())
        
        self.maximum_object_count = cps.Integer(
            "Maximum number of objects",
            value = 500,
            minval = 2,
            doc = """<i>(Used only when handling images with large numbers of objects by truncating)</i> <br>
            This setting limits the number of objects in the
            image. See the documentation for the previous setting 
            for details.""")

    def settings(self):
        return [self.image_name,self.object_name,self.size_range,
                self.exclude_size, self.merge_objects,
                self.exclude_border_objects, self.threshold_method,
                self.threshold_correction_factor, self.threshold_range,
                self.object_fraction, self.unclump_method,
                self.watershed_method, self.smoothing_filter_size,
                self.maxima_suppression_size, self.low_res_maxima,
                self.save_outlines, self.fill_holes, 
                self.automatic_smoothing, self.automatic_suppression,
                self.manual_threshold, self.binary_image,
                self.should_save_outlines,
                self.wants_automatic_log_threshold,
                self.manual_log_threshold,
                self.two_class_otsu, self.use_weighted_variance,
                self.assign_middle_to_foreground,
                self.wants_automatic_log_diameter, self.log_diameter,
                self.limit_choice, self.maximum_object_count,
                self.thresholding_measurement,
                self.adaptive_window_method, self.adaptive_window_size]
    
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        """Upgrade the strings in setting_values dependent on saved revision
        
        """
        if variable_revision_number == 12 and from_matlab:
            # Translating from Matlab:
            #
            # Variable # 16 (LaplaceValues) removed
            # Variable # 19 (test mode) removed
            #
            # Added automatic smoothing / suppression checkboxes
            # Added checkbox for setting manual threshold
            # Added checkbox for thresholding using a binary image
            # Added checkbox instead of "DO_NOT_USE" for saving outlines
            new_setting_values = list(setting_values[:18])
            #
            # Remove the laplace values setting
            #
            del new_setting_values[15]
            # Automatic smoothing checkbox - replace "Automatic" with
            # a number
            if setting_values[SMOOTHING_SIZE_VAR] == cps.AUTOMATIC:
                new_setting_values += [cps.YES]
                new_setting_values[SMOOTHING_SIZE_VAR] = '10'
            else:
                new_setting_values += [cps.NO]
            #
            # Automatic maxima suppression size
            #
            if setting_values[MAXIMA_SUPPRESSION_SIZE_VAR] == cps.AUTOMATIC:
                new_setting_values += [cps.YES]
                new_setting_values[MAXIMA_SUPPRESSION_SIZE_VAR] = '5'
            else:
                new_setting_values += [cps.NO]
            if not setting_values[THRESHOLD_METHOD_VAR] in cpthresh.TM_METHODS:
                # Try to figure out what the user wants if it's not one of the
                # pre-selected choices.
                try:
                    # If it's a floating point number, then the user
                    # was trying to type in a manual threshold
                    ignore = float(setting_values[THRESHOLD_METHOD_VAR])
                    new_setting_values[THRESHOLD_METHOD_VAR] = cpthresh.TM_MANUAL
                    # Set the manual threshold to be the contents of the
                    # old threshold method variable and ignore the binary mask
                    new_setting_values += [setting_values[THRESHOLD_METHOD_VAR],
                                           cps.DO_NOT_USE]
                except:
                    # Otherwise, assume that it's the name of a binary image
                    new_setting_values[THRESHOLD_METHOD_VAR] = cpthresh.TM_BINARY_IMAGE
                    new_setting_values += [ '0.0',
                                           setting_values[THRESHOLD_METHOD_VAR]]
            else:
                new_setting_values += [ '0.0',
                                       setting_values[THRESHOLD_METHOD_VAR]]
            #
            # The object fraction is stored as a percent in Matlab (sometimes)
            #
            m = re.match("([0-9.])%",setting_values[OBJECT_FRACTION_VAR])
            if m:
                setting_values[OBJECT_FRACTION_VAR] = str(float(m.groups()[0]) / 100.0)
            #
            # Check the "DO_NOT_USE" status of the save outlines variable
            # to get the value for should_save_outlines
            #
            if new_setting_values[SAVE_OUTLINES_VAR] == cps.DO_NOT_USE:
                new_setting_values += [ cps.NO ]
                new_setting_values[SAVE_OUTLINES_VAR] = "None"
            else:
                new_setting_values += [ cps.YES ]
            setting_values = new_setting_values
            if new_setting_values[UNCLUMP_METHOD_VAR] == cps.DO_NOT_USE:
                new_setting_values[UNCLUMP_METHOD_VAR] = UN_NONE
            if new_setting_values[WATERSHED_VAR] == cps.DO_NOT_USE:
                new_setting_values[WATERSHED_VAR] = WA_NONE
            variable_revision_number = 1
            from_matlab = False
        if (not from_matlab) and variable_revision_number == 1:
            # Added LOG method
            setting_values = list(setting_values)
            setting_values += [ cps.YES, ".5" ]
            variable_revision_number = 2
        
        if (not from_matlab) and variable_revision_number == 2:
            # Added Otsu options
            setting_values = list(setting_values)
            setting_values += [cpmi.O_TWO_CLASS, cpmi.O_WEIGHTED_VARIANCE,
                               cpmi.O_FOREGROUND]
            variable_revision_number = 3
        
        if (not from_matlab) and variable_revision_number == 3:
            # Added more LOG options
            setting_values = setting_values + [cps.YES, "5"]
            variable_revision_number = 4
        
        if (not from_matlab) and variable_revision_number == 4:
            # Added # of object limits
            setting_values = setting_values + [LIMIT_NONE, "500"]
            variable_revision_number = 5
            
        if (not from_matlab) and variable_revision_number == 5:
            # Changed object number limit option from "No action" to "Continue"
            if setting_values[-2] == "No action":
                setting_values[-2] = LIMIT_NONE
            variable_revision_number = 6
            
        if (not from_matlab) and variable_revision_number == 6:
            # Added measurements to threshold method
            setting_values = setting_values + ["None"]
            variable_revision_number = 7
        if (not from_matlab) and variable_revision_number == 7:
            # changed DISTANCE to SHAPE
            if setting_values[11] == "Distance":
                setting_values[11] = "Shape"
            variable_revision_number = 8
            
        if (not from_matlab) and variable_revision_number == 8:
            # Added adaptive thresholding settings
            setting_values += [FI_IMAGE_SIZE, "10"]
            variable_revision_number = 9
            
        return setting_values, variable_revision_number, from_matlab
            
    def help_settings(self):
        return [self.image_name, 
                self.object_name,
                self.size_range,
                self.exclude_size, 
                self.merge_objects,
                self.exclude_border_objects, 
                self.threshold_method,
                self.binary_image,
                self.manual_threshold,
                self.thresholding_measurement, 
                self.two_class_otsu, 
                self.use_weighted_variance,
                self.assign_middle_to_foreground,
                self.object_fraction, 
                self.adaptive_window_method,
                self.adaptive_window_size,
                self.wants_automatic_log_diameter,
                self.log_diameter,
                self.wants_automatic_log_threshold,
                self.manual_log_threshold,
                self.threshold_correction_factor, 
                self.threshold_range,
                self.unclump_method,
                self.watershed_method, 
                self.automatic_smoothing, 
                self.smoothing_filter_size,
                self.automatic_suppression,
                self.maxima_suppression_size,
                self.low_res_maxima,
                self.should_save_outlines,
                self.save_outlines, 
                self.fill_holes, 
                self.limit_choice,
                self.maximum_object_count ]

    def visible_settings(self):
        vv = [self.image_name,self.object_name,self.size_range,
              self.exclude_size, self.merge_objects,
              self.exclude_border_objects] + self.get_threshold_visible_settings()
        vv += [ self.unclump_method ]
        if self.unclump_method != UN_NONE:
            if self.unclump_method == UN_LOG:
                vv += [self.wants_automatic_log_threshold]
                if not self.wants_automatic_log_threshold.value:
                    vv += [self.manual_log_threshold]
                vv += [self.wants_automatic_log_diameter]
                if not self.wants_automatic_log_diameter.value:
                    vv += [self.log_diameter]
            vv += [self.watershed_method, self.automatic_smoothing]
            if not self.automatic_smoothing.value:
                vv += [self.smoothing_filter_size]
            vv += [self.automatic_suppression]
            if not self.automatic_suppression.value:
                vv += [self.maxima_suppression_size]
            vv += [self.low_res_maxima]
        vv += [self.should_save_outlines]
        if self.should_save_outlines.value:
            vv += [self.save_outlines]
        vv += [self.fill_holes, self.limit_choice]
        if self.limit_choice != LIMIT_NONE:
            vv += [self.maximum_object_count]
        return vv
    
    def is_interactive(self):
        return False

    def run(self,workspace):
        """Run the module
        
        pipeline     - instance of CellProfiler.Pipeline for this run
        workspace    - contains
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
        """
        #
        # Retrieve the relevant image and mask
        #
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_grayscale = True)
        img = image.pixel_data
        mask = image.mask
        #
        # Get a threshold to use for labeling
        #
        if self.threshold_modifier == cpthresh.TM_PER_OBJECT:
            masking_objects = image.labels
        else:
            masking_objects = None
        if self.threshold_method == cpthresh.TM_BINARY_IMAGE:
            binary_image = workspace.image_set.get_image(self.binary_image.value,
                                                         must_be_binary = True)
            local_threshold = np.ones(img.shape)
            local_threshold[binary_image.pixel_data] = 0
            global_threshold = otsu(img[mask],
                        self.threshold_range.min,
                        self.threshold_range.max)
        else:
            local_threshold,global_threshold = self.get_threshold(img, mask,
                                                              masking_objects, workspace)
        blurred_image = self.smooth_image(img,mask,1)
        binary_image = np.logical_and((blurred_image >= local_threshold),mask)
        #
        # Fill background holes inside foreground objects
        #
        if self.fill_holes.value:
            binary_image = fill_labeled_holes(binary_image)

        labeled_image,object_count = scipy.ndimage.label(binary_image,
                                                         np.ones((3,3),bool))
        labeled_image,object_count,maxima_suppression_size,LoG_threshold,LoG_filter_diameter = \
            self.separate_neighboring_objects(img, mask, 
                                              labeled_image,
                                              object_count,
                                              local_threshold)
        unedited_labels = labeled_image.copy()
        # Filter out objects touching the border or mask
        border_excluded_labeled_image = labeled_image.copy()
        labeled_image = self.filter_on_border(image, labeled_image)
        border_excluded_labeled_image[labeled_image > 0] = 0
        
        # Filter out small and large objects
        size_excluded_labeled_image = labeled_image.copy()
        labeled_image, small_removed_labels = \
            self.filter_on_size(labeled_image,object_count)
        size_excluded_labeled_image[labeled_image > 0] = 0
        
        # Relabel the image
        labeled_image,object_count = relabel(labeled_image)
        new_labeled_image, new_object_count = self.limit_object_count(
            labeled_image, object_count)
        if new_object_count < object_count:
            # Add the labels that were filtered out into the border
            # image.
            border_excluded_mask = ((border_excluded_labeled_image > 0) |
                                    ((labeled_image > 0) & 
                                     (new_labeled_image == 0)))
            border_excluded_labeled_image = scipy.ndimage.label(border_excluded_mask,
                                                                np.ones((3,3),bool))[0]
            object_count = new_object_count
            labeled_image = new_labeled_image
        
        # Make an outline image
        outline_image = cellprofiler.cpmath.outline.outline(labeled_image)
        outline_size_excluded_image = cellprofiler.cpmath.outline.outline(size_excluded_labeled_image)
        outline_border_excluded_image = cellprofiler.cpmath.outline.outline(border_excluded_labeled_image)
        
        if workspace.frame != None:
            statistics = []
            statistics.append(["Threshold","%0.3f"%(global_threshold)])
            statistics.append(["# of identified objects",
                               "%d"%(object_count)])
            if object_count > 0:
                areas = scipy.ndimage.sum(np.ones(labeled_image.shape), labeled_image, np.arange(1, object_count + 1))
                areas.sort()
                low_diameter  = (math.sqrt(float(areas[object_count / 10]) / np.pi) * 2)
                median_diameter = (math.sqrt(float(areas[object_count / 2]) / np.pi) * 2)
                high_diameter = (math.sqrt(float(areas[object_count * 9 / 10]) / np.pi) * 2)
                statistics.append(["10th pctile diameter",
                                   "%.1f pixels" % (low_diameter)])
                statistics.append(["median diameter",
                                   "%.1f pixels" % (median_diameter)])
                statistics.append(["90th pctile diameter",
                                   "%.1f pixels" % (high_diameter)])
                object_area = np.sum(areas)
                total_area  = np.product(labeled_image.shape[:2])
                statistics.append(["Area covered by objects",
                                   "%.1f %%" % (100.0 * float(object_area) /
                                              float(total_area))])
                if self.unclump_method != UN_NONE:
                    if self.unclump_method == UN_LOG:
                        statistics.append(["LoG threshold",
                                   "%.1f"%(LoG_threshold)])
                        statistics.append(["LoG filter diameter",
                                   "%.1f"%(LoG_filter_diameter)])
                    statistics.append(["Smoothing filter size",
                                   "%.1f"%(self.calc_smoothing_filter_size())])
                    statistics.append(["Maxima suppression size",
                                   "%.1f"%(maxima_suppression_size)])
            workspace.display_data.image = image
            workspace.display_data.labeled_image = labeled_image
            workspace.display_data.outline_image = outline_image
            workspace.display_data.outline_size_excluded_image = outline_size_excluded_image
            workspace.display_data.outline_border_excluded_image = outline_border_excluded_image
            workspace.display_data.statistics = statistics

        # Add image measurements
        objname = self.object_name.value
        measurements = workspace.measurements
        cpmi.add_object_count_measurements(measurements,
                                           objname, object_count)
        self.add_threshold_measurements(measurements, img, mask,
                                        local_threshold, global_threshold,
                                        self.object_name.value)
        # Add label matrices to the object set
        objects = cellprofiler.objects.Objects()
        objects.segmented = labeled_image
        objects.unedited_segmented = unedited_labels
        objects.small_removed_segmented = small_removed_labels
        objects.parent_image = image
        
        workspace.object_set.add_objects(objects,self.object_name.value)
        cpmi.add_object_location_measurements(workspace.measurements, 
                                              self.object_name.value,
                                              labeled_image)
        if self.should_save_outlines.value:
            out_img = cpi.Image(outline_image.astype(bool),
                                parent_image = image)
            workspace.image_set.add(self.save_outlines.value, out_img)
    
    def limit_object_count(self, labeled_image, object_count):
        '''Limit the object count according to the rules
        
        labeled_image - image to be limited
        object_count - check to see if this exceeds the maximum
        
        returns a new labeled_image and object count
        '''
        if object_count > self.maximum_object_count.value:
            if self.limit_choice == LIMIT_ERASE:
                labeled_image = np.zeros(labeled_image.shape, int)
                object_count = 0
            elif self.limit_choice == LIMIT_TRUNCATE:
                #
                # Pick arbitrary objects, doing so in a repeatable,
                # but pseudorandom manner.
                #
                r = np.random.RandomState()
                r.seed(abs(np.sum(labeled_image)))
                #
                # Pick an arbitrary ordering of the label numbers
                #
                index = r.permutation(object_count) + 1
                #
                # Pick only maximum_object_count of them
                #
                index = index[:self.maximum_object_count.value]
                #
                # Make a vector that maps old object numbers to new
                #
                mapping = np.zeros(object_count+1, int)
                mapping[index] = np.arange(1,len(index)+1)
                #
                # Relabel
                #
                labeled_image = mapping[labeled_image]
                object_count = len(index)
        return labeled_image, object_count
        
    def smooth_image(self, image, mask,sigma):
        """Apply the smoothing filter to the image"""
        
        filter_size = self.calc_smoothing_filter_size()
        if filter_size == 0:
            return image
        #
        # We not only want to smooth using a Gaussian, but we want to limit
        # the spread of the smoothing to 2 SD, partly to make things happen
        # locally, partly to make things run faster, partly to try to match
        # the Matlab behavior.
        #
        filter_size = max(int(float(filter_size) / 2.0),1)
        f = (1/np.sqrt(2.0 * np.pi ) / sigma * 
             np.exp(-0.5 * np.arange(-filter_size, filter_size+1)**2 / 
                    sigma ** 2))
        def fgaussian(image):
            output = scipy.ndimage.convolve1d(image, f,
                                              axis = 0,
                                              mode='constant')
            return scipy.ndimage.convolve1d(output, f,
                                            axis = 1,
                                            mode='constant')
        #
        # Use the trick where you similarly convolve an array of ones to find 
        # out the edge effects, then divide to correct the edge effects
        #
        edge_array = fgaussian(mask.astype(float))
        masked_image = image.copy()
        masked_image[~mask] = 0
        smoothed_image = fgaussian(masked_image)
        masked_image[mask] = smoothed_image[mask] / edge_array[mask]
        return masked_image
    
    def separate_neighboring_objects(self, image, mask, 
                                     labeled_image,object_count,threshold):
        """Separate objects based on local maxima or distance transform
        
        image         - the original grayscale image
        labeled_image - image labeled by scipy.ndimage.label
        object_count  - # of objects in image
        
        returns revised labeled_image, object count, maxima_suppression_size, LoG threshold and filter diameter
        """
        if self.unclump_method == UN_NONE or self.watershed_method == WA_NONE:
            return labeled_image, object_count, 7, 0.5, 5
        
        reported_LoG_filter_diameter = 5
        reported_LoG_threshold = 0.5
        sigma = self.calc_smoothing_filter_size() / 2.35
        blurred_image = self.smooth_image(image, mask, sigma)
        if self.low_res_maxima.value and self.size_range.min > 10:
            image_resize_factor = 10.0 / float(self.size_range.min)
            if self.automatic_suppression.value:
                maxima_suppression_size = 7
            else:
                maxima_suppression_size = (self.maxima_suppression_size.value *
                                           image_resize_factor+.5)
            reported_maxima_suppression_size = \
                    maxima_suppression_size / image_resize_factor
        else:
            image_resize_factor = 1.0
            if self.automatic_suppression.value:
                maxima_suppression_size = self.size_range.min/1.5
            else:
                maxima_suppression_size = self.maxima_suppression_size.value
            reported_maxima_suppression_size = maxima_suppression_size
        if maxima_suppression_size > 0:
            maxima_mask = strel_disk(maxima_suppression_size-.5)
        else:
            # User wants to use thresholding and shrinking
            maxima_mask = None
        distance_transformed_image = None
        if self.unclump_method == UN_LOG:
            if self.wants_automatic_log_diameter.value:
                diameter = (min(self.size_range.max, self.size_range.min**2) + 
                            self.size_range.min * 5)/6
            else:
                diameter = self.log_diameter.value
            reported_LoG_filter_diameter = diameter
            sigma = float(diameter) / 2.35
            #
            # Shrink the image to save processing time
            #
            if image_resize_factor < 1.0:
                shrunken = True
                shrunken_shape = (np.array(image.shape) * image_resize_factor+1).astype(int)
                i_j = np.mgrid[0:shrunken_shape[0],0:shrunken_shape[1]].astype(float) / image_resize_factor
                simage = scipy.ndimage.map_coordinates(image, i_j)
                smask = scipy.ndimage.map_coordinates(mask.astype(float), i_j) > .99
                diameter = diameter * image_resize_factor + 1
                sigma = sigma * image_resize_factor
            else:
                shrunken = False
                simage = image
                smask = mask
            normalized_image = 1 - stretch(simage, smask)
            
            window = max(3, int(diameter * 3 / 2))
            log_image = laplacian_of_gaussian(normalized_image, smask, 
                                              window, sigma)
            if shrunken:
                i_j = (np.mgrid[0:image.shape[0],
                                0:image.shape[1]].astype(float) * 
                       image_resize_factor)
                log_image = scipy.ndimage.map_coordinates(log_image, i_j)
            log_image = stretch(log_image, mask)
            if self.wants_automatic_log_threshold.value:
                log_threshold = otsu(log_image[mask], 0, 1, 256)
            else:
                log_threshold = self.manual_log_threshold.value
            reported_LoG_threshold = log_threshold
            log_image[log_image < log_threshold] = log_threshold
            log_image -= log_threshold
            maxima_image = self.get_maxima(log_image, labeled_image,
                                           maxima_mask, image_resize_factor)
        elif self.unclump_method == UN_INTENSITY:
            # Remove dim maxima
            maxima_image = blurred_image.copy()
            maxima_image[maxima_image < threshold] = 0
            maxima_image = self.get_maxima(blurred_image, 
                                           labeled_image,
                                           maxima_mask,
                                           image_resize_factor)
        elif self.unclump_method == UN_SHAPE:
            distance_transformed_image =\
                scipy.ndimage.distance_transform_edt(labeled_image>0)
            # randomize the distance slightly to get unique maxima
            np.random.seed(0)
            distance_transformed_image +=\
                np.random.uniform(0,.001,distance_transformed_image.shape)
            maxima_image = self.get_maxima(distance_transformed_image,
                                           labeled_image,
                                           maxima_mask,
                                           image_resize_factor)
        else:
            raise ValueError("Unsupported local maxima method: "%s(self.unclump_method.value))
        
        # Create the image for watershed
        if self.watershed_method == WA_INTENSITY:
            # use the reverse of the image to get valleys at peaks
            watershed_image = 1-image
        elif self.watershed_method == WA_SHAPE:
            if distance_transformed_image == None:
                distance_transformed_image =\
                    scipy.ndimage.distance_transform_edt(labeled_image>0)
            watershed_image = -distance_transformed_image
            watershed_image = watershed_image - np.min(watershed_image)
        elif self.watershed_method == WA_PROPAGATE:
            # No image used
            pass
        else:
            raise NotImplementedError("Watershed method %s is not implemented"%(self.watershed_method.value))
        #
        # Create a marker array where the unlabeled image has a label of
        # -(nobjects+1)
        # and every local maximum has a unique label which will become
        # the object's label. The labels are negative because that
        # makes the watershed algorithm use FIFO for the pixels which
        # yields fair boundaries when markers compete for pixels.
        #
        labeled_maxima,object_count = \
            scipy.ndimage.label(maxima_image, np.ones((3,3), bool))
        if self.watershed_method == WA_PROPAGATE:
            watershed_boundaries, distance =\
                propagate(np.zeros(labeled_maxima.shape),
                          labeled_maxima,
                          labeled_image != 0, 1.0)
        else:
            markers_dtype = (np.int16 
                             if object_count < np.iinfo(np.int16).max
                             else np.int32)
            markers = np.zeros(watershed_image.shape, markers_dtype)
            markers[labeled_maxima>0]=-labeled_maxima[labeled_maxima>0]
            #
            # Some labels have only one maker in them, some have multiple and
            # will be split up.
            # 
            watershed_boundaries = watershed(watershed_image,
                                             markers,
                                             np.ones((3,3),bool),
                                             mask=labeled_image!=0)
            watershed_boundaries = -watershed_boundaries
        
        return watershed_boundaries, object_count, reported_maxima_suppression_size, reported_LoG_threshold, reported_LoG_filter_diameter

    def get_maxima(self, image, labeled_image, maxima_mask, image_resize_factor):
        if image_resize_factor < 1.0:
            shape = np.array(image.shape) * image_resize_factor
            i_j = (np.mgrid[0:shape[0],0:shape[1]].astype(float) / 
                   image_resize_factor)
            resized_image = scipy.ndimage.map_coordinates(image, i_j)
            resized_labels = scipy.ndimage.map_coordinates(
                labeled_image, i_j, order=0).astype(labeled_image.dtype)
                                                           
        else:
            resized_image = image
            resized_labels = labeled_image
        #
        # find local maxima
        #
        if maxima_mask is not None:
            binary_maxima_image = is_local_maximum(resized_image,
                                                   resized_labels,
                                                   maxima_mask)
            binary_maxima_image[resized_image <= 0] = 0
        else:
            binary_maxima_image = (resized_image > 0) & (labeled_image > 0)
        if image_resize_factor < 1.0:
            inverse_resize_factor = (float(image.shape[0]) / 
                                     float(binary_maxima_image.shape[0]))
            i_j = (np.mgrid[0:image.shape[0],
                               0:image.shape[1]].astype(float) / 
                   inverse_resize_factor)
            binary_maxima_image = scipy.ndimage.map_coordinates(
                binary_maxima_image.astype(float), i_j) > .5
            assert(binary_maxima_image.shape[0] == image.shape[0])
            assert(binary_maxima_image.shape[1] == image.shape[1])
        
        # Erode blobs of touching maxima to a single point
        
        shrunk_image = binary_shrink(binary_maxima_image)
        return shrunk_image
    
    def filter_on_size(self,labeled_image,object_count):
        """ Filter the labeled image based on the size range
        
        labeled_image - pixel image labels
        object_count - # of objects in the labeled image
        returns the labeled image, and the labeled image with the 
        small objects removed
        """
        if self.exclude_size.value and object_count > 0:
            areas = scipy.ndimage.measurements.sum(np.ones(labeled_image.shape),
                                                   labeled_image,
                                                   np.array(range(0,object_count+1),dtype=np.int32))
            areas = np.array(areas,dtype=int)
            min_allowed_area = np.pi * (self.size_range.min * self.size_range.min)/4
            max_allowed_area = np.pi * (self.size_range.max * self.size_range.max)/4
            # area_image has the area of the object at every pixel within the object
            area_image = areas[labeled_image]
            labeled_image[area_image < min_allowed_area] = 0
            small_removed_labels = labeled_image.copy()
            labeled_image[area_image > max_allowed_area] = 0
        else:
            small_removed_labels = labeled_image.copy()
        return (labeled_image, small_removed_labels)

    def filter_on_border(self,image,labeled_image):
        """Filter out objects touching the border
        
        In addition, if the image has a mask, filter out objects
        touching the border of the mask.
        """
        if self.exclude_border_objects.value:
            border_labels = list(labeled_image[0,:])
            border_labels.extend(labeled_image[:,0])
            border_labels.extend(labeled_image[labeled_image.shape[0]-1,:])
            border_labels.extend(labeled_image[:,labeled_image.shape[1]-1])
            border_labels = np.array(border_labels)
            #
            # the following histogram has a value > 0 for any object
            # with a border pixel
            #
            histogram = scipy.sparse.coo_matrix((np.ones(border_labels.shape),
                                                 (border_labels,
                                                  np.zeros(border_labels.shape))),
                                                 shape=(np.max(labeled_image)+1,1)).todense()
            histogram = np.array(histogram).flatten()
            if any(histogram[1:] > 0):
                histogram_image = histogram[labeled_image]
                labeled_image[histogram_image > 0] = 0
            elif image.has_mask:
                # The assumption here is that, if nothing touches the border,
                # the mask is a large, elliptical mask that tells you where the
                # well is. That's the way the old Matlab code works and it's duplicated here
                #
                # The operation below gets the mask pixels that are on the border of the mask
                # The erosion turns all pixels touching an edge to zero. The not of this
                # is the border + formerly masked-out pixels.
                mask_border = np.logical_not(scipy.ndimage.binary_erosion(image.mask))
                mask_border = np.logical_and(mask_border,image.mask)
                border_labels = labeled_image[mask_border]
                border_labels = border_labels.flatten()
                histogram = scipy.sparse.coo_matrix((np.ones(border_labels.shape),
                                                     (border_labels,
                                                      np.zeros(border_labels.shape))),
                                                      shape=(np.max(labeled_image)+1,1)).todense()
                histogram = np.array(histogram).flatten()
                if any(histogram[1:] > 0):
                    histogram_image = histogram[labeled_image]
                    labeled_image[histogram_image > 0] = 0
        return labeled_image
    
    def display(self, workspace):
        import cellprofiler.gui.cpfigure as cpf
        if workspace.frame != None:
            """Display the image and labeling"""
            window_name = "CellProfiler:%s:%d"%(self.module_name, self.module_num)
            my_frame=cpf.create_or_find(workspace.frame, 
                                        title="IdentifyPrimaryObjects, image cycle #%d"%(
                workspace.measurements.image_set_number), 
                                        name=window_name, subplots=(2,2))
            
            orig_axes     = my_frame.subplot(0,0)
            label_axes    = my_frame.subplot(1,0, sharex=orig_axes, sharey=orig_axes)
            outlined_axes = my_frame.subplot(0,1, sharex=orig_axes, sharey=orig_axes)
            table_axes    = my_frame.subplot(1,1, sharex=orig_axes, sharey=orig_axes)
    
            title = "Original image, cycle #%d"%(workspace.image_set.number + 1,)
            my_frame.subplot_imshow_grayscale(0, 0,
                                              workspace.display_data.image.pixel_data,
                                              title)
            my_frame.subplot_imshow_labels(1, 0, workspace.display_data.labeled_image, 
                                           self.object_name.value)
    
            if workspace.display_data.image.pixel_data.ndim == 2:
                # Outline the size-excluded pixels in red
                outline_img = np.ndarray(shape=(workspace.display_data.image.pixel_data.shape[0],
                                                   workspace.display_data.image.pixel_data.shape[1],3))
                outline_img[:,:,0] = workspace.display_data.image.pixel_data 
                outline_img[:,:,1] = workspace.display_data.image.pixel_data
                outline_img[:,:,2] = workspace.display_data.image.pixel_data
            else:
                outline_img = workspace.display_data.image.pixel_data.copy()
            
            # Outline the accepted objects pixels
            draw_outline(outline_img, workspace.display_data.outline_image,
                         cpp.get_primary_outline_color())
            
            # Outline the size-excluded pixels
            draw_outline(outline_img,
                         workspace.display_data.outline_size_excluded_image,
                         cpp.get_secondary_outline_color())
            
            # Outline the border-excluded pixels in yellow
            draw_outline(outline_img,
                         workspace.display_data.outline_border_excluded_image,
                         cpp.get_tertiary_outline_color())
            
            title = "%s outlines"%(self.object_name.value) 
            my_frame.subplot_imshow(0, 1, outline_img, title, normalize=False)
            
            table_axes.clear()
            table = table_axes.table(cellText=workspace.display_data.statistics,
                                     colWidths=[.7,.3],
                                     loc='center',
                                     cellLoc='left')
            table_axes.set_frame_on(False)
            table_axes.set_axis_off()
            table.auto_set_font_size(False)
            table.set_fontsize(cpp.get_table_font_size())
            my_frame.Refresh()
    
    def calc_smoothing_filter_size(self):
        """Return the size of the smoothing filter, calculating it if in automatic mode"""
        if self.automatic_smoothing.value:
            return 2.35*self.size_range.min/3.5;
        else:
            return self.smoothing_filter_size.value
    
    def get_measurement_columns(self, pipeline):
        '''Column definitions for measurements made by IdentifyPrimAutomatic'''
        columns = cpmi.get_object_measurement_columns(self.object_name.value)
        columns += [(cpmeas.IMAGE, 
                     format%self.object_name.value,
                     cpmeas.COLTYPE_FLOAT)
                    for format in (FF_FINAL_THRESHOLD, FF_ORIG_THRESHOLD,
                                   FF_WEIGHTED_VARIANCE, FF_SUM_OF_ENTROPIES)]
        return columns
             
    def get_categories(self,pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        result = self.get_threshold_categories(pipeline, object_name)
        result += self.get_object_categories(pipeline, object_name,
                                             {self.object_name.value: [] })
        return result
      
    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        result = self.get_threshold_measurements(pipeline, object_name,
                                                 category)
        result += self.get_object_measurements(pipeline, object_name, category,
                                               {self.object_name.value: [] })
        return result
    
    def get_measurement_objects(self, pipeline, object_name, category, 
                                measurement):
        """Return the objects associated with image measurements
        
        """
        return self.get_threshold_measurement_objects(pipeline, object_name,
                                                      category, measurement,
                                                      self.object_name.value)
IdentifyPrimAutomatic = IdentifyPrimaryObjects
