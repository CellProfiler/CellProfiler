"""IdentifyPrimAutomatic - identify objects by thresholding and contouring

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import math
import scipy.ndimage
import scipy.sparse
import matplotlib.backends.backend_wxagg
import matplotlib.figure
import matplotlib.pyplot
import matplotlib.cm
import numpy as np
import re
import scipy.stats
import wx

import identify as cpmi
import cellprofiler.cpmodule
import cellprofiler.settings as cps
import cellprofiler.gui.cpfigure as cpf
import cellprofiler.preferences as cpp
from cellprofiler.cpmath.otsu import otsu
from cellprofiler.cpmath.cpmorphology import fill_labeled_holes, strel_disk
from cellprofiler.cpmath.cpmorphology import binary_shrink, relabel
from cellprofiler.cpmath.cpmorphology import regional_maximum
from cellprofiler.cpmath.filter import stretch, laplacian_of_gaussian
from cellprofiler.cpmath.watershed import fast_watershed as watershed
from cellprofiler.cpmath.smooth import smooth_with_noise
import cellprofiler.cpmath.outline
import cellprofiler.objects
from cellprofiler.settings import AUTOMATIC
import cellprofiler.cpmath.threshold as cpthresh

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
WA_DISTANCE                     = "Distance"
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


class IdentifyPrimAutomatic(cpmi.Identify):
    """This module identifies primary objects (e.g. nuclei) in grayscale images
that show bright objects on a dark background. The module has many
options which vary in terms of speed and sophistication. The objects that
are found are displayed with arbitrary colors - the colors do not mean 
anything but simply help you to tell various objects apart. You can 
change the colormap in File > Set Preferences.
%
Requirements for the images to be fed into this module:
* If the objects are dark on a light background, they must first be
inverted using the Invert Intensity module.
* If you are working with color images, they must first be converted to
grayscale using the Color To Gray module.
%
Overview of the strategy ('Settings' below has more details):
  Properly identifying primary objects (nuclei) that are well-dispersed,
non-confluent, and bright relative to the background is straightforward
by applying a simple threshold to the image. This is fast but usually
fails when nuclei are touching. In CellProfiler, several automatic
thresholding methods are available, including global and adaptive, using
Otsu's (Otsu, 1979) and our own version of a Mixture of Gaussians
algorithm (O. Friman, unpublished). For most biological images, at least
some nuclei are touching, so CellProfiler contains a novel modular
three-step strategy based on previously published algorithms (Malpica et
al., 1997; Meyer and Beucher, 1990; Ortiz de Solorzano et al., 1999;
Wahlby, 2003; Wahlby et al., 2004). Choosing different options for each
of these three steps allows CellProfiler to flexibly analyze a variety of
different cell types. Here are the three steps:
  In step 1, CellProfiler determines whether an object is an individual
nucleus or two or more clumped nuclei. This determination can be
accomplished in two ways, depending on the cell type: When nuclei are
bright in the middle and dimmer towards the edges (the most common case),
identifying local maxima in the smoothed intensity image works well
(Intensity option). When nuclei are quite round, identifying local maxima
in the distance-transformed thresholded image (where each pixel gets a
value equal to the distance to the nearest pixel below a certain
threshold) works well (Shape option). For quick processing where cells
are well-dispersed, you can choose to make no attempt to separate clumped
objects.
  In step 2, the edges of nuclei are identified. For nuclei within the
image that do not appear to touch, the edges are easily determined using
thresholding. For nuclei that do appear to touch, there are two options
for finding the edges of clumped nuclei. Where the dividing lines tend to
be dimmer than the remainder of the nucleus (the most common case), the
Intensity option works best (already identified nuclear markers are
starting points for a watershed algorithm (Vincent and Soille, 1991)
applied to the original image). When no dim dividing lines exist, the
Distance option places the dividing line at a point between the two
nuclei determined by their shape (the distance-transformed thresholded
image is used for the watershed algorithm). In other words, the dividing
line is usually placed where indentations occur along the edge of the
clumped nuclei.
  In step 3, some identified nuclei are discarded or merged together if
the user chooses. Incomplete nuclei touching the border of the image can
be discarded. Objects smaller than a user-specified size range, which are
likely to be fragments of real nuclei, can be discarded. Alternately, any
of these small objects that touch a valid nucleus can be merged together
based on a set of heuristic rules; for example similarity in intensity
and statistics of the two objects. A separate module,
FilterByObjectMeasurement, further refines the identified nuclei, if
desired, by excluding objects that are a particular size, shape,
intensity, or texture. This refining step could eventually be extended to
include other quality-control filters, e.g. a second watershed on the
distance transformed image to break up remaining clusters (Wahlby et al.,
2004).
%
For more details, see the Settings section below and also the notation
within the code itself (Developer's version).
%
Malpica, N., de Solorzano, C. O., Vaquero, J. J., Santos, A., Vallcorba,
I., Garcia-Sagredo, J. M., and del Pozo, F. (1997). Applying watershed
algorithms to the segmentation of clustered nuclei. Cytometry 28,
289-297.
Meyer, F., and Beucher, S. (1990). Morphological segmentation. J Visual
Communication and Image Representation 1, 21-46.
Ortiz de Solorzano, C., Rodriguez, E. G., Jones, A., Pinkel, D., Gray, J.
W., Sudar, D., and Lockett, S. J. (1999). Segmentation of confocal
microscope images of cell nuclei in thick tissue sections. Journal of
Microscopy-Oxford 193, 212-226.
Wahlby, C. (2003) Algorithms for applied digital image cytometry, Ph.D.,
Uppsala University, Uppsala.
Wahlby, C., Sintorn, I. M., Erlandsson, F., Borgefors, G., and Bengtsson,
E. (2004). Combining intensity, edge and shape information for 2D and 3D
segmentation of cell nuclei in tissue sections. J Microsc 215, 67-76.
%
Settings:
%
Typical diameter of objects, in pixel units (Min,Max):
This is a very important parameter which tells the module what you are
looking for. Most options within this module use this estimate of the
size range of the objects in order to distinguish them from noise in the
image. For example, for some of the identification methods, the smoothing
applied to the image is based on the minimum size of the objects. A comma
should be placed between the minimum and the maximum diameters. The units
here are pixels so that it is easy to zoom in on objects and determine
typical diameters. To measure distances easily, use the CellProfiler
Image Tool, 'ShowOrHidePixelData', in any open window. Once this tool is
activated, you can draw a line across objects in your image and the
length of the line will be shown in pixel units. Note that for non-round
objects, the diameter here is actually the 'equivalent diameter', meaning
the diameter of a circle with the same area as the object.
%
Discard objects outside the diameter range:
You can choose to discard objects outside the specified range of
diameters. This allows you to exclude small objects (e.g. dust, noise,
and debris) or large objects (e.g. clumps) if desired. See also the
FilterByObjectMeasurement module to further discard objects based on some
other measurement. During processing, the window for this module will
show that objects outlined in green were acceptable, objects outlined in
red were discarded based on their size, and objects outlined in yellow
were discarded because they touch the border.
%
Try to merge 'too small' objects with nearby larger objects:
Use caution when choosing 'Yes' for this option! This is an experimental
functionality that takes objects that were discarded because they were
smaller than the specified Minimum diameter and tries to merge them with
other surrounding objects. This is helpful in cases when an object was
incorrectly split into two objects, one of which is actually just a tiny
piece of the larger object. However, this could be dangerous if you have
selected poor settings which produce many tiny objects - the module
will take a very long time and you will not realize that it is because
the tiny objects are being merged. It is therefore a good idea to run the
module first without merging objects to make sure the settings are
reasonably effective.
%
Discard objects touching the border of the image:
You can choose to discard objects that touch the border of the image.
This is useful in cases when you do not want to make measurements of
objects that are not fully within the field of view (because, for
example, the area would not be accurate).
%
Select automatic thresholding method:
   The threshold affects the stringency of the lines between the objects
and the background. You can have the threshold automatically calculated
using several methods, or you can enter an absolute number between 0 and
1 for the threshold (to see the pixel intensities for your images in the
appropriate range of 0 to 1, use the CellProfiler Image Tool,
'ShowOrHidePixelData', in a window showing your image). There are
advantages either way. An absolute number treats every image identically,
but is not robust to slight changes in lighting/staining conditions
between images. An automatically calculated threshold adapts to changes
in lighting/staining conditions between images and is usually more
robust/accurate, but it can occasionally produce a poor threshold for
unusual/artifactual images. It also takes a small amount of time to
calculate.
   The threshold which is used for each image is recorded as a
measurement in the output file, so if you find unusual measurements from
one of your images, you might check whether the automatically calculated
threshold was unusually high or low compared to the other images.
   There are five methods for finding thresholds automatically, Otsu's
method, the Mixture of Gaussian (MoG) method, the Background method, the
Robust Background method and the Ridler-Calvard method. 
** The Otsu method
uses our version of the Matlab function graythresh (the code is in the
CellProfiler subfunction CPthreshold). Our modifications include taking
into account the max and min values in the image and log-transforming the
image prior to calculating the threshold. Otsu's method is probably best
if you don't know anything about the image, or if the percent of the
image covered by objects varies substantially from image to image. If you
know the object coverage percentage and it does not vary much from image
to image, the MoG can be better, especially if the coverage percentage is
not near 50%. Note, however, that the MoG function is experimental and
has not been thoroughly validated. 
** The Background method 
is simple and appropriate for images in which most of the image is 
background. It finds the mode of the histogram of the image, which is 
assumed to be the background of the image, and chooses a threshold at 
twice that value (which you can adjust with a Threshold Correction Factor,
see below).  Note that the mode is protected from a high number of 
saturated pixels by only counting pixels < 0.95. This can be very helpful,
for example, if your images vary in overall brightness but the objects of 
interest are always twice (or actually, any constant) as bright as the 
background of the image. 
** The Robust background
method trims the brightest and dimmest 5of pixel intensities off first
in the hopes that the remaining pixels represent a gaussian of intensity
values that are mostly background pixels. It then calculates the mean and
standard deviation of the remaining pixels and calculates the threshold
as the mean + 2 times the standard deviation. 
** The Ridler-Calvard method
is simple and its results are often very similar to Otsu's - according to
Sezgin and Sankur's paper (Journal of Electronic Imaging 2004), Otsu's 
overall quality on testing 40 nondestructive testing images is slightly 
better than Ridler's (Average error - Otsu: 0.318, Ridler: 0.401). 
It chooses an initial threshold, and then iteratively calculates the next 
one by taking the mean of the average intensities of the background and 
foreground pixels determined by the first threshold, repeating this until 
the threshold converges.
** The Kapur method
computes the threshold of an image by
log-transforming its values, then searching for the threshold that
maximizes the sum of entropies of the foreground and background
pixel values, when treated as separate distributions.
   You can also choose between Global, Adaptive, and Per object
thresholding:
Global: one threshold is used for the entire image (fast).
Adaptive: the threshold varies across the image - a bit slower but
provides more accurate edge determination which may help to separate
clumps, especially if you are not using a clump-separation method (see
below).
Per object: if you are using this module to find child objects located
*within* parent objects, the per object method will calculate a distinct
threshold for each parent object. This is especially helpful, for
example, when the background brightness varies substantially among the
parent objects. Important: the per object method requires that you run an
IdentifyPrim module to identify the parent objects upstream in the
pipeline. After the parent objects are identified in the pipeline, you
must then also run a Crop module as follows: the image to be cropped is the one
that you will want to use within this module to identify the children
objects (e.g., ChildrenStainedImage), and the shape in which to crop
is the name of the parent objects (e.g., Nuclei). Then, set this
IdentifyPrimAutomatic module to identify objects within the
CroppedChildrenStainedImage.

Threshold correction factor:
When the threshold is calculated automatically, it may consistently be
too stringent or too lenient. You may need to enter an adjustment factor
which you empirically determine is suitable for your images. The number 1
means no adjustment, 0 to 1 makes the threshold more lenient and greater
than 1 (e.g. 1.3) makes the threshold more stringent. For example, the
Otsu automatic thresholding inherently assumes that 50of the image is
covered by objects. If a larger percentage of the image is covered, the
Otsu method will give a slightly biased threshold that may have to be
corrected using a threshold correction factor.

Lower and upper bounds on threshold:
Can be used as a safety precaution when the threshold is calculated
automatically. For example, if there are no objects in the field of view,
the automatic threshold will be unreasonably low. In such cases, the
lower bound you enter here will override the automatic threshold.

Approximate percentage of image covered by objects:
An estimate of how much of the image is covered with objects. This
information is currently only used in the MoG (Mixture of Gaussian)
thresholding but may be used for other thresholding methods in the future
(see below).

Method to distinguish clumped objects:
Note: to choose between these methods, you can try test mode (see the
last setting for this module).
* Intensity - For objects that tend to have only one peak of brightness
per object (e.g. objects that are brighter towards their interiors), this
option counts each intensity peak as a separate object. The objects can
be any shape, so they need not be round and uniform in size as would be
required for a distance-based module. The module is more successful when
the objects have a smooth texture. By default, the image is automatically
blurred to attempt to achieve appropriate smoothness (see blur option),
but overriding the default value can improve the outcome on
lumpy-textured objects. Technical description: Object centers are defined
as local intensity maxima.
* Shape - For cases when there are definite indentations separating
objects. This works best for objects that are round. The intensity
patterns in the original image are irrelevant - the image is converted to
black and white (binary) and the shape is what determines whether clumped
objects will be distinguished. Therefore, the cells need not be brighter
towards the interior as is required for the Intensity option. The
de-clumping results of this method are affected by the thresholding
method you choose. Technical description: The binary thresholded image is
distance-transformed and object centers are defined as peaks in this
image. 

* None (fastest option) - If objects are far apart and are very well
separated, it may be unnecessary to attempt to separate clumped objects.
Using the 'None' option, a simple threshold will be used to identify
objects. This will override any declumping method chosen in the next
question.

Method to draw dividing lines between clumped objects:
* Intensity - works best where the dividing lines between clumped
objects are dim. Technical description: watershed on the intensity image.
* Distance - Dividing lines between clumped objects are based on the
shape of the clump. For example, when a clump contains two objects, the
dividing line will be placed where indentations occur between the two
nuclei. The intensity patterns in the original image are irrelevant - the
cells need not be dimmer along the lines between clumped objects.
Technical description: watershed on the distance-transformed thresholded
image.
* None (fastest option) - If objects are far apart and are very well
separated, it may be unnecessary to attempt to separate clumped objects.
Using the 'None' option, the thresholded image will be used to identify
objects. This will override any declumping method chosen in the above
question.

Size of smoothing filter, in pixel units:
   (Only used when distinguishing between clumped objects) This setting,
along with the suppress local maxima setting, affects whether objects
close to each other are considered a single object or multiple objects.
It does not affect the dividing lines between an object and the
background. If you see too many objects merged that ought to be separate,
the value should be lower. If you see too many objects split up that
ought to be merged, the value should be higher.
   The image is smoothed based on the specified minimum object diameter
that you have entered, but you may want to override the automatically
calculated value here. Reducing the texture of objects by increasing the
smoothing increases the chance that each real, distinct object has only
one peak of intensity but also increases the chance that two distinct
objects will be recognized as only one object. Note that increasing the
size of the smoothing filter increases the processing time exponentially.
%
Suppress local maxima within this distance (a positive integer, in pixel
units):
   (Only used when distinguishing between clumped objects) This setting,
along with the size of the smoothing filter, affects whether objects
close to each other are considered a single object or multiple objects.
It does not affect the dividing lines between an object and the
background. This setting looks for the maximum intensity in the size 
specified by the user.  The local intensity histogram is smoothed to 
remove the peaks within that distance. So,if you see too many objects 
merged that ought to be separate, the value should be lower. If you see 
too many objects split up that ought to be merged, the value should be higher.
   Object markers are suppressed based on the specified minimum object
diameter that you have entered, but you may want to override the
automatically calculated value here. The maxima suppression distance
should be set to be roughly equivalent to the minimum radius of a real
object of interest. Basically, any distinct 'objects' which are found but
are within two times this distance from each other will be assumed to be
actually two lumpy parts of the same object, and they will be merged.

Speed up by using lower-resolution image to find local maxima?
(Only used when distinguishing between clumped objects) If you have
entered a minimum object diameter of 10 or less, setting this option to
Yes will have no effect.

Technical notes: The initial step of identifying local maxima is
performed on the user-controlled heavily smoothed image, the
foreground/background is done on a hard-coded slightly smoothed image,
and the dividing lines between clumped objects (watershed) is done on the
non-smoothed image.

Laplacian of Gaussian method:
This is a specialized method to find objects and will override the above
settings in this module. The code was kindly donated by Zach Perlman and 
was used in this published work:
Multidimensional drug profiling by automated microscopy.
Science. 2004 Nov 12;306(5699):1194-8.  PMID: 15539606
Regrettably, we have no further description of its settings.

Special note on saving images: Using the settings in this module, object
outlines can be passed along to the module OverlayOutlines and then saved
with the SaveImages module. Objects themselves can be passed along to the
object processing module ConvertToImage and then saved with the
SaveImages module. This module produces several additional types of
objects with names that are automatically passed along with the following
naming structure: (1) The unedited segmented image, which includes
objects on the edge of the image and objects that are outside the size
range, can be saved using the name: UneditedSegmented + whatever you
called the objects (e.g. UneditedSegmentedNuclei). (2) The segmented
image which excludes objects smaller than your selected size range can be
saved using the name: SmallRemovedSegmented + whatever you called the
objects (e.g. SmallRemovedSegmented Nuclei).
"""
            
    variable_revision_number = 3

    category =  "Object Processing"
    
    def create_settings(self):
        self.set_module_name("IdentifyPrimAutomatic")
        self.image_name = cps.ImageNameSubscriber('What did you call the images you want to process?')
        self.object_name = cps.ObjectNameProvider('What do you want to call the objects identified by this module?', 'Nuclei')
        self.size_range = cps.IntegerRange('Typical diameter of objects, in pixel units (Min,Max):', 
                                           (10,40),minval=1)
        self.exclude_size = cps.Binary('Discard objects outside the diameter range?', True)
        self.merge_objects = cps.Binary('Try to merge too small objects with nearby larger objects?', False)
        self.exclude_border_objects = cps.Binary('Discard objects touching the border of the image?', True)
        self.threshold_method = cps.Choice('''Select an automatic thresholding method or choose "Manual" to enter a threshold manually.  To choose a binary image, select "Binary image".  Choosing 'All' will use the Otsu Global method to calculate a single threshold for the entire image group. The other methods calculate a threshold for each image individually. "Set interactively" will allow you to manually adjust the threshold during the first cycle to determine what will work well.''',
                                           cpthresh.TM_METHODS)
        self.threshold_correction_factor = cps.Float('Threshold correction factor', 1)
        self.threshold_range = cps.FloatRange('Lower and upper bounds on threshold, in the range [0,1]', (0,1),minval=0,maxval=1)
        self.object_fraction = cps.CustomChoice('For MoG thresholding, what is the approximate fraction of image covered by objects?',
                                                ['0.01','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','0.99'])
        self.unclump_method = cps.Choice('Method to distinguish clumped objects (see help for details):', 
                                          [UN_INTENSITY, UN_SHAPE, UN_LOG, UN_NONE])
        self.watershed_method = cps.Choice('Method to draw dividing lines between clumped objects (see help for details):', 
                                           [WA_INTENSITY,WA_DISTANCE,WA_NONE])
        self.automatic_smoothing = cps.Binary('Automatically calculate size of smoothing filter when separating clumped objects',True)
        self.smoothing_filter_size = cps.Integer('Size of smoothing filter, in pixel units (if you are distinguishing between clumped objects). Enter 0 for low resolution images with small objects (~< 5 pixel diameter) to prevent any image smoothing.', 10)
        self.automatic_suppression = cps.Binary('Automatically calculate minimum size of local maxima for clumped objects',True)
        self.maxima_suppression_size = cps.Integer( 'Suppress local maxima within this distance, (a positive integer, in pixel units) (if you are distinguishing between clumped objects)', 7)
        self.low_res_maxima = cps.Binary('Speed up by using lower-resolution image to find local maxima?  (if you are distinguishing between clumped objects)', True)
        self.should_save_outlines = cps.Binary('Do you want to save outlines?',False)
        self.save_outlines = cps.OutlineNameProvider('What do you want to call the outlines of the identified objects?')
        self.fill_holes = cps.Binary('Do you want to fill holes in identified objects?', True)
        self.test_mode = cps.Binary('Do you want to run in test mode where each method for distinguishing clumped objects is compared?', False)
        self.masking_object = cps.ObjectNameSubscriber('What are the objects you want to use for per-object thresholding?')
        self.manual_threshold = cps.Float("What is the manual threshold?",value=0.0,minval=0.0,maxval=1.0)
        self.binary_image = cps.ImageNameSubscriber("What is the binary thresholding image?","None")
        self.wants_automatic_log_threshold = cps.Binary('Do you want to calculate the Laplacian of Gaussian threshold automatically?',True)
        self.manual_log_threshold = cps.Float('What is the Laplacian of Gaussian threshold?',.5, 0, 1)
        self.two_class_otsu = cps.Choice('Does your image have two classes of intensity value or three?',
                                         [cpmi.O_TWO_CLASS, cpmi.O_THREE_CLASS])
        self.use_weighted_variance = cps.Choice('Do you want to minimize the weighted variance or the entropy?',
                                                [cpmi.O_WEIGHTED_VARIANCE, cpmi.O_ENTROPY])
        self.assign_middle_to_foreground = cps.Choice("Assign pixels in the middle intensity class to the foreground or the background?",
                                                      [cpmi.O_FOREGROUND, cpmi.O_BACKGROUND])

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
                self.assign_middle_to_foreground ]
    
    def backwards_compatibilize(self, setting_values, variable_revision_number, 
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
             
        return setting_values, variable_revision_number, from_matlab
            
    def visible_settings(self):
        vv = [self.image_name,self.object_name,self.size_range, \
                self.exclude_size, self.merge_objects, \
                self.exclude_border_objects, self.threshold_method]
        if self.threshold_method == cpthresh.TM_MANUAL:
            vv += [self.manual_threshold]
        elif self.threshold_method == cpthresh.TM_BINARY_IMAGE:
            vv += [self.binary_image]
        if self.threshold_algorithm == cpthresh.TM_OTSU:
            vv += [self.two_class_otsu, self.use_weighted_variance]
            if self.two_class_otsu == cpmi.O_THREE_CLASS:
                vv.append(self.assign_middle_to_foreground)
        if self.threshold_algorithm == cpthresh.TM_MOG:
            vv += [self.object_fraction]
        if not self.threshold_method in (cpthresh.TM_MANUAL, cpthresh.TM_BINARY_IMAGE):
            vv += [ self.threshold_correction_factor, self.threshold_range]
        vv += [ self.unclump_method ]
        if self.unclump_method != UN_NONE:
            if self.unclump_method == UN_LOG:
                vv += [self.wants_automatic_log_threshold]
                if not self.wants_automatic_log_threshold.value:
                    vv += [self.manual_log_threshold]
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
        vv += [self.fill_holes]
        return vv
    
    def run(self,workspace):
        """Run the module
        
        pipeline     - instance of CellProfiler.Pipeline for this run
        workspace    - contains
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
        """
        self.workspace_saved_for_debugging_take_this_out_please=workspace 
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
                                                              masking_objects)
        blurred_image = self.smooth_image(img,mask,1)
        binary_image = np.logical_and((blurred_image >= local_threshold),mask)
        labeled_image,object_count = scipy.ndimage.label(binary_image,
                                                         np.ones((3,3),bool))
        #
        # Fill holes if appropriate
        #
        if self.fill_holes.value:
            labeled_image = fill_labeled_holes(labeled_image)
        labeled_image,object_count,maxima_suppression_size = \
            self.separate_neighboring_objects(img, mask, 
                                              labeled_image,
                                              object_count,global_threshold)
        # Filter out small and large objects
        labeled_image, unedited_labels, small_removed_labels = \
            self.filter_on_size(labeled_image,object_count)
        # Filter out objects touching the border or mask
        labeled_image = self.filter_on_border(image, labeled_image)
        # Relabel the image
        labeled_image,object_count = relabel(labeled_image)
        # Make an outline image
        outline_image = cellprofiler.cpmath.outline.outline(labeled_image)
        if workspace.frame:
            statistics = []
            statistics.append(["Threshold","%0.3f"%(global_threshold)])
            statistics.append(["# of identified objects",
                               "%d"%(object_count)])
            if object_count > 0:
                areas = scipy.ndimage.histogram(labeled_image,1,object_count+1,object_count)
                areas.sort()
                low_diameter  = (math.sqrt(float(areas[object_count/10]))*2/
                                 np.pi)
                high_diameter = (math.sqrt(float(areas[object_count*9/10]))*2/
                                 np.pi)
                statistics.append(["10th pctile diameter",
                                   "%.1f pixels"%(low_diameter)])
                statistics.append(["90th pctile diameter",
                                   "%.1f pixels"%(high_diameter)])
                object_area = np.sum(labeled_image > 0)
                total_area  = np.product(labeled_image.shape[:2])
                statistics.append(["Area covered by objects",
                                   "%.1f %%"%(100.0*float(object_area)/
                                              float(total_area))])
                statistics.append(["Smoothing filter size",
                                   "%.1f"%(self.calc_smoothing_filter_size())])
                statistics.append(["Maxima suppression size",
                                   "%.1f"%(maxima_suppression_size)])
            self.display(workspace.frame,
                         image,
                         labeled_image,
                         outline_image,
                         statistics,
                         workspace.image_set.number+1)
        # Add image measurements
        objname = self.object_name.value
        measurements = workspace.measurements
        cpmi.add_object_count_measurements(measurements,
                                           objname, object_count)
        if self.threshold_modifier == cpthresh.TM_GLOBAL:
            # The local threshold is a single number
            assert(not isinstance(local_threshold,np.ndarray))
            ave_threshold = local_threshold
        else:
            # The local threshold is an array
            ave_threshold = local_threshold.mean()
        measurements.add_measurement('Image',
                                     'Threshold_FinalThreshold_%s'%(objname),
                                     np.array([ave_threshold],
                                                 dtype=float))
        measurements.add_measurement('Image',
                                     'Threshold_OrigThreshold_%s'%(objname),
                                     np.array([global_threshold],
                                                  dtype=float))
        wv = cpthresh.weighted_variance(img, mask, local_threshold)
        measurements.add_measurement('Image',
                                     'Threshold_WeightedVariance_%s'%(objname),
                                     np.array([wv],dtype=float))
        entropies = cpthresh.sum_of_entropies(img, mask, local_threshold)
        measurements.add_measurement('Image',
                                     'Threshold_SumOfEntropies_%s'%(objname),
                                     np.array([entropies],dtype=float))
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
            workspace.add_outline(self.save_outlines.value, outline_image)
    
    def smooth_image(self, image, mask,sigma):
        """Apply the smoothing filter to the image"""
        
        if self.calc_smoothing_filter_size() == 0:
            return image
        #
        # Use the trick where you similarly convolve an array of ones to find 
        # out the edge effects, then divide to correct the edge effects
        #
        edge_array = scipy.ndimage.gaussian_filter(mask.astype(float),
                                                   sigma,mode='constant')
        masked_image = image.copy()
        masked_image[~mask] = 0
        return scipy.ndimage.gaussian_filter(masked_image,sigma,mode='constant') / edge_array
    
    def separate_neighboring_objects(self, image, mask, 
                                     labeled_image,object_count,threshold):
        """Separate objects based on local maxima or distance transform
        
        image         - the original grayscale image
        labeled_image - image labeled by scipy.ndimage.label
        object_count  - # of objects in image
        
        returns revised labeled_image, object count and maxima_suppression_size
        """
        if self.unclump_method == UN_NONE or self.watershed_method == WA_NONE:
            return labeled_image, object_count, 7
        
        blurred_image = self.smooth_image(image, mask, 
                                          self.calc_smoothing_filter_size())
        if self.low_res_maxima.value and self.size_range.min > 10:
            image_resize_factor = 10.0 / float(self.size_range.min)
            if self.automatic_suppression.value:
                maxima_suppression_size = 7
            else:
                maxima_suppression_size = int(self.maxima_suppression_size.value *
                                              image_resize_factor+.5)
        else:
            image_resize_factor = 1.0
            if self.automatic_suppression.value:
                maxima_suppression_size = int(math.floor(self.size_range.min/1.5+.5))
            else:
                maxima_suppression_size = self.maxima_suppression_size.value
        maxima_mask = strel_disk(maxima_suppression_size)
        distance_transformed_image = None
        if self.unclump_method == UN_LOG:
            diameter = (self.size_range.max+self.size_range.min)/2
            sigma = float(diameter) / 2.35
            #
            # Shrink the image to save processing time
            #
            figure = self.workspace_saved_for_debugging_take_this_out_please.create_or_find_figure(subplots=(2,3),window_name='debugging')
            figure.subplot_imshow_grayscale(0,0,image,"original")
            if image_resize_factor < 1.0:
                shrunken = True
                shrunken_shape = (np.array(image.shape) * image_resize_factor+1).astype(int)
                i_j = np.mgrid[0:shrunken_shape[0],0:shrunken_shape[1]].astype(float) / image_resize_factor
                simage = scipy.ndimage.map_coordinates(image, i_j)
                smask = scipy.ndimage.map_coordinates(mask.astype(float), i_j) > .99
                diameter = diameter * image_resize_factor + 1
                sigma = sigma * image_resize_factor
                figure.subplot_imshow_grayscale(0,1,simage,"shrunken")
            else:
                shrunken = False
                simage = image
                smask = mask
            normalized_image = 1 - stretch(simage, smask)
            
            log_image = laplacian_of_gaussian(normalized_image, smask, 
                                              diameter * 3/2, sigma)
            figure.subplot_imshow_grayscale(1,0,log_image,"LoG shrunken")
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
            log_image[log_image < log_threshold] = log_threshold
            log_image -= log_threshold
            figure.subplot_imshow_grayscale(1,1,log_image,"Expanded LoG")
            maxima_image = self.get_maxima(log_image, labeled_image,
                                           maxima_mask, image_resize_factor)
            dilated_image = scipy.ndimage.binary_dilation(maxima_image > 0, iterations=6)
            figure.subplot_imshow_bw(0,2,dilated_image, "maxima")
            color_image = np.zeros((image.shape[0],image.shape[1],3))
            color_image[:,:,2] = stretch(image)
            color_image[:,:,1] = stretch(log_image)
            color_image[:,:,0] = dilated_image
            color_image[:,:,1][dilated_image] = 0
            color_image[:,:,2][dilated_image] = 0
            figure.subplot_imshow_color(1,2,color_image)
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
        elif self.watershed_method == WA_DISTANCE:
            if distance_transformed_image == None:
                distance_transformed_image =\
                    scipy.ndimage.distance_transform_edt(labeled_image>0)
            watershed_image = -distance_transformed_image
            watershed_image = watershed_image - np.min(watershed_image)
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
            scipy.ndimage.label(maxima_image>0,np.ones((3,3),bool))
        markers = np.zeros(watershed_image.shape,np.int16)
        markers[labeled_maxima>0]=-labeled_maxima[labeled_maxima>0]
        #
        # Some labels have only one marker in them, some have multiple and
        # will be split up.
        # 
        watershed_boundaries = watershed(watershed_image,
                                         markers,
                                         np.ones((3,3),bool),
                                         mask=labeled_image!=0)
        watershed_boundaries = -watershed_boundaries
        
        return watershed_boundaries, object_count, maxima_suppression_size

    def get_maxima(self,image,labeled_image,maxima_mask,image_resize_factor):
        if image_resize_factor < 1.0:
            shape = np.array(image.shape) * image_resize_factor
            i_j = (np.mgrid[0:shape[0],0:shape[1]].astype(float) / 
                   image_resize_factor)
            resized_image = scipy.ndimage.map_coordinates(image, i_j)
        else:
            resized_image = image.copy()
        #
        # set all pixels that aren't local maxima to zero
        #
        maxima_image = resized_image
        maximum_filtered_image = scipy.ndimage.maximum_filter(maxima_image,
                                                              footprint=maxima_mask)
        maxima_image[resized_image < maximum_filtered_image] = 0
        #
        # Get rid of the "maxima" that are really large areas of zero
        #
        maxima_image[resized_image == 0] = 0
        binary_maxima_image = maxima_image > 0
        if image_resize_factor < 1.0:
            inverse_resize_factor = float(image.shape[0]) / float(maxima_image.shape[0])
            i_j = (np.mgrid[0:image.shape[0],
                               0:image.shape[1]].astype(float) / 
                   inverse_resize_factor)
            binary_maxima_image = scipy.ndimage.map_coordinates(binary_maxima_image.astype(float), i_j) > .5
            maxima_image = scipy.ndimage.map_coordinates(maxima_image, i_j)
            assert(binary_maxima_image.shape[0] == image.shape[0])
            assert(binary_maxima_image.shape[1] == image.shape[1])
        
        # Erode blobs of touching maxima to a single point
        
        shrunk_image = binary_shrink(binary_maxima_image)
        maxima_image[np.logical_not(shrunk_image)]=0
        maxima_image[labeled_image==0]=0
        return maxima_image
    
    def filter_on_size(self,labeled_image,object_count):
        """ Filter the labeled image based on the size range
        
        labeled_image - pixel image labels
        object_count - # of objects in the labeled image
        returns the labeled image, the labeled image before filtering and
        the labeled image with the small objects removed
        """
        unedited_labels = labeled_image.copy()
        if self.exclude_size.value and object_count > 0:
            areas = scipy.ndimage.measurements.sum(np.ones(labeled_image.shape),
                                                   labeled_image,
                                                   range(0,object_count+1))
            areas = np.array(areas,dtype=int)
            min_allowed_area = np.pi * self.size_range.min * self.size_range.min
            max_allowed_area = np.pi * self.size_range.max * self.size_range.max
            # area_image has the area of the object at every pixel within the object
            area_image = areas[labeled_image]
            labeled_image[area_image < min_allowed_area] = 0
            small_removed_labels = labeled_image.copy()
            labeled_image[area_image > max_allowed_area] = 0
        else:
            small_removed_labels = labeled_image.copy()
        return (labeled_image, unedited_labels, small_removed_labels)

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
    
    def display(self, frame, image, labeled_image, outline_image,statistics,
                image_set_number):
        """Display the image and labeling"""
        window_name = "CellProfiler(%s:%d)"%(self.module_name,self.module_num)
        my_frame=cpf.create_or_find(frame, title="Identify primary automatic", 
                                    name=window_name, subplots=(2,2))
        
        orig_axes     = my_frame.subplot(0,0)
        label_axes    = my_frame.subplot(1,0)
        outlined_axes = my_frame.subplot(0,1)
        table_axes    = my_frame.subplot(1,1)

        title = "Original image, cycle #%d"%(image_set_number)
        my_frame.subplot_imshow_grayscale(0, 0,
                                          image.pixel_data,
                                          title)
        my_frame.subplot_imshow_labels(1, 0, labeled_image, 
                                       self.object_name.value)

        if image.pixel_data.ndim == 2:
            outline_img = np.ndarray(shape=(image.pixel_data.shape[0],
                                               image.pixel_data.shape[1],3))
            outline_img[:,:,0] = image.pixel_data 
            outline_img[:,:,1] = image.pixel_data
            outline_img[:,:,2] = image.pixel_data
        else:
            outline_img = image.pixel_data.copy()
        outline_img[outline_image != 0,0]=1
        outline_img[outline_image != 0,1]=1 
        outline_img[outline_image != 0,2]=0
        
        title = "%s outlines"%(self.object_name.value) 
        my_frame.subplot_imshow(0,1,outline_img, title)
        
        table_axes.clear()
        table = table_axes.table(cellText=statistics,
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
             
    def get_categories(self,pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        if object_name == 'Image':
            return ['Threshold','Count']
        elif object_name == self.object_name.value:
            return ['Location']
        return []
      
    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        if object_name == 'Image' and category == 'Threshold':
            return ['FinalThreshold','OrigThreshold','WeightedVariance',
                    'SumOfEntropies']
        elif object_name == 'Image' and category == 'Count':
            return [ self.object_name.value ]
        elif object_name == self.object_name.value and category == 'Location':
            return ['Center_X','Center_Y']
        return []
    
    def get_measurement_objects(self, pipeline, object_name, category, 
                                measurement):
        """Return the objects associated with image measurements
        
        """
        if object_name == 'Image' and category == 'Threshold':
            return [ self.object_name.value ]
        return []
    
