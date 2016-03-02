import cellprofiler.icons 
from cellprofiler.gui.help import PROTIP_RECOMEND_ICON, PROTIP_AVOID_ICON, TECH_NOTE_ICON
__doc__ = '''
<b>Identify Primary Objects</b> identifies biological components of interest in grayscale images
containing bright objects on a dark background.
<hr>
<h4>What is a primary object?</h4>
In CellProfiler, we use the term <i>object</i> as a generic term to refer to an identifed
feature in an image, usually a cellular subcompartment of some kind (for example,
nuclei, cells, colonies, worms).
We define an object as <i>primary</i> when it can be found in an image without needing
the assistance of another cellular feature as a reference. For example:
<ul>
<li>The nuclei of cells are usually more easily identifiable due to their more uniform
morphology, high contrast relative to the background when stained, and good separation
between adjacent nuclei. These qualities typically make them appropriate candidates for primary object
identification.</li>
<li>In contrast, cells often have irregular intensity patterns and are lower-contrast with more diffuse
staining, making them more challenging to identify than nuclei. In addition, cells often
touch their neighbors making it harder to delineate the cell borders. For these reasons,
cell bodies are better suited for <i>secondary object</i> identification, since they are
best identified by using a previously-identified primary object (i.e, the nuclei) as
a reference. See the <b>IdentifySecondaryObjects</b> module for details on how to
do this.</li>
</ul>

<h4>What do I need as input?</h4>
To use this module, you will need to make sure that your input image has the following qualities:
<ul>
<li>The image should be grayscale.</li>
<li>The foreground (i.e, regions of interest) are lighter than the background.</li>
</ul>
If this is not the case, other modules can be used to pre-process the images to ensure they are in
the proper form:
<ul>
<li>If the objects in your images are dark on a light background, you
should invert the images using the Invert operation in the <b>ImageMath</b> module.</li>
<li>If you are working with color images, they must first be converted to
grayscale using the <b>ColorToGray</b> module.</li>
</ul>
<p>If you have images in which the foreground and background cannot be distinguished by intensity alone
(e.g, brightfield or DIC images), you can use the <a href="http://www.ilastik.org/">ilastik</a> package
bundled with CellProfiler to perform pixel-based classification (Windows only). You first train a classifier
by identifying areas of images that fall into one of several classes, such as cell body, nucleus,
background, etc. Then, the <b>ClassifyPixels</b> module takes the classifier and applies it to each image
to identify areas that correspond to the trained classes. The result of <b>ClassifyPixels</b> is
an image in which the region that falls into the class of interest is light on a dark background. Since
this new image satisfies the constraints above, it can be used as input in <b>IdentifyPrimaryObjects</b>.
See the <b>ClassifyPixels</b> module for more information.</p>

<h4>What do the settings mean?</h4>
See below for help on the individual settings. The following icons are used to call attention to
key items:
<ul>
<li><img src="memory:%(PROTIP_RECOMEND_ICON)s">&nbsp;Our recommendation or example use case
for which a particular setting is best used.</li>
<li><img src="memory:%(PROTIP_AVOID_ICON)s">&nbsp;Indicates a condition under which
a particular setting may not work well.</li>
<li><img src="memory:%(TECH_NOTE_ICON)s">&nbsp;Technical note. Provides more
detailed information on the setting.</li>
</ul>

<h4>What do I get as output?</h4>
A set of primary objects are produced by this module, which can be used in downstream modules
for measurement purposes or other operations.
See the section <a href="#Available_measurements">"Available measurements"</a> below for
the measurements that are produced by this module.

Once the module has finished processing, the module display window
will show the following panels:
<ul>
<li><i>Upper left:</i> The raw, original image.</li>
<li><i>Upper right:</i> The identified objects shown as a color
image where connected pixels that belong to the same object are assigned the same
color (<i>label image</i>). It is important to note that assigned colors are
arbitrary; they are used simply to help you distingush the various objects. </li>
<li><i>Lower left:</i> The raw image overlaid with the colored outlines of the
identified objects. Each object is assigned one of three (default) colors:
<ul>
<li>Green: Acceptable; passed all criteria</li>
<li>Magenta: Discarded based on size</li>
<li>Yellow: Discarded due to touching the border</li>
</ul>
If you need to change the color defaults, you can
make adjustments in <i>File > Preferences</i>.</li>
<li><i>Lower right:</i> A table showing some of the settings selected by the user, as well as
those calculated by the module in order to produce the objects shown.</li>
</ul>

<a name="Available_measurements">
<h4>Available measurements</h4>
<b>Image measurements:</b>
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

<b>Object measurements:</b>
<ul>
<li><i>Location_X, Location_Y:</i> The pixel (X,Y) coordinates of the primary
object centroids. The centroid is calculated as the center of mass of the binary
representation of the object.</li>
</ul>

<h4>Technical notes</h4>

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
where nuclei are the primary objects:
<ol>
<li>CellProfiler determines whether a foreground region is an individual
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

<h4>References</h4>
<ul>
<li>Malpica N, de Solorzano CO, Vaquero JJ, Santos, A, Vallcorba I,
Garcia-Sagredo JM, del Pozo F (1997) "Applying watershed
algorithms to the segmentation of clustered nuclei." <i> Cytometry</i> 28, 289-297.
(<a href="http://dx.doi.org/10.1002/(SICI)1097-0320(19970801)28:4<289::AID-CYTO3>3.0.CO;2-7">link</a>)</li>
<li>Meyer F, Beucher S (1990) "Morphological segmentation." <i>J Visual
Communication and Image Representation</i> 1, 21-46.
(<a href="http://dx.doi.org/10.1016/1047-3203(90)90014-M">link</a>)</li>
<li>Ortiz de Solorzano C, Rodriguez EG, Jones A, Pinkel D, Gray JW,
Sudar D, Lockett SJ. (1999) "Segmentation of confocal
microscope images of cell nuclei in thick tissue sections." <i>Journal of
Microscopy-Oxford</i> 193, 212-226.
(<a href="http://dx.doi.org/10.1046/j.1365-2818.1999.00463.x">link</a>)</li>
<li>W&auml;hlby C (2003) <i>Algorithms for applied digital image cytometry</i>, Ph.D.,
Uppsala University, Uppsala.</li>
<li>W&auml;hlby C, Sintorn IM, Erlandsson F, Borgefors G, Bengtsson E. (2004)
"Combining intensity, edge and shape information for 2D and 3D
segmentation of cell nuclei in tissue sections." <i>J Microsc</i> 215, 67-76.
(<a href="http://dx.doi.org/10.1111/j.0022-2720.2004.01338.x">link</a>)</li>
</ul>

<p>See also <b>IdentifySecondaryObjects</b>, <b>IdentifyTertiaryObjects</b>,
<b>IdentifyObjectsManually</b> and <b>ClassifyPixels</b> </p>
'''%globals()

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
from cellprofiler.settings import YES, NO
import cellprofiler.preferences as cpp
from centrosome.otsu import otsu
from centrosome.cpmorphology import fill_labeled_holes, strel_disk
from centrosome.cpmorphology import binary_shrink, relabel
from centrosome.cpmorphology import is_local_maximum
from centrosome.filter import stretch, laplacian_of_gaussian
from centrosome.watershed import watershed
from centrosome.propagate import propagate
from centrosome.smooth import smooth_with_noise
import centrosome.outline
import cellprofiler.objects
from cellprofiler.settings import AUTOMATIC
import centrosome.threshold as cpthresh
from identify import TSM_AUTOMATIC, TS_BINARY_IMAGE
from identify import draw_outline
from identify import FI_IMAGE_SIZE
from cellprofiler.gui.help import HELP_ON_MEASURING_DISTANCES, RETAINING_OUTLINES_HELP, NAMING_OUTLINES_HELP

#################################################
#
# Ancient offsets into the settings for Matlab pipelines
#
#################################################
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
WATERSHED_VAR                   = 11
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

#################################################
#
# V10 introduced a more unified handling of
#     threshold settings.
#
#################################################
OFF_THRESHOLD_METHOD_V9            = 6
OFF_THRESHOLD_CORRECTION_V9        = 7
OFF_THRESHOLD_RANGE_V9             = 8
OFF_OBJECT_FRACTION_V9             = 9
OFF_MANUAL_THRESHOLD_V9            = 19
OFF_BINARY_IMAGE_V9                = 20
OFF_TWO_CLASS_OTSU_V9              = 24
OFF_USE_WEIGHTED_VARIANCE_V9       = 25
OFF_ASSIGN_MIDDLE_TO_FOREGROUND_V9 = 26
OFF_THRESHOLDING_MEASUREMENT_V9    = 31
OFF_ADAPTIVE_WINDOW_METHOD_V9      = 32
OFF_ADAPTIVE_WINDOW_SIZE_V9        = 33
OFF_FILL_HOLES_V10                 = 12

'''The number of settings, exclusive of threshold settings in V10'''
N_SETTINGS_V10 = 22

UN_INTENSITY                    = "Intensity"
UN_SHAPE                        = "Shape"
UN_LOG                          = "Laplacian of Gaussian"
UN_NONE                         = "None"

WA_INTENSITY                    = "Intensity"
WA_SHAPE                        = "Shape"
WA_PROPAGATE                    = "Propagate"
WA_NONE                         = "None"

LIMIT_NONE = "Continue"
LIMIT_TRUNCATE = "Truncate"
LIMIT_ERASE = "Erase"

'''Never fill holes'''
FH_NEVER = "Never"
FH_THRESHOLDING = "After both thresholding and declumping"
FH_DECLUMP = "After declumping only"

FH_ALL = (FH_NEVER, FH_THRESHOLDING, FH_DECLUMP)

# Settings text which is referenced in various places in the help
SIZE_RANGE_SETTING_TEXT = "Typical diameter of objects, in pixel units (Min,Max)"
EXCLUDE_SIZE_SETTING_TEXT = "Discard objects outside the diameter range?"
AUTOMATIC_SMOOTHING_SETTING_TEXT = "Automatically calculate size of smoothing filter for declumping?"
SMOOTHING_FILTER_SIZE_SETTING_TEXT  = "Size of smoothing filter"
AUTOMATIC_MAXIMA_SUPPRESSION_SETTING_TEXT = "Automatically calculate minimum allowed distance between local maxima?"
WANTS_AUTOMATIC_LOG_DIAMETER_SETTING_TEXT = "Automatically calculate the size of objects for the Laplacian of Gaussian filter?"

# Icons for use in the help
INTENSITY_DECLUMPING_ICON = "IdentifyPrimaryObjects_IntensityDeclumping.png"
SHAPE_DECLUMPING_ICON = "IdentifyPrimaryObjects_ShapeDeclumping.png"

class IdentifyPrimaryObjects(cpmi.Identify):

    variable_revision_number = 10
    category =  "Object Processing"
    module_name = "IdentifyPrimaryObjects"

    def create_settings(self):

        self.image_name = cps.ImageNameSubscriber(
            "Select the input image",doc="""
            Select the image that you want to use to identify objects.""")

        self.object_name = cps.ObjectNameProvider(
            "Name the primary objects to be identified",
            "Nuclei",doc="""
            Enter the name that you want to call the objects identified by this module.""")

        self.size_range = cps.IntegerRange(
            SIZE_RANGE_SETTING_TEXT,
            (10,40), minval=1, doc='''
            This setting allows the user to make a distinction on the basis of size, which can
            be used in conjunction with the <i>%(EXCLUDE_SIZE_SETTING_TEXT)s</i> setting
            below to remove objects that fail this criteria.
            <dl>
            <dd><img src="memory:%(PROTIP_RECOMEND_ICON)s">&nbsp;
            The units used here are pixels so that it is easy to zoom in on objects and determine
            typical diameters. %(HELP_ON_MEASURING_DISTANCES)s</dd>
            </dl>
            <p>A few important notes:
            <ul>
            <li>Several other settings make use of the minimum object size entered here,
            whether the <i>%(EXCLUDE_SIZE_SETTING_TEXT)s</i> setting is used or not:
            <ul>
            <li><i>%(AUTOMATIC_SMOOTHING_SETTING_TEXT)s</i></li>
            <li><i>%(AUTOMATIC_MAXIMA_SUPPRESSION_SETTING_TEXT)s</i></li>
            <li><i>%(WANTS_AUTOMATIC_LOG_DIAMETER_SETTING_TEXT)s</i> (shown only if Laplacian of
            Gaussian is selected as the declumping method)</li>
            </ul>
            </li>
            <li>For non-round objects, the diameter here is actually the "equivalent diameter", i.e.,
            the diameter of a circle with the same area as the object.</li>
            </ul>
            </p>'''%globals())

        self.exclude_size = cps.Binary(
            EXCLUDE_SIZE_SETTING_TEXT,
            True, doc='''
            Select <i>%(YES)s</i> to discard objects outside the range you specified in the
            <i>%(SIZE_RANGE_SETTING_TEXT)s</i> setting. Select <i>%(NO)s</i> to ignore this
            criterion.
            <p>Objects discarded
            based on size are outlined in magenta in the module's display. See also the
            <b>FilterObjects</b> module to further discard objects based on some
            other measurement.
            <dl>
            <dd><img src="memory:%(PROTIP_RECOMEND_ICON)s">&nbsp;
            Select <i>%(YES)s</i> allows you to exclude small objects (e.g., dust, noise,
            and debris) or large objects (e.g., large clumps) if desired. </dd>
            </dl>
            '''%globals())

        self.merge_objects = cps.Binary(
            "Try to merge too small objects with nearby larger objects?",
            False, doc='''
            Select <i>%(YES)s</i> to cause objects that are
            smaller than the specified minimum diameter to be merged, if possible, with
            other surrounding objects.
            <p>This is helpful in cases when an object was
            incorrectly split into two objects, one of which is actually just a tiny
            piece of the larger object. However, this could be problematic if the other
            settings in the module are set poorly, producing many tiny objects; the module
            will take a very long time trying to merge the tiny objects back together again; you may
            not notice that this is the case, since it may successfully piece together the
            objects again. It is therefore a good idea to run the
            module first without merging objects to make sure the settings are
            reasonably effective.</p>'''%globals())

        self.exclude_border_objects = cps.Binary(
            "Discard objects touching the border of the image?",
            True, doc='''
            Select <i>%(YES)s</i> to discard objects that touch the border of the image.
            Select <i>%(NO)s</i> to ignore this criterion.
            <dl>
            <dd><img src="memory:%(PROTIP_RECOMEND_ICON)s">&nbsp;
            Removing objects that touch the image border is useful when you do
            not want to make downstream measurements of objects that are not fully within the
            field of view. For example, morphological measurements obtained from
            a portion of an object would not be accurate.</dd>
            </dl>
            <p>Objects discarded due to border touching are outlined in yellow in the module's display.
            Note that if a per-object thresholding method is used or if the image has been
            previously cropped or masked, objects that touch the
            border of the cropped or masked region may also discarded.</p>'''%globals())

        self.create_threshold_settings()

        self.unclump_method = cps.Choice(
            'Method to distinguish clumped objects',
            [UN_INTENSITY, UN_SHAPE, UN_LOG, UN_NONE], doc="""
            This setting allows you to choose the method that is used to segment
            objects, i.e., "declump" a large, merged object into individual objects of interest.
            To decide between these methods, you can run Test mode to see the results of each.
            <ul>
            <li>
            <table cellpadding="0"><tr><td>
            <i>%(UN_INTENSITY)s:</i> For objects that tend to have only a single peak of brightness
            (e.g. objects that are brighter towards their interiors and
            dimmer towards their edges), this option counts each intensity peak as a separate object.
            The objects can
            be any shape, so they need not be round and uniform in size as would be
            required for the <i>%(UN_SHAPE)s</i> option.
            <dl>
            <dd><img src="memory:%(PROTIP_RECOMEND_ICON)s">&nbsp;
            This choice is more successful when
            the objects have a smooth texture. By default, the image is automatically
            blurred to attempt to achieve appropriate smoothness (see <i>Smoothing filter</i> options),
            but overriding the default value can improve the outcome on
            lumpy-textured objects.</dd>
            </dl></td>
            <td><img src="memory:%(INTENSITY_DECLUMPING_ICON)s"></td>
            </tr></table>
            <dl>
            <dd><img src="memory:%(TECH_NOTE_ICON)s">&nbsp;
            The object centers are defined as local intensity maxima in the smoothed image.</dd></dl></li>

            <li>
            <table cellpadding="0"><tr><td>
            <i>%(UN_SHAPE)s:</i> For cases when there are definite indentations separating
            objects. The image is converted to
            black and white (binary) and the shape determines whether clumped
            objects will be distinguished. The
            declumping results of this method are affected by the thresholding
            method you choose.
            <dl>
            <dd><img src="memory:%(PROTIP_RECOMEND_ICON)s">&nbsp;
            This choice works best for objects that are round. In this case, the intensity
            patterns in the original image are largely irrelevant. Therefore, the cells need not be brighter
            towards the interior as is required for the <i>%(UN_INTENSITY)s</i> option.</dd>
            </dl></td>
            <td><img src="memory:%(SHAPE_DECLUMPING_ICON)s"></td>
            </tr></table>
            <dl>
            <dd><img src="memory:%(TECH_NOTE_ICON)s">&nbsp;
            The binary thresholded image is
            distance-transformed and object centers are defined as peaks in this
            image. A distance-transform gives each pixel a value equal to the distance
            to the nearest pixel below a certain threshold, so it indicates the <i>%(UN_SHAPE)s</i>
            of the object.</dd>
            </dl></li>
            <li><i>%(UN_LOG)s:</i> For objects that have an increasing intensity
            gradient toward their center, this option performs a Laplacian of Gaussian (or Mexican hat)
            transform on the image, which accentuates pixels that are local maxima of a desired size. It
            thresholds the result and finds pixels that are both local maxima and above
            threshold. These pixels are used as the seeds for objects in the watershed.</li>
            <li><i>%(UN_NONE)s:</i> If objects are well separated and bright relative to the
            background, it may be unnecessary to attempt to separate clumped objects.
            Using the very fast <i>%(UN_NONE)s</i> option, a simple threshold will be used to identify
            objects. This will override any declumping method chosen in the settings below.</li>
            </ul>"""%globals())

        self.watershed_method = cps.Choice(
            'Method to draw dividing lines between clumped objects',
            [WA_INTENSITY, WA_SHAPE, WA_PROPAGATE, WA_NONE], doc="""
            This setting allows you to choose the method that is used to draw the line
            bewteen segmented objects, provided that you have chosen to declump the objects.
            To decide between these methods, you can run Test mode to see the results of each.
            <ul>
            <li><i>%(WA_INTENSITY)s:</i> Works best where the dividing lines between clumped
            objects are dimmer than the remainder of the objects.
            <p><b>Technical description:</b>
            Using the previously identified local maxima as seeds, this method is a
            watershed (<i>Vincent and Soille, 1991</i>) on the intensity image.</p></li>
            <li><i>%(WA_SHAPE)s:</i> Dividing lines between clumped objects are based on the
            shape of the clump. For example, when a clump contains two objects, the
            dividing line will be placed where indentations occur between the two
            objects. The intensity patterns in the original image are largely irrelevant: the
            cells need not be dimmer along the lines between clumped objects.
            Technical description: Using the previously identified local maxima as seeds,
            this method is a
            watershed on the distance-transformed thresholded image.</li>
            <li><i>%(WA_PROPAGATE)s:</i> This method uses a propagation algorithm
            instead of a watershed. The image is ignored and the pixels are
            assigned to the objects by repeatedly adding unassigned pixels to
            the objects that are immediately adjacent to them. This method
            is suited in cases such as objects with branching extensions,
            for instance neurites, where the goal is to trace outward from
            the cell body along the branch, assigning pixels in the branch
            along the way. See the help for the <b>IdentifySecondary</b> module for more
            details on this method.</li>
            <li><i>%(WA_NONE)s</i>: If objects are well separated and bright relative to the
            background, it may be unnecessary to attempt to separate clumped objects.
            Using the very fast <i>%(WA_NONE)s</i> option, a simple threshold will be used to identify
            objects. This will override any declumping method chosen in the previous
            question.</li>
            </ul>"""%globals())

        self.automatic_smoothing = cps.Binary(
            AUTOMATIC_SMOOTHING_SETTING_TEXT,
            True, doc="""
            <i>(Used only when distinguishing between clumped objects)</i><br>
            Select <i>%(YES)s</i> to automatically calculate the amount of smoothing
            applied to the image to assist in declumping. Select <i>%(NO)s</i> to
            manually enter the smoothing filter size.

            <p>This setting, along with the <i>Minimum allowed distance between local maxima</i>
            setting, affects whether objects
            close to each other are considered a single object or multiple objects.
            It does not affect the dividing lines between an object and the
            background.</p>

            <p>Please note that this smoothing setting is applied after thresholding,
            and is therefore distinct from the threshold smoothing method setting above,
            which is applied <i>before</i> thresholding.</p>

            <p>The size of the smoothing filter is automatically
            calculated based on the <i>%(SIZE_RANGE_SETTING_TEXT)s</i> setting above.
            If you see too many objects merged that ought to be separate
            or too many objects split up that
            ought to be merged, you may want to override the automatically
            calculated value.</p>"""%globals())

        self.smoothing_filter_size = cps.Integer(
            SMOOTHING_FILTER_SIZE_SETTING_TEXT, 10, doc="""
            <i>(Used only when distinguishing between clumped objects)</i> <br>
            If you see too many objects merged that ought to be separated
            (under-segmentation), this value
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
            AUTOMATIC_MAXIMA_SUPPRESSION_SETTING_TEXT,
            True, doc="""
            <i>(Used only when distinguishing between clumped objects)</i><br>
            Select <i>%(YES)s</i> to automatically calculate the distance between
            intensity maxima to assist in declumping. Select <i>%(NO)s</i> to
            manually enter the permissible maxima distance.

            <p>This setting, along with the <i>%(SMOOTHING_FILTER_SIZE_SETTING_TEXT)s</i> setting,
            affects whether objects close to each other are considered a single object
            or multiple objects. It does not affect the dividing lines between an object and the
            background. Local maxima that are closer together than the minimum
            allowed distance will be suppressed (the local intensity histogram is smoothed to
            remove the peaks within that distance). The distance can be automatically
            calculated based on the minimum entered for the
            <i>%(SIZE_RANGE_SETTING_TEXT)s</i> setting above,
            but if you see too many objects merged that ought to be separate, or
            too many objects split up that ought to be merged, you may want to override the
            automatically calculated value."""%globals())

        self.maxima_suppression_size = cps.Float(
            'Suppress local maxima that are closer than this minimum allowed distance',
            7, minval=0, doc="""
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
            True, doc="""
            <i>(Used only when distinguishing between clumped objects)</i><br>
            Select <i>%(YES)s</i> to down-sample the image for declumping. This can be
            helpful for saving processing time on large images.
            <p>Note that if you have entered a minimum object diameter of 10 or less, checking
            this box will have no effect.</p>"""%globals())

        self.should_save_outlines = cps.Binary(
            'Retain outlines of the identified objects?', False, doc="""
            %(RETAINING_OUTLINES_HELP)s"""%globals())

        self.save_outlines = cps.OutlineNameProvider(
            'Name the outline image',"PrimaryOutlines", doc="""
            %(NAMING_OUTLINES_HELP)s"""%globals())

        self.fill_holes = cps.Choice(
            'Fill holes in identified objects?',
            FH_ALL, value = FH_THRESHOLDING,
            doc="""
            This option controls how holes are filled in:
            <ul>
            <li><i>%(FH_THRESHOLDING)s:</i> Fill in background holes
            that are smaller than the maximum object size prior to declumping
            and to fill in any holes after declumping.</li>
            <li><i>%(FH_DECLUMP)s:</i> Fill in background holes
            located within identified objects after declumping.</li>
            <li><i>%(FH_NEVER)s:</i> Leave holes within objects.<br>
            Please note that if a foreground object is located within a hole
            and this option is enabled, the object will be lost when the hole
            is filled in.</li>
            </ul>"""%globals())

        self.wants_automatic_log_threshold = cps.Binary(
            'Automatically calculate the threshold using the Otsu method?', True)

        self.manual_log_threshold = cps.Float(
            'Enter Laplacian of Gaussian threshold', .5, 0, 1)

        self.wants_automatic_log_diameter = cps.Binary(
            WANTS_AUTOMATIC_LOG_DIAMETER_SETTING_TEXT, True,doc="""
            <i>(Used only when applying the LoG thresholding method)</i><br>
            <p>Select <i>%(YES)s</i> to use the filtering diameter range above
            when constructing the LoG filter. </p>
            <p>Select <i>%(NO)s</i> in order to manually specify the size.
            You may want to specify a custom size if you want to filter
            using loose criteria, but have objects that are generally of
            similar sizes.</p>"""%globals())

        self.log_diameter = cps.Float(
            'Enter LoG filter diameter',
            5, minval=1, maxval=100,doc="""
            <i>(Used only when applying the LoG thresholding method)</i><br>
            The size to use when calculating the LoG filter. The filter enhances
            the local maxima of objects whose diameters are roughly the entered
            number or smaller.""")

        self.limit_choice = cps.Choice(
            "Handling of objects if excessive number of objects identified",
            [LIMIT_NONE, LIMIT_TRUNCATE, LIMIT_ERASE],doc = """
            This setting deals with images that are segmented
            into an unreasonable number of objects. This might happen if
            the module calculates a low threshold or if the image has
            unusual artifacts. <b>IdentifyPrimaryObjects</b> can handle
            this condition in one of three ways:
            <ul>
            <li><i>%(LIMIT_NONE)s</i>: Don't check for large numbers
            of objects.</li>
            <li><i>%(LIMIT_TRUNCATE)s</i>: Limit the number of objects.
            Arbitrarily erase objects to limit the number to the maximum
            allowed.</li>
            <li><i>%(LIMIT_ERASE)s</i>: Erase all objects if the number of
            objects exceeds the maximum. This results in an image with
            no primary objects. This option is a good choice if a large
            number of objects indicates that the image should not be
            processed.</li>
            </ul>""" % globals())

        self.maximum_object_count = cps.Integer(
            "Maximum number of objects",
            value = 500, minval = 2,doc = """
            <i>(Used only when handling images with large numbers of objects by truncating)</i> <br>
            This setting limits the number of objects in the
            image. See the documentation for the previous setting
            for details.""")

    def settings(self):
        return [self.image_name, self.object_name, self.size_range,
                self.exclude_size, self.merge_objects,
                self.exclude_border_objects, self.unclump_method,
                self.watershed_method, self.smoothing_filter_size,
                self.maxima_suppression_size, self.low_res_maxima,
                self.save_outlines, self.fill_holes,
                self.automatic_smoothing, self.automatic_suppression,
                self.should_save_outlines,
                self.wants_automatic_log_threshold,
                self.manual_log_threshold,
                self.wants_automatic_log_diameter, self.log_diameter,
                self.limit_choice, self.maximum_object_count] + \
               self.get_threshold_settings()

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
                new_setting_values[SAVE_OUTLINES_VAR] = cps.NONE
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
            setting_values = setting_values + [cps.NONE]
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

        if (not from_matlab) and variable_revision_number == 9:
            #
            # Unified threshold measurements.
            #
            threshold_method = setting_values[OFF_THRESHOLD_METHOD_V9]
            threshold_correction = setting_values[OFF_THRESHOLD_CORRECTION_V9]
            threshold_range = setting_values[OFF_THRESHOLD_RANGE_V9]
            object_fraction = setting_values[OFF_OBJECT_FRACTION_V9]
            manual_threshold = setting_values[OFF_MANUAL_THRESHOLD_V9]
            binary_image = setting_values[OFF_BINARY_IMAGE_V9]
            two_class_otsu = setting_values[OFF_TWO_CLASS_OTSU_V9]
            use_weighted_variance = setting_values[OFF_USE_WEIGHTED_VARIANCE_V9]
            assign_middle_to_foreground = setting_values[OFF_ASSIGN_MIDDLE_TO_FOREGROUND_V9]
            thresholding_measurement = setting_values[OFF_THRESHOLDING_MEASUREMENT_V9]
            adaptive_window_method = setting_values[OFF_ADAPTIVE_WINDOW_METHOD_V9]
            adaptive_window_size = setting_values[OFF_ADAPTIVE_WINDOW_SIZE_V9]

            threshold_settings = self.upgrade_legacy_threshold_settings(
                threshold_method, TSM_AUTOMATIC, threshold_correction,
                threshold_range, object_fraction, manual_threshold,
                thresholding_measurement, binary_image, two_class_otsu,
                use_weighted_variance, assign_middle_to_foreground,
                adaptive_window_method, adaptive_window_size)

            setting_values = \
                setting_values[:OFF_THRESHOLD_METHOD_V9] + \
                setting_values[(OFF_OBJECT_FRACTION_V9+1):
                               OFF_MANUAL_THRESHOLD_V9] + \
                setting_values[(OFF_BINARY_IMAGE_V9+1):
                               OFF_TWO_CLASS_OTSU_V9] + \
                setting_values[(OFF_ASSIGN_MIDDLE_TO_FOREGROUND_V9+1):
                               OFF_THRESHOLDING_MEASUREMENT_V9] + \
                threshold_settings
            variable_revision_number = 10
        if variable_revision_number == 10:
            setting_values = list(setting_values)
            if setting_values[OFF_FILL_HOLES_V10] == cps.NO:
                setting_values[OFF_FILL_HOLES_V10] = FH_NEVER
            elif setting_values[OFF_FILL_HOLES_V10] == cps.YES:
                setting_values[OFF_FILL_HOLES_V10] = FH_THRESHOLDING

        # upgrade threshold settings
        setting_values = setting_values[:N_SETTINGS_V10] + \
            self.upgrade_threshold_settings(setting_values[N_SETTINGS_V10:])
        return setting_values, variable_revision_number, from_matlab

    def help_settings(self):
        return [self.image_name,
                self.object_name,
                self.size_range,
                self.exclude_size,
                self.merge_objects,
                self.exclude_border_objects
                ] +  self.get_threshold_help_settings() + [
                self.wants_automatic_log_diameter,
                self.log_diameter,
                self.wants_automatic_log_threshold,
                self.manual_log_threshold,
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
              self.exclude_size, self.exclude_border_objects
              ] + self.get_threshold_visible_settings()
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

    def run(self,workspace):
        """Run the module

        pipeline     - instance of CellProfiler.Pipeline for this run
        workspace    - contains
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
        """
        image_name = self.image_name.value
        image = workspace.image_set.get_image(image_name)
        workspace.display_data.statistics = []
        binary_image = self.threshold_image(image_name, workspace)
        #
        # Fill background holes inside foreground objects
        #
        def size_fn(size, is_foreground):
            return size < self.size_range.max * self.size_range.max

        if self.fill_holes.value == FH_THRESHOLDING:
            binary_image = fill_labeled_holes(binary_image, size_fn=size_fn)

        labeled_image,object_count = scipy.ndimage.label(binary_image,
                                                         np.ones((3,3),bool))
        labeled_image, object_count, maxima_suppression_size, \
            LoG_threshold, LoG_filter_diameter = \
            self.separate_neighboring_objects(workspace,
                                              labeled_image,
                                              object_count)
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

        #
        # Fill holes again after watershed
        #
        if self.fill_holes != FH_NEVER:
            labeled_image = fill_labeled_holes(labeled_image)

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
        outline_image = centrosome.outline.outline(labeled_image)
        outline_size_excluded_image = centrosome.outline.outline(size_excluded_labeled_image)
        outline_border_excluded_image = centrosome.outline.outline(border_excluded_labeled_image)

        if self.show_window:
            statistics = workspace.display_data.statistics
            statistics.append(["# of accepted objects",
                               "%d"%(object_count)])
            if object_count > 0:
                areas = scipy.ndimage.sum(np.ones(labeled_image.shape), labeled_image, np.arange(1, object_count + 1))
                areas.sort()
                low_diameter  = (math.sqrt(float(areas[object_count / 10]) / np.pi) * 2)
                median_diameter = (math.sqrt(float(areas[object_count / 2]) / np.pi) * 2)
                high_diameter = (math.sqrt(float(areas[object_count * 9 / 10]) / np.pi) * 2)
                statistics.append(["10th pctile diameter",
                                   "%.1f pixels" % (low_diameter)])
                statistics.append(["Median diameter",
                                   "%.1f pixels" % (median_diameter)])
                statistics.append(["90th pctile diameter",
                                   "%.1f pixels" % (high_diameter)])
                object_area = np.sum(areas)
                total_area  = np.product(labeled_image.shape[:2])
                statistics.append(["Area covered by objects",
                                   "%.1f %%" % (100.0 * float(object_area) /
                                              float(total_area))])
                if self.threshold_scope != TS_BINARY_IMAGE:
                    statistics.append(["Thresholding filter size",
                        "%.1f"%(workspace.display_data.threshold_sigma)])
                if self.unclump_method != UN_NONE:
                    if self.unclump_method == UN_LOG:
                        statistics.append(["LoG threshold",
                                   "%.1f"%(LoG_threshold)])
                        statistics.append(["LoG filter diameter",
                                   "%.1f"%(LoG_filter_diameter)])
                    statistics.append(["Declumping smoothing filter size",
                                   "%.1f"%(self.calc_smoothing_filter_size())])
                    statistics.append(["Maxima suppression size",
                                   "%.1f"%(maxima_suppression_size)])
            workspace.display_data.image = image.pixel_data
            workspace.display_data.labeled_image = labeled_image
            workspace.display_data.size_excluded_labels = size_excluded_labeled_image
            workspace.display_data.border_excluded_labels = border_excluded_labeled_image

        # Add image measurements
        objname = self.object_name.value
        measurements = workspace.measurements
        cpmi.add_object_count_measurements(measurements,
                                           objname, object_count)
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
                r.seed(abs(np.sum(labeled_image)) % (2 ** 16))
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

    def smooth_image(self, image, mask):
        """Apply the smoothing filter to the image"""

        filter_size = self.calc_smoothing_filter_size()
        if filter_size == 0:
            return image
        sigma = filter_size / 2.35
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

    def separate_neighboring_objects(self, workspace, labeled_image,
                                     object_count):
        """Separate objects based on local maxima or distance transform

        workspace - get the image from here

        labeled_image - image labeled by scipy.ndimage.label

        object_count  - # of objects in image

        returns revised labeled_image, object count, maxima_suppression_size,
        LoG threshold and filter diameter
        """
        if self.unclump_method == UN_NONE or self.watershed_method == WA_NONE:
            return labeled_image, object_count, 7, 0.5, 5

        cpimage = workspace.image_set.get_image(
            self.image_name.value, must_be_grayscale=True)
        image = cpimage.pixel_data
        mask = cpimage.mask

        reported_LoG_filter_diameter = 5
        reported_LoG_threshold = 0.5
        blurred_image = self.smooth_image(image, mask)
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
        maxima_mask = strel_disk(max(1, maxima_suppression_size-.5))
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
            maxima_image = self.get_maxima(blurred_image,
                                           labeled_image,
                                           maxima_mask,
                                           image_resize_factor)
        elif self.unclump_method == UN_SHAPE:
            if self.fill_holes == FH_NEVER:
                # For shape, even if the user doesn't want to fill holes,
                # a point far away from the edge might be near a hole.
                # So we fill just for this part.
                foreground = fill_labeled_holes(labeled_image) > 0
            else:
                foreground = labeled_image > 0
            distance_transformed_image =\
                scipy.ndimage.distance_transform_edt(foreground)
            # randomize the distance slightly to get unique maxima
            np.random.seed(0)
            distance_transformed_image +=\
                np.random.uniform(0,.001,distance_transformed_image.shape)
            maxima_image = self.get_maxima(distance_transformed_image,
                                           labeled_image,
                                           maxima_mask,
                                           image_resize_factor)
        else:
            raise ValueError("Unsupported local maxima method: %s" % (self.unclump_method.value))

        # Create the image for watershed
        if self.watershed_method == WA_INTENSITY:
            # use the reverse of the image to get valleys at peaks
            watershed_image = 1-image
        elif self.watershed_method == WA_SHAPE:
            if distance_transformed_image is None:
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

    def display(self, workspace, figure):
        if self.show_window:
            """Display the image and labeling"""
            figure.set_subplots((2, 2))


            orig_axes     = figure.subplot(0,0)
            label_axes    = figure.subplot(1,0, sharexy = orig_axes)
            outlined_axes = figure.subplot(0,1, sharexy = orig_axes)

            title = "Input image, cycle #%d"%(workspace.measurements.image_number,)
            image = workspace.display_data.image
            labeled_image = workspace.display_data.labeled_image
            size_excluded_labeled_image = workspace.display_data.size_excluded_labels
            border_excluded_labeled_image = workspace.display_data.border_excluded_labels

            ax = figure.subplot_imshow_grayscale(0, 0, image, title)
            figure.subplot_imshow_labels(1, 0, labeled_image,
                                         self.object_name.value,
                                         sharexy = ax)

            cplabels = [
                dict(name = self.object_name.value,
                     labels = [labeled_image]),
                dict(name = "Objects filtered out by size",
                     labels = [size_excluded_labeled_image]),
                dict(name = "Objects touching border",
                     labels = [border_excluded_labeled_image])]
            title = "%s outlines"%(self.object_name.value)
            figure.subplot_imshow_grayscale(
                0, 1, image, title, cplabels = cplabels, sharexy = ax)

            figure.subplot_table(
                1, 1,
                [[x[1]] for x in workspace.display_data.statistics],
                row_labels = [x[0] for x in workspace.display_data.statistics])

    def calc_smoothing_filter_size(self):
        """Return the size of the smoothing filter, calculating it if in automatic mode"""
        if self.automatic_smoothing.value:
            return 2.35*self.size_range.min/3.5;
        else:
            return self.smoothing_filter_size.value

    def is_object_identification_module(self):
        '''IdentifyPrimaryObjects makes primary objects sets so it's a identification module'''
        return True

    def get_measurement_objects_name(self):
        '''Return the name to be appended to image measurements made by module
        '''
        return self.object_name.value

    def get_measurement_columns(self, pipeline):
        '''Column definitions for measurements made by IdentifyPrimAutomatic'''
        columns = cpmi.get_object_measurement_columns(self.object_name.value)
        columns += self.get_threshold_measurement_columns(pipeline)
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
                                                      category, measurement)
IdentifyPrimAutomatic = IdentifyPrimaryObjects
