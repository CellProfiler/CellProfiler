#################################
#
# Imports from useful Python libraries
#
#################################

import numpy
import scipy.ndimage

#################################
#
# Imports from CellProfiler
#
##################################

# Reference
# 
# If you're using a technique or method that's used in this module 
# and has a publication attached to it, please include the paper link below.
# Otherwise, remove the line below and remove the "References" section from __doc__.
#

cite_paper_link = "https://doi.org/10.1016/1047-3203(90)90014-M"

__doc__ = """\
ImageTemplate
=============

**ImageTemplate** is an example image processing module. It's recommended to
put a brief description of this module here and go into more detail below.

This is an example of a module that takes one image as an input and
produces a second image for downstream processing. You can use this as
a starting point for your own module: rename this file and put it in your
plugins directory.

The text you see here will be displayed as the help for your module, formatted
as `reStructuredText <http://docutils.sourceforge.net/rst.html>`_.

Note whether or not this module supports 3D image data and respects masks.
A module which respects masks applies an image's mask and operates only on
the image data not obscured by the mask. Update the table below to indicate 
which image processing features this module supports.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

Is there another **Module** that is related to this one? If so, refer
to that **Module** in this section. Otherwise, this section can be omitted.

What do I need as input?
^^^^^^^^^^^^^^^^^^^^^^^^

Are there any assumptions about input data someone using this module
should be made aware of? For example, is there a strict requirement that
image data be single-channel, or that the foreground is brighter than
the background? Describe any assumptions here.

This section can be omitted if there is no requirement on the input.

What do I get as output?
^^^^^^^^^^^^^^^^^^^^^^^^

Describe the output of this module. This is necessary if the output is
more complex than a single image. For example, if there is data displayed
over the image then describe what the data represents.

This section can be omitted if there is no specialized output.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Describe the measurements made by this module, if any. Typically, measurements
are described in the following format:

**Measurement category:**

-  *MeasurementName*: A brief description of the measurement.
-  *MeasurementName*: A brief description of the measurement.

**Measurement category:**

-  *MeasurementName*: A brief description of the measurement.
-  *MeasurementName*: A brief description of the measurement.

This section can be omitted if no measurements are made by this module.

Technical notes
^^^^^^^^^^^^^^^

Include implementation details or notes here. Additionally provide any 
other background information about this module, including definitions
or adopted conventions. Information which may be too specific to fit into
the general description should be provided here.

Omit this section if there is no technical information to mention.

References
^^^^^^^^^^

Provide citations here, if appropriate. Citations are formatted as a list and,
wherever possible, include a link to the original work. For example,

-  Meyer F, Beucher S (1990) “Morphological segmentation.” *J Visual
   Communication and Image Representation* 1, 21-46.
   {cite_paper_link}
"""

#
# Constants
#
# It's good programming practice to replace things like strings with
# constants if they will appear more than once in your program. That way,
# if someone wants to change the text, that text will change everywhere.
# Also, you can't misspell it by accident.
#
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.text import Float

GRADIENT_MAGNITUDE = "Gradient magnitude"
GRADIENT_DIRECTION_X = "Gradient direction - X"
GRADIENT_DIRECTION_Y = "Gradient direction - Y"


#
# The module class.
#
# Your module should "inherit" from cellprofiler_core.module.Module, or a
# subclass of cellprofiler_core.module.Module. This module inherits from
# cellprofiler_core.module.ImageProcessing, which is the base class for
# image processing modules. Image processing modules take an image as
# input and output an image.
#
# This module will use the methods from cellprofiler_core.module.ImageProcessing
# unless you re-implement them. You can let cellprofiler_core.module.ImageProcessing
# do most of the work and implement only what you need.
#
# Other classes you can inherit from are:
#
# -  cellprofiler_core.module.image_segmentation.ImageSegmentation: modules which take an image
#    as input and output a segmentation (objects) should inherit from this
#    class.
# -  cellprofiler_core.module.image_segmentation.ObjectProcessing: modules which operate on objects
#    should inherit from this class. These are modules that take objects as
#    input and output new objects.
#
class ImageTemplate(ImageProcessing):
    #
    # The module starts by declaring the name that's used for display,
    # the category under which it is stored and the variable revision
    # number which can be used to provide backwards compatibility if
    # you add user-interface functionality later.
    #
    # This module's category is "Image Processing" which is defined
    # by its superclass.
    #
    module_name = "ImageTemplate"

    variable_revision_number = 1

    #
    # Citation
    #
    # If you're using a technique or method that's used in this module 
    # and has a publication attached to it, please include the paper link below.
    # Edit accordingly and add the link for the paper as "https://doi.org/XXXX".
    # If no citation is necessary, remove the "doi" dictionary below. 
    #

    doi = {"Please cite the following when using ImageTemplate:": 'https://doi.org/10.1016/1047-3203(90)90014-M', 
           "If you're also using specific technique X, cite:": 'https://doi.org/10.1016/1047-3203(90)90014-M'}
    #
    # "create_settings" is where you declare the user interface elements
    # (the "settings") which the user will use to customize your module.
    #
    # You can look at other modules and in cellprofiler_core.settings for
    # settings you can use.
    #
    def create_settings(self):
        #
        # The superclass (ImageProcessing) defines two
        # settings for image input and output:
        #
        # -  x_name: an ImageNameSubscriber which "subscribes" to all
        #    ImageNameProviders in prior modules. Modules before yours will
        #    put images into CellProfiler. The ImageNameSubscriber gives
        #    your user a list of these images which can then be used as inputs
        #    in your module.
        # -  y_name: an ImageName makes the image available to subsequent
        #    modules.
        super(ImageTemplate, self).create_settings()

        #
        # reST help that gets displayed when the user presses the
        # help button to the right of the edit box.
        #
        # The superclass defines some generic help test. You can add
        # module-specific help text by modifying the setting's "doc"
        # string.
        #
        self.x_name.doc = """\
This is the image that the module operates on. You can choose any image
that is made available by a prior module.

**ImageTemplate** will do something to this image.
"""

        #
        # Here's a choice box - the user gets a drop-down list of what
        # can be done.
        #
        self.gradient_choice = Choice(
            text="Gradient choice:",
            # The choice takes a list of possibilities. The first one
            # is the default - the one the user will typically choose.
            choices=[GRADIENT_DIRECTION_X, GRADIENT_DIRECTION_Y, GRADIENT_MAGNITUDE],
            # The default value is the first choice in choices. You can
            # specify a different initial value using the value keyword.
            value=GRADIENT_MAGNITUDE,
            #
            # Here, in the documentation, we do a little trick so that
            # we use the actual text that's displayed in the documentation.
            #
            # {GRADIENT_MAGNITUDE} will get changed into "Gradient magnitude"
            # etc. Python will look in keyword arguments for format()
            # for the "GRADIENT_" names and paste them in where it sees
            # a matching {GRADIENT_...}.
            #
            doc="""\
Choose what to calculate:

-  *{GRADIENT_MAGNITUDE}*: calculate the magnitude of the gradient at
   each pixel.
-  *{GRADIENT_DIRECTION_X}*: get the relative contribution of the
   gradient in the X direction (.5 = no contribution, 0 to .5 =
   decreasing with increasing X, .5 to 1 = increasing with increasing
   X).
-  *{GRADIENT_DIRECTION_Y}*: get the relative contribution of the
   gradient in the Y direction.
""".format(
                **{
                    "GRADIENT_MAGNITUDE": GRADIENT_MAGNITUDE,
                    "GRADIENT_DIRECTION_X": GRADIENT_DIRECTION_X,
                    "GRADIENT_DIRECTION_Y": GRADIENT_DIRECTION_Y,
                }
            ),
        )

        #
        # A binary setting displays a checkbox.
        #
        self.automatic_smoothing = Binary(
            text="Automatically choose the smoothing scale?",
            value=True,  # The default value is to choose automatically
            doc="The module will automatically choose a smoothing scale for you if you leave this checked.",
        )

        #
        # We do a little smoothing which supplies a scale to the gradient.
        #
        # We use a float setting so that the user can give us a number
        # for the scale. The control will turn red if the user types in
        # an invalid scale.
        #
        self.scale = Float(
            text="Scale",
            value=1,  # The default value is 1 - a short-range scale
            minval=0.1,  # We don't let the user type in really small values
            maxval=100,  # or large values
            doc="""\
This is a scaling factor that supplies the sigma for a gaussian that's
used to smooth the image. The gradient is calculated on the smoothed
image, so large scales will give you long-range gradients and small
scales will give you short-range gradients.
""",
        )

    #
    # The "settings" method tells CellProfiler about the settings you
    # have in your module. CellProfiler uses the list for saving
    # and restoring values for your module when it saves or loads a
    # pipeline file.
    #
    def settings(self):
        #
        # The superclass's "settings" method returns [self.x_name, self.y_name],
        # which are the input and output image settings.
        #
        settings = super(ImageTemplate, self).settings()

        # Append additional settings here.
        return settings + [self.gradient_choice, self.automatic_smoothing, self.scale]

    #
    # "visible_settings" tells CellProfiler which settings should be
    # displayed and in what order.
    #
    # You don't have to implement "visible_settings" - if you delete
    # visible_settings, CellProfiler will use "settings" to pick settings
    # for display.
    #
    def visible_settings(self):
        #
        # The superclass's "visible_settings" method returns [self.x_name,
        # self.y_name], which are the input and output image settings.
        #
        visible_settings = super(ImageTemplate, self).visible_settings()

        # Configure the visibility of additional settings below.
        visible_settings += [self.gradient_choice, self.automatic_smoothing]

        #
        # Show the user the scale only if self.wants_smoothing is checked
        #
        if not self.automatic_smoothing:
            visible_settings += [self.scale]

        return visible_settings

    #
    # CellProfiler calls "run" on each image set in your pipeline.
    #
    def run(self, workspace):
        #
        # The superclass's "run" method handles retreiving the input image
        # and saving the output image. Module-specific behavior is defined
        # by setting "self.function", defined in this module. "self.function"
        # is called after retrieving the input image and before saving
        # the output image.
        #
        # The first argument of "self.function" is always the input image
        # data (as a numpy array). The remaining arguments are the values of
        # the module settings as they are returned from "settings" (excluding
        # "self.y_data", or the output image).
        #
        self.function = gradient_image

        super(ImageTemplate, self).run(workspace)

    #
    # "volumetric" indicates whether or not this module supports 3D images.
    # The "gradient_image" function is inherently 2D, and we've noted this
    # in the documentation for the module. Explicitly return False here
    # to indicate that 3D images are not supported.
    #
    def volumetric(self):
        return False


#
# This is the function that gets called during "run" to create the output image.
# The first parameter must be the input image data. The remaining parameters are
# the additional settings defined in "settings", in the order they are returned.
#
# This function must return the output image data (as a numpy array).
#
def gradient_image(pixels, gradient_choice, automatic_smoothing, scale):
    #
    # Get the smoothing parameter
    #
    if automatic_smoothing:
        # Pick the mode of the power spectrum - obviously this
        # is pretty hokey, not intended to really find a good number.
        #
        fft = numpy.fft.fft2(pixels)
        power2 = numpy.sqrt((fft * fft.conjugate()).real)
        mode = numpy.argwhere(power2 == power2.max())[0]
        scale = numpy.sqrt(numpy.sum((mode + 0.5) ** 2))

    gradient_magnitude = scipy.ndimage.gaussian_gradient_magnitude(pixels, scale)

    if gradient_choice == GRADIENT_MAGNITUDE:
        gradient_image = gradient_magnitude
    else:
        # Image data is indexed by rows and columns, with a given point located at
        # position (row, column). Here, x represents the column coordinate (at index 1)
        # and y represents the row coordinate (at index 0).
        #
        # You can learn more about image coordinate systems here:
        # http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        x = scipy.ndimage.correlate1d(gradient_magnitude, [-1, 0, 1], 1)
        y = scipy.ndimage.correlate1d(gradient_magnitude, [-1, 0, 1], 0)
        norm = numpy.sqrt(x ** 2 + y ** 2)
        if gradient_choice == GRADIENT_DIRECTION_X:
            gradient_image = 0.5 + x / norm / 2
        else:
            gradient_image = 0.5 + y / norm / 2

    return gradient_image
