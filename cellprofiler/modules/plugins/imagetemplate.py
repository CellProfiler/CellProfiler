'''<b>ImageTemplate</b> - an example image processing module
<hr>
This is an example of a module that takes one image as an input and
produces a second image for downstream processing. You can use this as
a starting point for your own module: rename this file and put it in your
plugins directory.

The text you see here will be displayed as the help for your module. You
can use HTML markup here and in the settings text; the Python HTML control
does not fully support the HTML specification, so you may have to experiment
to get it to display correctly.
'''
#################################
#
# Imports from useful Python libraries
#
#################################

import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude, correlate1d

#################################
#
# Imports from CellProfiler
#
# The package aliases are the standard ones we use
# throughout the code.
#
##################################

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.setting as cps

###################################
#
# Constants
#
# It's good programming practice to replace things like strings with
# constants if they will appear more than once in your program. That way,
# if someone wants to change the text, that text will change everywhere.
# Also, you can't misspell it by accident.
###################################

GRADIENT_MAGNITUDE = "Gradient magnitude"
GRADIENT_DIRECTION_X = "Gradient direction - X"
GRADIENT_DIRECTION_Y = "Gradient direction - Y"


###################################
#
# The module class
#
# Your module should "inherit" from cellprofiler.cpmodule.CPModule.
# This means that your module will use the methods from CPModule unless
# you re-implement them. You can let CPModule do most of the work and
# implement only what you need.
#
###################################

class ImageTemplate(cpm.Module):
    ###############################################
    #
    # The module starts by declaring the name that's used for display,
    # the category under which it is stored and the variable revision
    # number which can be used to provide backwards compatibility if
    # you add user-interface functionality later.
    #
    ###############################################
    module_name = "ImageTemplate"
    category = "Image Processing"
    variable_revision_number = 1

    ###############################################
    #
    # create_settings is where you declare the user interface elements
    # (the "settings") which the user will use to customize your module.
    #
    # You can look at other modules and in cellprofiler.settings for
    # settings you can use.
    #
    ################################################

    def create_settings(self):
        #
        # The ImageNameSubscriber "subscribes" to all ImageNameProviders in
        # prior modules. Modules before yours will put images into CellProfiler.
        # The ImageSubscriber gives your user a list of these images
        # which can then be used as inputs in your module.
        #
        self.input_image_name = cps.ImageNameSubscriber(
                # The text to the left of the edit box
                "Input image name:",
                # HTML help that gets displayed when the user presses the
                # help button to the right of the edit box
                doc="""This is the image that the module operates on. You can
            choose any image that is made available by a prior module.
            <br>
            <b>ImageTemplate</b> will do something to this image.
            """)
        #
        # The ImageNameProvider makes the image available to subsequent
        # modules.
        #
        self.output_image_name = cps.ImageNameProvider(
                "Output image name:",
                # The second parameter holds a suggested name for the image.
                "OutputImage",
                doc="""This is the image resulting from the operation.""")
        #
        # Here's a choice box - the user gets a drop-down list of what
        # can be done.
        #
        self.gradient_choice = cps.Choice(
                "Gradient choice:",
                # The choice takes a list of possibilities. The first one
                # is the default - the one the user will typically choose.
                [GRADIENT_MAGNITUDE, GRADIENT_DIRECTION_X, GRADIENT_DIRECTION_Y],
                #
                # Here, in the documentation, we do a little trick so that
                # we use the actual text that's displayed in the documentation.
                #
                # %(GRADIENT_MAGNITUDE)s will get changed into "Gradient magnitude"
                # etc. Python will look in globals() for the "GRADIENT_" names
                # and paste them in where it sees %(GRADIENT_...)s
                #
                # The <ul> and <li> tags make a neat bullet-point list in the docs
                #
                doc="""Choose what to calculate:
            <ul>
            <li><i>%(GRADIENT_MAGNITUDE)s</i> to calculate the
            magnitude of the gradient at each pixel.</li>
            <li><i>%(GRADIENT_DIRECTION_X)s</i> to get the relative contribution
            of the gradient in the X direction (.5 = no contribution,
            0 to .5 = decreasing with increasing X, .5 to 1 = increasing
            with increasing X).</li>
            <li><i>%(GRADIENT_DIRECTION_Y)s</i> to get the relative
            contribution of the gradient in the Y direction.</li></ul>
            """ % globals()

        )
        #
        # A binary setting displays a checkbox.
        #
        self.automatic_smoothing = cps.Binary(
                "Automatically choose the smoothing scale?",
                # The default value is to choose automatically
                True,
                doc="""The module will automatically choose a
            smoothing scale for you if you leave this checked.""")
        #
        # We do a little smoothing which supplies a scale to the gradient.
        #
        # We use a float setting so that the user can give us a number
        # for the scale. The control will turn red if the user types in
        # an invalid scale.
        #
        self.scale = cps.Float(
                "Scale:",
                # The default value is 1 - a short-range scale
                1,
                # We don't let the user type in really small values
                minval=.1,
                # or large values
                maxval=100,
                doc="""This is a scaling factor that supplies the sigma for
            a gaussian that's used to smooth the image. The gradient is
            calculated on the smoothed image, so large scales will give
            you long-range gradients and small scales will give you
            short-range gradients""")

    #
    # The "settings" method tells CellProfiler about the settings you
    # have in your module. CellProfiler uses the list for saving
    # and restoring values for your module when it saves or loads a
    # pipeline file.
    #
    def settings(self):
        return [self.input_image_name, self.output_image_name,
                self.gradient_choice, self.automatic_smoothing,
                self.scale]

    #
    # visible_settings tells CellProfiler which settings should be
    # displayed and in what order.
    #
    # You don't have to implement "visible_settings" - if you delete
    # visible_settings, CellProfiler will use "settings" to pick settings
    # for display.
    #
    def visible_settings(self):
        result = [self.input_image_name, self.output_image_name,
                  self.gradient_choice, self.automatic_smoothing]
        #
        # Show the user the scale only if self.wants_smoothing is checked
        #
        if not self.automatic_smoothing:
            result += [self.scale]
        return result

    #
    # CellProfiler calls "run" on each image set in your pipeline.
    # This is where you do the real work.
    #
    def run(self, workspace):
        #
        # Get the input and output image names. You need to get the .value
        # because otherwise you'll get the setting object instead of
        # the string name.
        #
        input_image_name = self.input_image_name.value
        output_image_name = self.output_image_name.value
        #
        # Get the image set. The image set has all of the images in it.
        #
        image_set = workspace.image_set
        #
        # Get the input image object. We want a grayscale image here.
        # The image set will convert a color image to a grayscale one
        # and warn the user.
        #
        input_image = image_set.get_image(input_image_name,
                                          must_be_grayscale=True)
        #
        # Get the pixels - these are a 2-d Numpy array.
        #
        pixels = input_image.pixel_data
        #
        # Get the smoothing parameter
        #
        if self.automatic_smoothing:
            # Pick the mode of the power spectrum - obviously this
            # is pretty hokey, not intended to really find a good number.
            #
            fft = np.fft.fft2(pixels)
            power2 = np.sqrt((fft * fft.conjugate()).real)
            mode = np.argwhere(power2 == power2.max())[0]
            scale = np.sqrt(np.sum((mode + .5) ** 2))
        else:
            scale = self.scale.value
        g = gaussian_gradient_magnitude(pixels, scale)
        if self.gradient_choice == GRADIENT_MAGNITUDE:
            output_pixels = g
        else:
            # Numpy uses i and j instead of x and y. The x axis is 1
            # and the y axis is 0
            x = correlate1d(g, [-1, 0, 1], 1)
            y = correlate1d(g, [-1, 0, 1], 0)
            norm = np.sqrt(x ** 2 + y ** 2)
            if self.gradient_choice == GRADIENT_DIRECTION_X:
                output_pixels = .5 + x / norm / 2
            else:
                output_pixels = .5 + y / norm / 2
        #
        # Make an image object. It's nice if you tell CellProfiler
        # about the parent image - the child inherits the parent's
        # cropping and masking, but it's not absolutely necessary
        #
        output_image = cpi.Image(output_pixels, parent_image=input_image)
        image_set.add(output_image_name, output_image)
        #
        # Save intermediate results for display if the window frame is on
        #
        if self.show_window:
            workspace.display_data.input_pixels = pixels
            workspace.display_data.gradient = g
            workspace.display_data.output_pixels = output_pixels

    #
    # display lets you use matplotlib to display your results.
    #
    def display(self, workspace, figure):
        #
        # the "figure" is really the frame around the figure. You almost always
        # use figure.subplot or figure.subplot_imshow to get axes to draw on
        # so we pretty much ignore the figure.
        #
        figure = workspace.create_or_find_figure(subplots=(3, 1))
        #
        # Show the user the input image
        #
        figure.subplot_imshow_grayscale(
                0, 0,  # show the image in the first row and column
                workspace.display_data.input_pixels,
                title=self.input_image_name.value)
        lead_subplot = figure.subplot(0, 0)
        #
        # Show the user the gradient image, linking it to the first
        # so that they zoom and pan together
        #
        figure.subplot_imshow_grayscale(
                1, 0,  # show the image in the first row and second column
                workspace.display_data.gradient,
                title="Gradient",
                sharex=lead_subplot, sharey=lead_subplot)
        #
        # Show the user the final image
        #
        figure.subplot_imshow_grayscale(
                2, 0,  # show the image in the first row and last column
                workspace.display_data.output_pixels,
                title=self.output_image_name.value,
                sharex=lead_subplot, sharey=lead_subplot)
