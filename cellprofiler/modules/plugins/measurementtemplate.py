#################################
#
# Imports from useful Python libraries
#
#################################

import centrosome.cpmorphology
import centrosome.zernike
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
MeasurementTemplate
===================

**MeasurementTemplate** - an example measurement module. It's recommended to
put a brief description of this module here and go into more detail below.

This is an example of a module that measures a property of an image both
for the image as a whole and for every object in the image. It
demonstrates how to load an image, how to load an object and how to
record a measurement. 

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

Describe the measurements made by this module. Typically, measurements
are described in the following format:

**Measurement category:**

-  *MeasurementName*: A brief description of the measurement.
-  *MeasurementName*: A brief description of the measurement.

**Measurement category:**

-  *MeasurementName*: A brief description of the measurement.
-  *MeasurementName*: A brief description of the measurement.

This module makes the following measurements:

**MT** (the MeasurementTemplate category):

-  *Intensity_[IMAGE_NAME]_N[Ni]_M[Mj]*: the Zernike feature of the
   IMAGE_NAME image with radial degree Ni and Azimuthal degree Mj,
   Mj >= 0.
-  *Intensity_[IMAGE_NAME]_N[Ni]_MM[Mj]*: the Zernike feature of
   the IMAGE_NAME image with radial degree Ni and Azimuthal degree
   Mj, Mj < 0.

Technical notes
^^^^^^^^^^^^^^^

Include implementation details or notes here. Additionally provide any 
other background information about this module, including definitions
or adopted conventions. Information which may be too specific to fit into
the general description should be provided here.

Omit this section if there is no technical information to mention.

The Zernike features measured here are themselves interesting. You can 
reconstruct the image of a cell, approximately, by constructing the Zernike 
functions on a unit circle, multiplying the real parts by the corresponding 
features for positive M, multiplying the imaginary parts by the corresponding 
features for negative M and adding real and imaginary parts.

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
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.setting.subscriber import ImageSubscriber, LabelSubscriber
from cellprofiler_core.setting.text import Integer

"""This is the measurement template category"""
C_MEASUREMENT_TEMPLATE = "MT"


#
# The module class
#
# Your module should "inherit" from cellprofiler_core.module.Module.
# This means that your module will use the methods from Module unless
# you re-implement them. You can let Module do most of the work and
# implement only what you need.
#
class MeasurementTemplate(Module):
    #
    # The module starts by declaring the name that's used for display,
    # the category under which it is stored and the variable revision
    # number which can be used to provide backwards compatibility if
    # you add user-interface functionality later.
    #
    module_name = "MeasurementTemplate"
    category = "Measurement"
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
        # The ImageNameSubscriber "subscribes" to all ImageNameProviders in
        # prior modules. Modules before yours will put images into CellProfiler.
        # The ImageSubscriber gives your user a list of these images
        # which can then be used as inputs in your module.
        #
        self.input_image_name = ImageSubscriber(
            # The text to the left of the edit box
            text="Input image name:",
            # reST help that gets displayed when the user presses the
            # help button to the right of the edit box.
            doc="""\
This is the image that the module operates on. You can choose any image
that is made available by a prior module.

**MeasurementTemplate** will measure something about this image.
""",
        )

        #
        # The ObjectNameSubscriber is similar to the ImageNameSubscriber.
        # It will ask the user which object to pick from the list of
        # objects provided by upstream modules.
        #
        self.input_object_name = LabelSubscriber(
            text="Input object name",
            doc="These are the objects that the module operates on.",
        )

        #
        # The radial degree is the "N" parameter in the Zernike - how many
        # inflection points there are, radiating out from the center. Higher
        # N means more features and a more detailed description
        #
        # The setting is an integer setting, bounded between 1 and 50.
        # N = 50 generates 1200 features!
        #
        self.radial_degree = Integer(
            text="Radial degree",
            value=10,
            minval=1,
            maxval=50,
            doc="""\
Calculate all Zernike features up to the given radial
degree. The Zernike function is parameterized by a radial
and azimuthal degree. The module will calculate all Zernike
features for all azimuthal degrees up to and including the
radial degree you enter here.
""",
        )

    #
    # The "settings" method tells CellProfiler about the settings you
    # have in your module. CellProfiler uses the list for saving
    # and restoring values for your module when it saves or loads a
    # pipeline file.
    #
    # This module does not have a "visible_settings" method. CellProfiler
    # will use "settings" to make the list of user-interface elements
    # that let the user configure the module. See imagetemplate.py for
    # a template for visible_settings that you can cut and paste here.
    #
    def settings(self):
        return [self.input_image_name, self.input_object_name, self.radial_degree]

    #
    # CellProfiler calls "run" on each image set in your pipeline.
    #
    def run(self, workspace):
        #
        # Get the measurements object - we put the measurements we
        # make in here
        #
        measurements = workspace.measurements

        #
        # We record some statistics which we will display later.
        # We format them so that Matplotlib can display them in a table.
        # The first row is a header that tells what the fields are.
        #
        statistics = [["Feature", "Mean", "Median", "SD"]]

        #
        # Put the statistics in the workspace display data so we
        # can get at them when we display
        #
        workspace.display_data.statistics = statistics

        #
        # Get the input image and object. You need to get the .value
        # because otherwise you'll get the setting object instead of
        # the string name.
        #
        input_image_name = self.input_image_name.value
        input_object_name = self.input_object_name.value

        ################################################################
        #
        # GETTING AN IMAGE FROM THE IMAGE SET
        #
        # Get the image set. The image set has all of the images in it.
        #
        image_set = workspace.image_set
        #
        # Get the input image object. We want a grayscale image here.
        # The image set will convert a color image to a grayscale one
        # and warn the user.
        #
        input_image = image_set.get_image(input_image_name, must_be_grayscale=True)
        #
        # Get the pixels - these are a 2-d Numpy array.
        #
        pixels = input_image.pixel_data
        #
        ###############################################################

        ###############################################################
        #
        # GETTING THE LABELS MATRIX FROM THE OBJECT SET
        #
        # The object set has all of the objects in it.
        #
        object_set = workspace.object_set
        #
        # Get objects from the object set. The most useful array in
        # the objects is "objects.segmented" which is a labels matrix
        # in which each pixel has an integer value.
        #
        # The value, "0", is reserved for "background" - a pixel with
        # a zero value is not in an object. Each object has an object
        # number, starting at "1" and each pixel in the object is
        # labeled with that number.
        #
        # The other useful array is "objects.small_removed_segmented" which
        # is another labels matrix. There are objects that touch the edge of
        # the image and get filtered out and there are large objects that
        # get filtered out. Modules like "IdentifySecondaryObjects" may
        # want to associate pixels near the objects in the labels matrix to
        # those objects - the large and touching objects should compete with
        # the real ones, so you should use "objects.small_removed_segmented"
        # for those cases.
        #
        objects = object_set.get_objects(input_object_name)
        labels = objects.segmented
        ###############################################################

        #
        # The minimum enclosing circle (MEC) is the smallest circle that
        # will fit around the object. We get the centers and radii of
        # all of the objects at once. You'll see how that lets us
        # compute the X and Y position of each pixel in a label all at
        # one go.
        #
        # First, get an array that lists the whole range of indexes in
        # the labels matrix.
        #
        indexes = objects.indices
        #
        # Then ask for the minimum_enclosing_circle for each object named
        # in those indexes. MEC returns the i and j coordinate of the center
        # and the radius of the circle and that defines the circle entirely.
        #
        centers, radius = centrosome.cpmorphology.minimum_enclosing_circle(
            labels, indexes
        )
        #
        # The module computes a measurement based on the image intensity
        # inside an object times a Zernike polynomial inscribed in the
        # minimum enclosing circle around the object. The details are
        # in the "measure_zernike" function. We call into the function with
        # an N and M which describe the polynomial.
        #
        for n, m in self.get_zernike_indexes():
            # Compute the zernikes for each object, returned in an array
            zr, zi = self.measure_zernike(
                pixels, labels, indexes, centers, radius, n, m
            )

            # Get the name of the measurement feature for this zernike
            feature = self.get_measurement_name(n, m)

            # Add a measurement for this kind of object
            if m != 0:
                measurements.add_measurement(input_object_name, feature, zr)

                # Do the same with -m
                feature = self.get_measurement_name(n, -m)
                measurements.add_measurement(input_object_name, feature, zi)
            else:
                # For zero, the total is the sum of real and imaginary parts
                measurements.add_measurement(input_object_name, feature, zr + zi)

            # Record the statistics.
            zmean = numpy.mean(zr)
            zmedian = numpy.median(zr)
            zsd = numpy.std(zr)
            statistics.append([feature, zmean, zmedian, zsd])

    #
    # "display" lets you use matplotlib to display your results.
    #
    def display(self, workspace, figure):
        statistics = workspace.display_data.statistics

        figure.set_subplots((1, 1))

        figure.subplot_table(0, 0, statistics)

    def get_zernike_indexes(self, wants_negative=False):
        """Get an N x 2 numpy array containing the M and N Zernike degrees

        Use the radial_degree setting to determine which Zernikes to do.

        wants_negative - if True, return both positive and negative M, if false
                         return only positive
        """
        zi = centrosome.zernike.get_zernike_indexes(self.radial_degree.value + 1)

        if wants_negative:
            #
            # numpy.vstack means concatenate rows of two 2d arrays.
            # The multiplication by [1, -1] negates every m, but preserves n.
            # zi[zi[:, 1] != 0] picks out only elements with m not equal to zero.
            #
            zi = numpy.vstack([zi, zi[zi[:, 1] != 0] * numpy.array([1, -1])])

            #
            # Sort by azimuth degree and radial degree so they are ordered
            # reasonably
            #
            order = numpy.lexsort((zi[:, 1], zi[:, 0]))
            zi = zi[order, :]

        return zi

    #
    # "measure_zernike" makes one Zernike measurement on each object
    #
    def measure_zernike(self, pixels, labels, indexes, centers, radius, n, m):
        """Measure the intensity of the image with Zernike (N, M)

        pixels - the intensity image to be measured
        labels - the labels matrix that labels each object with an integer
        indexes - the label #s in the image
        centers - the centers of the minimum enclosing circle for each object
        radius - the radius of the minimum enclosing circle for each object
        n, m - the Zernike coefficients.

        See http://en.wikipedia.org/wiki/Zernike_polynomials for an
        explanation of the Zernike polynomials
        """
        #
        # The strategy here is to operate on the whole array instead
        # of operating on one object at a time. The most important thing
        # is to avoid having to run the Python interpreter once per pixel
        # in the image and the second most important is to avoid running
        # it per object in case there are hundreds of objects.
        #
        # We play lots of indexing tricks here to operate on the whole image.
        # I'll try to explain some - hopefully, you can reuse.
        center_x = centers[:, 1]
        center_y = centers[:, 0]

        #
        # Make up fake values for 0 (the background). This lets us do our
        # indexing tricks. Really, we're going to ignore the background,
        # but we want to do the indexing without ignoring the background
        # because that's easier.
        #
        center_x = numpy.hstack([[0], center_x])
        center_y = numpy.hstack([[0], center_y])
        radius = numpy.hstack([[1], radius])

        #
        # Now get one array that's the y coordinate of each pixel and one
        # that's the x coordinate. This might look stupid and wasteful,
        # but these "arrays" are never actually realized and made into
        # real memory.
        #
        # Using 0.0:X creates floating point indices. This makes the
        # data type of x, y consistent with center_x, center_y and
        # raidus. Numpy requires consistent data types for in-place
        # operations like -= and /=.
        #
        y, x = numpy.mgrid[0.0 : labels.shape[0], 0.0 : labels.shape[1]]

        #
        # Get the x and y coordinates relative to the object centers.
        # This uses Numpy broadcasting. For each pixel, we use the
        # value in the labels matrix as an index into the appropriate
        # one-dimensional array. So we get the value for that object.
        #
        y -= center_y[labels]
        x -= center_x[labels]

        #
        # Zernikes take x and y values from zero to one. We scale the
        # integer coordinate values by dividing them by the radius of
        # the circle. Again, we use the indexing trick to look up the
        # values for each object.
        #
        y /= radius[labels]
        x /= radius[labels]

        #
        # Now we can get Zernike polynomials per-pixel where each pixel
        # value is calculated according to its object's MEC.
        #
        # We use a mask of all of the non-zero labels so the calculation
        # runs a little faster.
        #
        zernike_polynomial = centrosome.zernike.construct_zernike_polynomials(
            x, y, numpy.array([[n, m]]), labels > 0
        )

        #
        # For historical reasons, centrosome didn't multiply by the per/zernike
        # normalizing factor: 2*n + 2 / E / pi where E is 2 if m is zero and 1
        # if m is one. We do it here to aid with the reconstruction
        #
        zernike_polynomial *= (2 * n + 2) / (2 if m == 0 else 1) / numpy.pi

        #
        # Multiply the Zernike polynomial by the image to dissect
        # the image by the Zernike basis set.
        #
        output_pixels = pixels * zernike_polynomial[:, :, 0]

        #
        # The sum function calculates the sum of the pixel values for
        # each pixel in an object, using the labels matrix to name
        # the pixels in an object
        #
        zr = scipy.ndimage.sum(output_pixels.real, labels, indexes)
        zi = scipy.ndimage.sum(output_pixels.imag, labels, indexes)

        return zr, zi

    #
    # Here, we go about naming the measurements.
    #
    # Measurement names have parts to them, separated by underbars.
    # There's always a category and a feature name
    # and sometimes there are modifiers such as the image that
    # was measured or the scale at which it was measured.
    #
    # We have functions that build the names so that we can
    # use the same functions in different places.
    #
    def get_feature_name(self, n, m):
        """Return a measurement feature name for the given Zernike"""
        #
        # Something nice and simple for a name... Intensity_DNA_N4M2 for instance
        #
        if m >= 0:
            return "Intensity_%s_N%dM%d" % (self.input_image_name.value, n, m)

        return "Intensity_%s_N%dMM%d" % (self.input_image_name.value, n, -m)

    def get_measurement_name(self, n, m):
        """Return the whole measurement name"""
        input_image_name = self.input_image_name.value

        return "_".join([C_MEASUREMENT_TEMPLATE, self.get_feature_name(n, m)])

    #
    # We have to tell CellProfiler about the measurements we produce.
    # There are two parts: one that is for database-type modules and one
    # that is for the UI. The first part gives a comprehensive list
    # of measurement columns produced. The second is more informal and
    # tells CellProfiler how to categorize its measurements.
    #
    # "get_measurement_columns" gets the measurements for use in the database
    # or in a spreadsheet. Some modules need this because they
    # might make measurements of measurements and need those names.
    #
    def get_measurement_columns(self, pipeline):
        #
        # We use a list comprehension here.
        # See http://docs.python.org/tutorial/datastructures.html#list-comprehensions
        # for how this works.
        #
        # The first thing in the list is the object being measured. If it's
        # the whole image, use IMAGE as the name.
        #
        # The second thing is the measurement name.
        #
        # The third thing is the column type. See the COLTYPE constants
        # in measurement.py for what you can use
        #
        input_object_name = self.input_object_name.value

        return [
            (input_object_name, self.get_measurement_name(n, m), COLTYPE_FLOAT,)
            for n, m in self.get_zernike_indexes(True)
        ]

    #
    # "get_categories" returns a list of the measurement categories produced
    # by this module. It takes an object name - only return categories
    # if the name matches.
    #
    def get_categories(self, pipeline, object_name):
        if object_name == self.input_object_name:
            return [C_MEASUREMENT_TEMPLATE]

        return []

    #
    # Return the feature names if the object_name and category match
    #
    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.input_object_name and category == C_MEASUREMENT_TEMPLATE:
            return ["Intensity"]

        return []

    #
    # This module makes per-image measurements. That means we need
    # "get_measurement_images" to distinguish measurements made on two
    # different images by this module
    #
    def get_measurement_images(self, pipeline, object_name, category, measurement):
        #
        # This might seem wasteful, but UI code can be slow. Just see
        # if the measurement is in the list returned by get_measurements
        #
        if measurement in self.get_measurements(pipeline, object_name, category):
            return [self.input_image_name.value]

        return []

    def get_measurement_scales(
        self, pipeline, object_name, category, measurement, image_name
    ):
        """Get the scales for a measurement

        For the Zernikes, the scales are of the form, N2M2 or N2MM2 for
        negative azimuthal degree
        """

        def get_scale(n, m):
            if m >= 0:
                return "N%dM%d" % (n, m)

            return "N%dMM%d" % (n, -m)

        if image_name in self.get_measurement_images(
            pipeline, object_name, category, measurement
        ):
            return [get_scale(n, m) for n, m in self.get_zernike_indexes(True)]

        return []

    @staticmethod
    def get_image_from_features(radius, feature_dictionary):
        """Reconstruct the intensity image from the zernike features

        radius - the radius of the minimum enclosing circle

        feature_dictionary - keys are (n, m) tuples and values are the
        magnitudes.

        returns a greyscale image based on the feature dictionary.
        """
        i, j = (
            numpy.mgrid[-radius : (radius + 1), -radius : (radius + 1)].astype(float)
            / radius
        )
        mask = (i * i + j * j) <= 1

        zernike_indexes = numpy.array(list(feature_dictionary.keys()))
        zernike_features = numpy.array(list(feature_dictionary.values()))

        z = centrosome.zernike.construct_zernike_polynomials(
            j, i, numpy.abs(zernike_indexes), mask=mask
        )
        zn = (
            (2 * zernike_indexes[:, 0] + 2)
            / ((zernike_indexes[:, 1] == 0) + 1)
            / numpy.pi
        )
        z *= zn[numpy.newaxis, numpy.newaxis, :]
        z = (
            z.real * (zernike_indexes[:, 1] >= 0)[numpy.newaxis, numpy.newaxis, :]
            + z.imag * (zernike_indexes[:, 1] <= 0)[numpy.newaxis, numpy.newaxis, :]
        )

        return numpy.sum(z * zernike_features[numpy.newaxis, numpy.newaxis, :], 2)
