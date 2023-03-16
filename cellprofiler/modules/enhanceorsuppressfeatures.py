"""
EnhanceOrSuppressFeatures
=========================

**EnhanceOrSuppressFeatures** enhances or suppresses certain image
features (such as speckles, ring shapes, and neurites), which can
improve subsequent identification of objects.

This module enhances or suppresses the intensity of certain pixels
relative to the rest of the image, by applying image processing filters
to the image. It produces a grayscale image in which objects can be
identified using an **Identify** module.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============
"""

import centrosome.filter
import numpy
import scipy.ndimage
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.transform
from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.range import IntegerRange
from cellprofiler_core.setting.text import Float
from cellprofiler_core.setting.text import Integer

from cellprofiler.modules import _help

ENHANCE = "Enhance"
SUPPRESS = "Suppress"

E_SPECKLES = "Speckles"
E_NEURITES = "Neurites"
E_DARK_HOLES = "Dark holes"
E_CIRCLES = "Circles"
E_TEXTURE = "Texture"
E_DIC = "DIC"

S_FAST = "Fast"
S_SLOW = "Slow"

N_GRADIENT = "Line structures"
N_TUBENESS = "Tubeness"


class EnhanceOrSuppressFeatures(ImageProcessing):
    module_name = "EnhanceOrSuppressFeatures"

    variable_revision_number = 7

    def create_settings(self):
        super(EnhanceOrSuppressFeatures, self).create_settings()

        self.method = Choice(
            "Select the operation",
            [ENHANCE, SUPPRESS],
            doc="""\
Select whether you want to enhance or suppress the features you
designate.

-  *{ENHANCE}:* Produce an image whose intensity is largely composed
   of the features of interest.
-  *{SUPPRESS}:* Produce an image with the features largely removed.
""".format(
                **{"ENHANCE": ENHANCE, "SUPPRESS": SUPPRESS}
            ),
        )

        self.enhance_method = Choice(
            "Feature type",
            [E_SPECKLES, E_NEURITES, E_DARK_HOLES, E_CIRCLES, E_TEXTURE, E_DIC],
            doc="""\
*(Used only if "{ENHANCE}" is selected)*

This module can enhance several kinds of image features:

-  *{E_SPECKLES}:* A speckle is an area of enhanced intensity
   relative to its immediate neighborhood. The module enhances speckles
   using a white tophat filter, which is the image minus the
   morphological grayscale opening of the image. The opening operation
   first suppresses the speckles by applying a grayscale erosion to
   reduce everything within a given radius to the lowest value within
   that radius, then uses a grayscale dilation to restore objects larger
   than the radius to an approximation of their former shape. The white
   tophat filter enhances speckles by subtracting the effects of opening
   from the original image.
-  *{E_NEURITES}:* Neurites are taken to be long, thin features of
   enhanced intensity. Choose this option to enhance the intensity of
   the neurites using the {N_GRADIENT} or {N_TUBENESS} methods
   described in a later setting.
-  *{E_DARK_HOLES}:* The module uses morphological reconstruction
   (the rolling-ball algorithm) to identify dark holes within brighter
   areas, or brighter ring shapes. The image is inverted so that the
   dark holes turn into bright peaks. The image is successively eroded
   and the eroded image is reconstructed at each step, resulting in an
   image that is missing the peaks. Finally, the reconstructed image is
   subtracted from the previous reconstructed image. This leaves
   circular bright spots with a radius equal to the number of iterations
   performed.
-  *{E_CIRCLES}:* The module calculates the circular Hough transform
   of the image at the diameter given by the feature size. The Hough
   transform will have the highest intensity at points that are centered
   within a ring of high intensity pixels where the ring diameter is the
   feature size. You may want to use the **EnhanceEdges** module to find
   the edges of your circular object and then process the output by
   enhancing circles. You can use **IdentifyPrimaryObjects** to find the
   circle centers and then use these centers as seeds in
   **IdentifySecondaryObjects** to find whole, circular objects using a
   watershed.
-  *{E_TEXTURE}:* This option produces an image
   whose intensity is the variance among nearby pixels. The method
   weights pixel contributions by distance using a Gaussian to calculate
   the weighting. You can use this method to separate foreground from
   background if the foreground is textured and the background is not.
-  *{E_DIC}:* This method recovers the optical density of a DIC image
   by integrating in a direction perpendicular to the shear direction of
   the image.

""".format(
                **{
                    "E_CIRCLES": E_CIRCLES,
                    "E_DARK_HOLES": E_DARK_HOLES,
                    "E_DIC": E_DIC,
                    "N_GRADIENT": N_GRADIENT,
                    "E_NEURITES": E_NEURITES,
                    "E_SPECKLES": E_SPECKLES,
                    "E_TEXTURE": E_TEXTURE,
                    "ENHANCE": ENHANCE,
                    "N_TUBENESS": N_TUBENESS,
                }
            ),
        )

        self.object_size = Integer(
            "Feature size",
            10,
            2,
            doc="""\
*(Used only if “{E_CIRCLES}”, “{E_SPECKLES}” or “{E_NEURITES}” are
selected, or if suppressing features)*

Enter the diameter of the largest speckle, the width of the circle, or
the width of the neurites to be enhanced or suppressed, which will be
used to calculate an appropriate filter size.

{HELP_ON_MEASURING_DISTANCES}
""".format(
                **{
                    "E_CIRCLES": E_CIRCLES,
                    "E_NEURITES": E_NEURITES,
                    "E_SPECKLES": E_SPECKLES,
                    "HELP_ON_MEASURING_DISTANCES": _help.HELP_ON_MEASURING_DISTANCES,
                }
            ),
        )

        self.hole_size = IntegerRange(
            "Range of hole sizes",
            value=(1, 10),
            minval=1,
            doc="""\
*(Used only if "{E_DARK_HOLES}" is selected)*

The range of hole sizes to be enhanced. The algorithm will identify only
holes whose diameters fall between these two values.
""".format(
                **{"E_DARK_HOLES": E_DARK_HOLES}
            ),
        )

        self.smoothing = Float(
            "Smoothing scale",
            value=2.0,
            minval=0.0,
            doc="""\
*(Used only for the "{E_TEXTURE}", "{E_DIC}" or "{E_NEURITES}" methods)*

-  *{E_TEXTURE}*: This is roughly the scale of the texture features, in
   pixels. The algorithm uses the smoothing value entered as the sigma
   of the Gaussian used to weight nearby pixels by distance in the
   variance calculation.
-  *{E_DIC}:* Specifies the amount of smoothing of the image in the
   direction parallel to the shear axis of the image. The line
   integration method will leave streaks in the image without smoothing
   as it encounters noisy pixels during the course of the integration.
   The smoothing takes contributions from nearby pixels, which decreases
   the noise but smooths the resulting image. Increase the smoothing to eliminate streakiness and
   decrease the smoothing to sharpen the image.
-  *{E_NEURITES}:* The *{N_TUBENESS}* option uses this scale as the
   sigma of the Gaussian used to smooth the image prior to gradient
   detection.

|image0| Smoothing can be turned off by entering a value of zero, but
this is not recommended.

.. |image0| image:: {PROTIP_AVOID_ICON}
""".format(
                **{
                    "E_DIC": E_DIC,
                    "E_NEURITES": E_NEURITES,
                    "E_TEXTURE": E_TEXTURE,
                    "N_TUBENESS": N_TUBENESS,
                    "PROTIP_AVOID_ICON": _help.PROTIP_AVOID_ICON,
                }
            ),
        )

        self.angle = Float(
            "Shear angle",
            value=0,
            doc="""\
*(Used only for the "{E_DIC}" method)*

The shear angle is the direction of constant value for the shadows and
highlights in a DIC image. The gradients in a DIC image run in the
direction perpendicular to the shear angle. For example, if the shadows
run diagonally from lower left to upper right and the highlights appear
above the shadows, the shear angle is 45°. If the shadows appear on top,
the shear angle is 180° + 45° = 225°.
""".format(
                **{"E_DIC": E_DIC}
            ),
        )

        self.decay = Float(
            "Decay",
            value=0.95,
            minval=0.1,
            maxval=1,
            doc="""\
*(Used only for the "{E_DIC}" method)*

The decay setting applies an exponential decay during the process of
integration by multiplying the accumulated sum by the decay at each
step. This lets the integration recover from accumulated error during
the course of the integration, but it also results in diminished
intensities in the middle of large objects. Set the decay to a large
value, on the order of 1 - 1/diameter of your objects if the intensities
decrease toward the middle. Set the decay to a small value if there
appears to be a bias in the integration direction.
""".format(
                **{"E_DIC": E_DIC}
            ),
        )

        self.neurite_choice = Choice(
            "Enhancement method",
            [N_TUBENESS, N_GRADIENT],
            doc="""\
*(Used only for the "{E_NEURITES}" method)*

Two methods can be used to enhance neurites:

-  *{N_TUBENESS}*: This method is an adaptation of the method used by
   the `ImageJ Tubeness plugin`_. The image is smoothed with a Gaussian.
   The Hessian is then computed at every point to measure the intensity
   gradient and the eigenvalues of the Hessian are computed to determine
   the magnitude of the intensity. The absolute maximum of the two
   eigenvalues gives a measure of the ratio of the intensity of the
   gradient in the direction of its most rapid descent versus in the
   orthogonal direction. The output image is the absolute magnitude of
   the highest eigenvalue if that eigenvalue is negative (white neurite
   on dark background), otherwise, zero.
-  *{N_GRADIENT}*: The module takes the difference of the white and
   black tophat filters (a white tophat filtering is the image minus the
   morphological grayscale opening of the image; a black tophat
   filtering is the morphological grayscale closing of the image minus
   the image). The effect is to enhance lines whose width is the
   feature size.

.. _ImageJ Tubeness plugin: http://www.longair.net/edinburgh/imagej/tubeness/
""".format(
                **{
                    "E_NEURITES": E_NEURITES,
                    "N_GRADIENT": N_GRADIENT,
                    "N_TUBENESS": N_TUBENESS,
                }
            ),
        )

        self.speckle_accuracy = Choice(
            "Speed and accuracy",
            choices=[S_FAST, S_SLOW],
            doc="""\
*(Used only for the "{E_SPECKLES}" method)*

*{E_SPECKLES}* can use a fast or slow algorithm to find speckles.

-  *{S_FAST}:* Select this option for speckles that have a large radius
   (greater than 10 pixels) and need not be exactly circular.
-  *{S_SLOW}:* Use for speckles of small radius.
""".format(
                **{"E_SPECKLES": E_SPECKLES, "S_FAST": S_FAST, "S_SLOW": S_SLOW}
            ),
        )

        self.wants_rescale = Binary(
            "Rescale result image",
            False,
            doc="""\
*(Used only for the "{E_NEURITES}" method)*

*{E_NEURITES}* can rescale the resulting values to use the 
whole intensity range of the image (0-1). This can make 
the output easier to display.
""".format(
                **{"E_NEURITES": E_NEURITES}
            ),
        )

    def settings(self):
        __settings__ = super(EnhanceOrSuppressFeatures, self).settings()
        return __settings__ + [
            self.method,
            self.object_size,
            self.enhance_method,
            self.hole_size,
            self.smoothing,
            self.angle,
            self.decay,
            self.neurite_choice,
            self.speckle_accuracy,
            self.wants_rescale,
        ]

    def visible_settings(self):
        __settings__ = super(EnhanceOrSuppressFeatures, self).visible_settings()
        __settings__ += [self.method]
        if self.method == ENHANCE:
            __settings__ += [self.enhance_method]
            self.object_size.min_value = 2
            if self.enhance_method == E_DARK_HOLES:
                __settings__ += [self.hole_size]
            elif self.enhance_method == E_TEXTURE:
                __settings__ += [self.smoothing]
            elif self.enhance_method == E_DIC:
                __settings__ += [self.smoothing, self.angle, self.decay]
            elif self.enhance_method == E_NEURITES:
                __settings__ += [self.neurite_choice]
                if self.neurite_choice == N_GRADIENT:
                    __settings__ += [self.object_size]
                else:
                    __settings__ += [self.smoothing]
                __settings__ += [self.wants_rescale]
            elif self.enhance_method == E_SPECKLES:
                __settings__ += [self.object_size, self.speckle_accuracy]
                self.object_size.min_value = 3
            else:
                __settings__ += [self.object_size]
        else:
            __settings__ += [self.object_size]
        return __settings__

    def run(self, workspace):
        image = workspace.image_set.get_image(self.x_name.value, must_be_grayscale=True)

        radius = self.object_size.value / 2

        if self.method == ENHANCE:
            if self.enhance_method == E_SPECKLES:
                result = self.enhance_speckles(
                    image, radius, self.speckle_accuracy.value
                )
            elif self.enhance_method == E_NEURITES:
                result = self.enhance_neurites(image, radius, self.neurite_choice.value)
                if self.wants_rescale.value:
                    result = skimage.exposure.rescale_intensity(result)
            elif self.enhance_method == E_DARK_HOLES:
                min_radius = max(1, int(self.hole_size.min / 2))

                max_radius = int((self.hole_size.max + 1) / 2)

                result = self.enhance_dark_holes(image, min_radius, max_radius)
            elif self.enhance_method == E_CIRCLES:
                result = self.enhance_circles(image, radius)
            elif self.enhance_method == E_TEXTURE:
                result = self.enhance_texture(image, self.smoothing.value)
            elif self.enhance_method == E_DIC:
                result = self.enhance_dic(
                    image, self.angle.value, self.decay.value, self.smoothing.value
                )
            else:
                raise NotImplementedError(
                    "Unimplemented enhance method: %s" % self.enhance_method.value
                )
        elif self.method == SUPPRESS:
            result = self.suppress(image, radius)
        else:
            raise ValueError("Unknown filtering method: %s" % self.method)

        result_image = Image(result, parent_image=image, dimensions=image.dimensions)

        workspace.image_set.add(self.y_name.value, result_image)

        if self.show_window:
            workspace.display_data.x_data = image.pixel_data

            workspace.display_data.y_data = result

            workspace.display_data.dimensions = image.dimensions

    def __mask(self, pixel_data, mask):
        data = numpy.zeros_like(pixel_data)

        data[mask] = pixel_data[mask]

        return data

    def __unmask(self, data, pixel_data, mask):
        data[~mask] = pixel_data[~mask]

        return data

    def __structuring_element(self, radius, volumetric):
        if volumetric:
            return skimage.morphology.ball(radius)

        return skimage.morphology.disk(radius)

    def enhance_speckles(self, image, radius, accuracy):
        data = self.__mask(image.pixel_data, image.mask)

        footprint = self.__structuring_element(radius, image.volumetric)

        if accuracy == "Slow" or radius <= 3:
            result = skimage.morphology.white_tophat(data, footprint=footprint)
        else:
            #
            # white_tophat = img - opening
            #              = img - dilate(erode)
            #              = img - maximum_filter(minimum_filter)
            minimum = scipy.ndimage.filters.minimum_filter(data, footprint=footprint)

            maximum = scipy.ndimage.filters.maximum_filter(minimum, footprint=footprint)

            result = data - maximum

        return self.__unmask(result, image.pixel_data, image.mask)

    def enhance_neurites(self, image, radius, method):
        data = self.__mask(image.pixel_data, image.mask)

        if method == N_GRADIENT:
            # desired effect = img + white_tophat - black_tophat
            footprint = self.__structuring_element(radius, image.volumetric)

            white = skimage.morphology.white_tophat(data, footprint=footprint)

            black = skimage.morphology.black_tophat(data, footprint=footprint)

            result = data + white - black

            result[result > 1] = 1

            result[result < 0] = 0
        else:
            sigma = self.smoothing.value

            smoothed = scipy.ndimage.gaussian_filter(
                data, numpy.divide(sigma, image.spacing)
            )

            if image.volumetric:
                result = numpy.zeros_like(smoothed)

                for index, plane in enumerate(smoothed):
                    hessian = centrosome.filter.hessian(
                        plane, return_hessian=False, return_eigenvectors=False
                    )

                    result[index] = (
                        -hessian[:, :, 0] * (hessian[:, :, 0] < 0) * (sigma ** 2)
                    )
            else:
                hessian = centrosome.filter.hessian(
                    smoothed, return_hessian=False, return_eigenvectors=False
                )

                #
                # The positive values are darker pixels with lighter
                # neighbors. The original ImageJ code scales the result
                # by sigma squared - I have a feeling this might be
                # a first-order correction for e**(-2*sigma), possibly
                # because the hessian is taken from one pixel away
                # and the gradient is less as sigma gets larger.
                result = -hessian[:, :, 0] * (hessian[:, :, 0] < 0) * (sigma ** 2)

        return self.__unmask(result, image.pixel_data, image.mask)

    def enhance_circles(self, image, radius):
        data = self.__mask(image.pixel_data, image.mask)

        if image.volumetric:
            result = numpy.zeros_like(data)

            for index, plane in enumerate(data):
                result[index] = skimage.transform.hough_circle(plane, radius)[0]
        else:
            result = skimage.transform.hough_circle(data, radius)[0]

        return self.__unmask(result, image.pixel_data, image.mask)

    def enhance_texture(self, image, sigma):
        mask = image.mask

        data = self.__mask(image.pixel_data, mask)

        gmask = skimage.filters.gaussian(
            mask.astype(float), sigma, mode="constant"
        )

        img_mean = (
            skimage.filters.gaussian(data, sigma, mode="constant")
            / gmask
        )

        img_squared = (
            skimage.filters.gaussian(
                data ** 2, sigma, mode="constant"
            )
            / gmask
        )

        result = img_squared - img_mean ** 2

        return self.__unmask(result, image.pixel_data, mask)

    def enhance_dark_holes(self, image, min_radius, max_radius):
        pixel_data = image.pixel_data

        mask = image.mask if image.has_mask else None

        se = self.__structuring_element(1, image.volumetric)

        inverted_image = pixel_data.max() - pixel_data

        previous_reconstructed_image = inverted_image

        eroded_image = inverted_image

        smoothed_image = numpy.zeros(pixel_data.shape)

        for i in range(max_radius + 1):
            eroded_image = skimage.morphology.erosion(eroded_image, se)

            if mask is not None:
                eroded_image *= mask

            reconstructed_image = skimage.morphology.reconstruction(
                eroded_image, inverted_image, "dilation", se
            )

            output_image = previous_reconstructed_image - reconstructed_image

            if i >= min_radius:
                smoothed_image = numpy.maximum(smoothed_image, output_image)

            previous_reconstructed_image = reconstructed_image

        return smoothed_image

    def enhance_dic(self, image, angle, decay, smoothing):
        pixel_data = image.pixel_data

        if image.volumetric:
            result = numpy.zeros_like(pixel_data)

            for index, plane in enumerate(pixel_data):
                result[index] = centrosome.filter.line_integration(
                    plane, angle, decay, smoothing
                )

            return result

        if smoothing == 0:
            smoothing = numpy.finfo(float).eps

        return centrosome.filter.line_integration(pixel_data, angle, decay, smoothing)

    def suppress(self, image, radius):
        data = self.__mask(image.pixel_data, image.mask)

        footprint = self.__structuring_element(radius, image.volumetric)

        result = skimage.morphology.opening(data, footprint)

        return self.__unmask(result, image.pixel_data, image.mask)

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust setting values if they came from a previous revision

        setting_values - a sequence of strings representing the settings
                         for the module as stored in the pipeline
        variable_revision_number - the variable revision number of the
                         module at the time the pipeline was saved. Use this
                         to determine how the incoming setting values map
                         to those of the current module version.
        module_name - the name of the module that did the saving. This can be
                      used to import the settings from another module if
                      that module was merged into the current module
        """
        if variable_revision_number == 1:
            #
            # V1 -> V2, added enhance method and hole size
            #
            setting_values = setting_values + [E_SPECKLES, "1,10"]
            variable_revision_number = 2
        if variable_revision_number == 2:
            #
            # V2 -> V3, added texture and DIC
            #
            setting_values = setting_values + ["2.0", "0", ".95"]
            variable_revision_number = 3
        if variable_revision_number == 3:
            setting_values = setting_values + [N_GRADIENT]
            variable_revision_number = 4
        if variable_revision_number == 4:
            setting_values = setting_values + ["Slow / circular"]
            variable_revision_number = 5

        if variable_revision_number == 5:
            if setting_values[-1] == "Slow / circular":
                setting_values[-1] = "Slow"
            else:
                setting_values[-1] = "Fast"

            variable_revision_number = 6

        if variable_revision_number == 6:
            # Add neurite rescaling option
            setting_values.append("Yes")
            variable_revision_number = 7

        return setting_values, variable_revision_number


EnhanceOrSuppressSpeckles = EnhanceOrSuppressFeatures
