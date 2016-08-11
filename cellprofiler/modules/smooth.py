"""<b>Smooth</b> smooths (i.e., blurs) images.
<hr>
This module allows you to smooth (blur) images, which can be helpful to
remove artifacts of a particular size.
Note that smoothing can be a time-consuming process.
"""

import cellprofiler.gui.help
import cellprofiler.image
import cellprofiler.module
import cellprofiler.modules
import cellprofiler.setting
import cellprofiler.setting
import centrosome.filter
import centrosome.smooth
import centrosome.smooth
import numpy
import scipy.ndimage

FIT_POLYNOMIAL = 'Fit Polynomial'
MEDIAN_FILTER = 'Median Filter'
GAUSSIAN_FILTER = 'Gaussian Filter'
SMOOTH_KEEPING_EDGES = 'Smooth Keeping Edges'
CIRCULAR_AVERAGE_FILTER = 'Circular Average Filter'
SM_TO_AVERAGE = "Smooth to Average"


class Smooth(cellprofiler.module.Module):
    module_name = 'Smooth'
    category = "Image Processing"
    variable_revision_number = 2

    def create_settings(self):
        self.image_name = cellprofiler.setting.ImageNameSubscriber('Select the input image', cellprofiler.setting.NONE)

        self.filtered_image_name = cellprofiler.setting.ImageNameProvider('Name the output image', 'FilteredImage')

        self.smoothing_method = cellprofiler.setting.Choice(
                'Select smoothing method',
                [FIT_POLYNOMIAL, GAUSSIAN_FILTER, MEDIAN_FILTER, SMOOTH_KEEPING_EDGES, CIRCULAR_AVERAGE_FILTER,
                 SM_TO_AVERAGE], doc="""
            This module smooths images using one of several filters.
            Fitting a polynomial
            is fastest but does not allow a very tight fit compared to the other methods:
            <ul>
            <li><i>%(FIT_POLYNOMIAL)s:</i> This method treats the intensity of the image pixels
            as a polynomial function of the x and y position of
            each pixel. It fits the intensity to the polynomial,
            <i>A x<sup>2</sup> + B y<sup>2</sup> + C xy + D x + E y + F</i>.
            This will produce a smoothed image with a single peak or trough of intensity
            that tapers off elsewhere in the image. For many microscopy images (where
            the illumination of the lamp is brightest in the center of field of view),
            this method will produce an image with a bright central region and dimmer
            edges. But, in some cases the peak/trough of the polynomial may actually
            occur outside of the image itself.</li>
            <li><i>%(GAUSSIAN_FILTER)s:</i> This method convolves the image with a Gaussian whose
            full width at half maximum is the artifact diameter entered.
            Its effect is to blur and obscure features
            smaller than the artifact diameter and spread bright or
            dim features larger than the artifact diameter.</li>
            <li><i>%(MEDIAN_FILTER)s:</i> This method finds the median pixel value within the
            artifact diameter you specify. It removes bright or dim features that are much smaller
            than the artifact diameter.</li>
            <li><i>%(SMOOTH_KEEPING_EDGES)s:</i> This method uses a bilateral filter which
            limits Gaussian smoothing across an edge while
            applying smoothing perpendicular to an edge. The effect
            is to respect edges in an image while smoothing other
            features. <i>%(SMOOTH_KEEPING_EDGES)s</i> will filter an image with reasonable
            speed for artifact diameters greater than 10 and for
            intensity differences greater than 0.1. The algorithm
            will consume more memory and operate more slowly as
            you lower these numbers.</li>
            <li><i>%(CIRCULAR_AVERAGE_FILTER)s:</i> This method convolves the image with
            a uniform circular averaging filter whose size is the artifact diameter entered. This filter is
            useful for re-creating an out-of-focus blur to an image.</li>
            <li><i>%(SM_TO_AVERAGE)s:</i> Creates a flat, smooth image where every pixel
            of the image equals the average value of the original image.</li>
            </ul>""" % globals())

        self.wants_automatic_object_size = cellprofiler.setting.Binary(
            'Calculate artifact diameter automatically?',
            True,
            doc="""
            <i>(Used only if "{gaussian_filter}", "{median_filter}", "{smooth_keeping_edges}" or "{circular_average_filter}" is selected)</i><br>
            Select <i>{yes}</i> to choose an artifact diameter based on
            the size of the image. The minimum size it will choose is 30 pixels,
            otherwise the size is 1/40 of the size of the image.
            <p>Select <i>{yes}</i> to manually enter an artifact diameter.</p>""".format(**{
                'gaussian_filter': GAUSSIAN_FILTER,
                'median_filter': MEDIAN_FILTER,
                'smooth_keeping_edges': SMOOTH_KEEPING_EDGES,
                'circular_average_filter': CIRCULAR_AVERAGE_FILTER,
                'yes': cellprofiler.setting.YES
            })
        )

        self.object_size = cellprofiler.setting.Float(
            'Typical artifact diameter',
            16.0,
            doc="""
            <i>(Used only if choosing the artifact diameter automatically is set to "{no}")</i><br>
            Enter the approximate diameter (in pixels) of the features to be blurred by
            the smoothing algorithm. This value is used to calculate the size of
            the spatial filter. {help_on_measuring_distances}
            For most smoothing methods, selecting a
            diameter over ~50 will take substantial amounts of time to process.""".format(**{
                'no': cellprofiler.setting.NO,
                'help_on_measuring_distances': cellprofiler.gui.help.HELP_ON_MEASURING_DISTANCES
            })
        )

        self.sigma_range = cellprofiler.setting.Float(
            'Edge intensity difference',
            0.1,
            doc="""
            <i>(Used only if "{smooth_keeping_edges}" is selected)</i><br>
            Enter the intensity step (which indicates an edge in an image) that you want to preserve.
            Edges are locations where the intensity changes precipitously, so this
            setting is used to adjust the rough magnitude of these changes. A lower
            number will preserve weaker edges. A higher number will preserve only stronger edges.
            Values should be between zero and one. {help_on_pixel_intensities}""".format(**{
                'smooth_keeping_edges': SMOOTH_KEEPING_EDGES,
                'help_on_pixel_intensities': cellprofiler.gui.help.HELP_ON_PIXEL_INTENSITIES
            })
        )

        self.clip = cellprofiler.setting.Binary(
            'Clip intensities to 0 and 1?',
            True,
            doc="""
            <i>(Used only if {fit_polynomial} is selected)</i><br>
            The <i>{fit_polynomial}</i> method is the only smoothing option that can yield
            an output image whose values are outside of the values of the
            input image. This setting controls whether to limit the image
            intensity to the 0 - 1 range used by CellProfiler.
            <p>Select <i>{yes}</i> to set all output image pixels less than zero to zero
            and all pixels greater than one to one. </p>
            <p>Select <i>{no}</i> to
            allow values less than zero and greater than one in the output
            image.</p>""".format(**{
                'fit_polynomial': FIT_POLYNOMIAL,
                'yes': cellprofiler.setting.YES,
                'no': cellprofiler.setting.NO
            })
        )

    def settings(self):
        return [self.image_name, self.filtered_image_name,
                self.smoothing_method, self.wants_automatic_object_size,
                self.object_size, self.sigma_range, self.clip]

    def visible_settings(self):
        result = [self.image_name, self.filtered_image_name,
                  self.smoothing_method]
        if self.smoothing_method.value not in [FIT_POLYNOMIAL, SM_TO_AVERAGE]:
            result.append(self.wants_automatic_object_size)
            if not self.wants_automatic_object_size.value:
                result.append(self.object_size)
            if self.smoothing_method.value == SMOOTH_KEEPING_EDGES:
                result.append(self.sigma_range)
        if self.smoothing_method.value == FIT_POLYNOMIAL:
            result.append(self.clip)
        return result

    def run(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_grayscale=True)
        pixel_data = image.pixel_data
        if self.wants_automatic_object_size.value:
            object_size = min(30, max(1, numpy.mean(pixel_data.shape) / 40))
        else:
            object_size = float(self.object_size.value)
        sigma = object_size / 2.35
        if self.smoothing_method.value == GAUSSIAN_FILTER:
            def fn(image):
                return scipy.ndimage.gaussian_filter(image, sigma,
                                                     mode='constant', cval=0)

            output_pixels = centrosome.smooth.smooth_with_function_and_mask(pixel_data, fn,
                                                                            image.mask)
        elif self.smoothing_method.value == MEDIAN_FILTER:
            output_pixels = centrosome.filter.median_filter(pixel_data, image.mask,
                                                            object_size / 2 + 1)
        elif self.smoothing_method.value == SMOOTH_KEEPING_EDGES:
            sigma_range = float(self.sigma_range.value)
            output_pixels = centrosome.filter.bilateral_filter(pixel_data, image.mask,
                                                               sigma, sigma_range)
        elif self.smoothing_method.value == FIT_POLYNOMIAL:
            output_pixels = centrosome.smooth.fit_polynomial(pixel_data, image.mask,
                                                             self.clip.value)
        elif self.smoothing_method.value == CIRCULAR_AVERAGE_FILTER:
            output_pixels = centrosome.filter.circular_average_filter(pixel_data, object_size / 2 + 1, image.mask)
        elif self.smoothing_method.value == SM_TO_AVERAGE:
            if image.has_mask:
                mean = numpy.mean(pixel_data[image.mask])
            else:
                mean = numpy.mean(pixel_data)
            output_pixels = numpy.ones(pixel_data.shape, pixel_data.dtype) * mean
        else:
            raise ValueError("Unsupported smoothing method: %s" %
                             self.smoothing_method.value)
        output_image = cellprofiler.image.Image(output_pixels, parent=image)
        workspace.image_set.add(self.filtered_image_name.value,
                                output_image)
        workspace.display_data.pixel_data = pixel_data
        workspace.display_data.output_pixels = output_pixels

    def display(self, workspace, figure):
        figure.set_subplots((2, 1))
        figure.subplot_imshow_grayscale(0, 0,
                                        workspace.display_data.pixel_data,
                                        "Original: %s" %
                                        self.image_name.value)
        figure.subplot_imshow_grayscale(1, 0,
                                        workspace.display_data.output_pixels,
                                        "Filtered: %s" %
                                        self.filtered_image_name.value,
                                        sharexy=figure.subplot(0, 0))

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if (module_name == 'SmoothKeepingEdges' and from_matlab and
                    variable_revision_number == 1):
            image_name, smoothed_image_name, spatial_radius, \
            intensity_radius = setting_values
            setting_values = [image_name,
                              smoothed_image_name,
                              'Smooth Keeping Edges',
                              'Automatic',
                              cellprofiler.setting.DO_NOT_USE,
                              cellprofiler.setting.NO,
                              spatial_radius,
                              intensity_radius]
            module_name = 'SmoothOrEnhance'
            variable_revision_number = 5
        if (module_name == 'SmoothOrEnhance' and from_matlab and
                    variable_revision_number == 4):
            # Added spatial radius
            setting_values = setting_values + ["0.1"]
            variable_revision_number = 5
        if (module_name == 'SmoothOrEnhance' and from_matlab and
                    variable_revision_number == 5):
            if setting_values[2] in ('Remove BrightRoundSpeckles',
                                     'Enhance BrightRoundSpeckles (Tophat Filter)'):
                raise ValueError(
                        'The Smooth module does not support speckles operations. Please use EnhanceOrSuppressFeatures with the Speckles feature type instead')
            setting_values = [setting_values[0],  # image name
                              setting_values[1],  # result name
                              setting_values[2],  # smoothing method
                              cellprofiler.setting.YES if setting_values[3] == 'Automatic'
                              else cellprofiler.setting.NO,  # wants smoothing
                              '16.0' if setting_values[3] == 'Automatic'
                              else (setting_values[6]
                                    if setting_values[2] == SMOOTH_KEEPING_EDGES
                                    else setting_values[3]),
                              setting_values[7]]
            module_name = 'Smooth'
            from_matlab = False
            variable_revision_number = 1
        if variable_revision_number == 1 and not from_matlab:
            setting_values = setting_values + [cellprofiler.setting.YES]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab
