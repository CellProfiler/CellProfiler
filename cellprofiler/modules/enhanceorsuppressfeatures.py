'''<b>Enhance Or Suppress Features</b> enhances or suppresses certain image features 
(such as speckles, ring shapes, and neurites), which can improve subsequent 
identification of objects.
<hr>
This module enhances or suppresses the intensity of certain pixels relative
to the rest of the image, by applying image processing filters to the image. It 
produces a grayscale image in which objects can be identified using an <b>Identify</b> module.
'''

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2014 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org


import numpy as np
from scipy.ndimage import gaussian_filter

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
from cellprofiler.cpmath.cpmorphology import opening, closing, white_tophat
from cellprofiler.cpmath.filter import enhance_dark_holes, circular_hough
from cellprofiler.cpmath.filter import variance_transform, line_integration
from cellprofiler.cpmath.filter import hessian
from cellprofiler.gui.help import HELP_ON_PIXEL_INTENSITIES, PROTIP_AVOID_ICON

ENHANCE = 'Enhance'
SUPPRESS = 'Suppress'

E_SPECKLES = 'Speckles'
E_NEURITES = 'Neurites'
E_DARK_HOLES = 'Dark holes'
E_CIRCLES = 'Circles'
E_TEXTURE = 'Texture'
E_DIC = 'DIC'

N_GRADIENT = "Line structures"
N_TUBENESS = "Tubeness"

class EnhanceOrSuppressFeatures(cpm.CPModule):

    module_name = 'EnhanceOrSuppressFeatures'
    category = "Image Processing"
    variable_revision_number = 4
    
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
            'Select the input image',
            cps.NONE,doc="""
            Select the image with features to be enhanced or suppressed.""")
        
        self.filtered_image_name = cps.ImageNameProvider(
            'Name the output image',
            'FilteredBlue',doc="""
            Enter a name for the feature-enhanced or suppressed image.""")
        
        self.method = cps.Choice(
            'Select the operation',
            [ ENHANCE, SUPPRESS], doc="""
            Select whether you want to enhance or suppress the features you designated.
            <ul>
            <li><i>%(ENHANCE)s:</i> Produce an image whose intensity is largely 
            composed of the features of interest.</li>
            <li <i>%(SUPPRESS)s:</i> Produce an image with the features largely
            removed.</li>
            </ul>"""%globals())
        
        self.enhance_method = cps.Choice(
            'Feature type',
            [E_SPECKLES, E_NEURITES, E_DARK_HOLES, E_CIRCLES, E_TEXTURE, E_DIC],doc="""
            <i>(Used only if %(ENHANCE)s is selected)</i><br>
            This module can enhance three kinds of image intensity features:
            <ul>
            <li><i>%(E_SPECKLES)s:</i> A speckle is an area of enhanced intensity
            relative to its immediate neighborhood. The module enhances
            speckles using a white tophat filter, which is the image minus the
            morphological grayscale opening of the image. The opening operation
            first suppresses the speckles by applying a grayscale erosion to reduce everything
            within a given radius to the lowest value within that radius, then uses
            a grayscale dilation to restore objects larger than the radius to an
            approximation of their former shape. The white tophat filter enhances 
            speckles by subtracting the effects of opening from the original image.
            </li>
            <li><i>%(E_NEURITES)s:</i> Neurites are taken to be long, thin features
            of enhanced intensity. Choose this option to enhance the intensity
            of the neurites using the %(N_GRADIENT)s or %(N_TUBENESS)s methods
            described below.</li>
            <li><i>%(E_DARK_HOLES)s:</i> The module uses morphological reconstruction 
            (the rolling-ball algorithm) to identify dark holes within brighter
            areas, or brighter ring shapes. The image is inverted so that the dark holes turn into
            bright peaks. The image is successively eroded and the eroded image
            is reconstructed at each step, resulting in an image which is
            missing the peaks. Finally, the reconstructed image is subtracted
            from the previous reconstructed image. This leaves circular bright
            spots with a radius equal to the number of iterations performed.
            </li>
            <li><i>%(E_CIRCLES)s:</i> The module calculates the circular Hough transform of
            the image at the diameter given by the feature size. The Hough transform
            will have the highest intensity at points that are centered within a ring
            of high intensity pixels where the ring diameter is the feature size. You
            may want to use the <b>EnhanceEdges</b> module to find the edges of your
            circular object and then process the output by enhancing circles. You can
            use <b>IdentifyPrimaryObjects</b> to find the circle centers and then use
            these centers as seeds in <b>IdentifySecondaryObjects</b> to find whole,
            circular objects using a watershed.</li>
            <li><i>%(E_TEXTURE)s:</i> <b>EnanceOrSuppressFeatures</b> produces an image
            whose intensity is the variance among nearby pixels. This method weights
            pixel contributions by distance using a Gaussian to calculate the weighting.
            You can use this method to separate foreground from background if the foreground
            is textured and the background is not.
            </li>
            <li><i>%(E_DIC)s:</i> This method recovers the optical density of a DIC image by
            integrating in a direction perpendicular to the shear direction of the image.
            </li>
            </ul>
            In addition, this module enables you to suppress certain features (such as speckles)
            by specifying the feature size.""" % globals())
        
        self.object_size = cps.Integer(
            'Feature size', 10,2,doc="""
            <i>(Used only if circles, speckles or neurites are selected, or if suppressing features)</i><br>
            Enter the diameter of the largest speckle, the width of the circle
            or the width of the neurites to be enhanced or suppressed, which
            will be used to calculate an adequate filter size. %(HELP_ON_PIXEL_INTENSITIES)s"""%globals())

        self.hole_size = cps.IntegerRange(
            'Range of hole sizes', value=(1,10),minval=1, doc="""
            <i>(Used only if %(E_DARK_HOLES)s is selected)</i><br>
            The range of hole sizes to be enhanced. The algorithm will
            identify only holes whose diameters fall between these two 
            values."""%globals())

        self.smoothing = cps.Float(
            'Smoothing scale', value = 2.0, minval = 0, doc = """
            <i>(Used only for the %(E_TEXTURE)s, %(E_DIC)s or %(E_NEURITES)s methods)</i><br>
            <ul>
            <li><i>%(E_TEXTURE)s</i>: This is the scale of the texture features, roughly
            in pixels. The algorithm uses the smoothing value entered as
            the sigma of the Gaussian used to weight nearby pixels by distance
            in the variance calculation.</li>
            <li><i>%(E_DIC)s:</i> Specifies the amount of smoothing of the image in the direction parallel to the
            shear axis of the image. The line integration method will leave
            streaks in the image without smoothing as it encounters noisy
            pixels during the course of the integration. The smoothing takes
            contributions from nearby pixels which decreases the noise but
            smooths the resulting image. </li>
            <li><i>%(E_DIC)s:</i> Increase the smoothing to
            eliminate streakiness and decrease the smoothing to sharpen
            the image.</li>
            <li><i>%(E_NEURITES)s:</i> The <i>%(N_TUBENESS)s</i> option uses this scale
            as the sigma of the Gaussian used to smooth the image prior to
            gradient detection.</li>
            </ul>
            <img src="memory:%(PROTIP_AVOID_ICON)s">&nbsp;
            Smoothing can be turned off by entering a value of zero, but this
            is not recommended.""" % globals())
        
        self.angle = cps.Float(
            'Shear angle', value = 0,doc = """
            <i>(Used only for the %(E_DIC)s method)</i><br>
            The shear angle is the direction of constant value for the
            shadows and highlights in a DIC image. The gradients in a DIC
            image run in the direction perpendicular to the shear angle.
            For example, if the shadows run diagonally from lower left
            to upper right and the highlights appear above the shadows,
            the shear angle is 45&deg;. If the shadows appear on top,
            the shear angle is 180&deg; + 45&deg; = 225&deg;.
            """%globals())
        
        self.decay = cps.Float(
            'Decay', value = 0.95, minval = 0.1, maxval = 1,doc = 
            """<i>(Used only for the %(E_DIC)s method)</i><br>
            The decay setting applies an exponential decay during the process
            of integration by multiplying the accumulated sum by the decay
            at each step. This lets the integration recover from accumulated
            error during the course of the integration, but it also results
            in diminished intensities in the middle of large objects.
            Set the decay to a large value, on the order of 1 - 1/diameter
            of your objects if the intensities decrease toward the middle.
            Set the decay to a small value if there appears to be a bias
            in the integration direction."""%globals())
        
        self.neurite_choice = cps.Choice(
            "Enhancement method", 
            [N_TUBENESS, N_GRADIENT],doc = """
            <i>(Used only for the %(E_NEURITES)s method)</i><br>
            Two methods can be used to enhance neurites:<br>
            <ul>
            <li><i>%(N_TUBENESS)s</i>: This method is an adaptation of
            the method used by the <a href="http://www.longair.net/edinburgh/imagej/tubeness/">
            ImageJ Tubeness plugin</a>. The image
            is smoothed with a Gaussian. The Hessian is then computed at every
            point to measure the intensity gradient and the eigenvalues of the
            Hessian are computed to determine the magnitude of the intensity.
            The absolute maximum of the two eigenvalues gives a measure of
            the ratio of the intensity of the gradient in the direction of
            its most rapid descent versus in the orthogonal direction. The
            output image is the absolute magnitude of the highest eigenvalue
            if that eigenvalue is negative (white neurite on dark background),
            otherwise, zero.</li>
            <li><i>%(N_GRADIENT)s</i>: The module takes the difference of the
            white and black tophat filters (a white tophat filtering is the image minus 
            the morphological grayscale opening of the image; a black tophat filtering is the 
            morphological grayscale closing of the image minus the image). 
            The effect is to enhance lines whose width is the "feature size".</li>
            </ul>"""%globals())
        
    def settings(self):
        return [ self.image_name, self.filtered_image_name,
                self.method, self.object_size, self.enhance_method,
                self.hole_size, self.smoothing, self.angle, self.decay,
                self.neurite_choice]


    def visible_settings(self):
        result = [self.image_name, self.filtered_image_name,
                  self.method]
        if self.method == ENHANCE:
            result += [self.enhance_method]
            if self.enhance_method == E_DARK_HOLES:
                result += [self.hole_size]
            elif self.enhance_method == E_TEXTURE:
                result += [self.smoothing]
            elif self.enhance_method == E_DIC:
                result += [self.smoothing, self.angle, self.decay]
            elif self.enhance_method == E_NEURITES:
                result += [self.neurite_choice]
                if self.neurite_choice == N_GRADIENT:
                    result += [self.object_size]
                else:
                    result += [self.smoothing]
            else:
                result += [self.object_size]
        else:
            result += [self.object_size]
        return result
            
    def run(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_grayscale = True)
        #
        # Match against Matlab's strel('disk') operation.
        #
        radius = (float(self.object_size.value)-1.0) / 2.0
        mask = image.mask if image.has_mask else None
        pixel_data = image.pixel_data
        if self.method == ENHANCE:
            if self.enhance_method == E_SPECKLES:
                result = white_tophat(pixel_data, radius, mask)
            elif self.enhance_method == E_NEURITES:
                if self.neurite_choice == N_GRADIENT:
                    #
                    # white_tophat = img - opening
                    # black_tophat = closing - img
                    # desired effect = img + white_tophat - black_tophat
                    #                = img + img - opening - closing + img
                    #                = 3*img - opening - closing
                    result = (3 * pixel_data - 
                              opening(pixel_data, radius, mask) -
                              closing(pixel_data, radius, mask))
                    result[result > 1] = 1
                    result[result < 0] = 0
                else:
                    sigma = self.smoothing.value
                    smoothed = gaussian_filter(pixel_data, sigma)
                    L = hessian(smoothed, return_hessian = False,
                                return_eigenvectors = False)
                    #
                    # The positive values are darker pixels with lighter
                    # neighbors. The original ImageJ code scales the result
                    # by sigma squared - I have a feeling this might be
                    # a first-order correction for e**(-2*sigma), possibly
                    # because the hessian is taken from one pixel away
                    # and the gradient is less as sigma gets larger.
                    #
                    result = -L[:, :, 0] * (L[:, :, 0] < 0) * sigma * sigma
                if image.has_mask:
                    result[~mask] = pixel_data[~mask]
            elif self.enhance_method == E_DARK_HOLES:
                min_radius = max(1,int(self.hole_size.min / 2))
                max_radius = int((self.hole_size.max+1)/2)
                result = enhance_dark_holes(pixel_data, min_radius,
                                            max_radius, mask)
            elif self.enhance_method == E_CIRCLES:
                result = circular_hough(pixel_data, radius + .5, mask=mask)
            elif self.enhance_method == E_TEXTURE:
                result = variance_transform(pixel_data,
                                            self.smoothing.value,
                                            mask = mask)
            elif self.enhance_method == E_DIC:
                result = line_integration(pixel_data, 
                                          self.angle.value,
                                          self.decay.value,
                                          self.smoothing.value)
            else:
                raise NotImplementedError("Unimplemented enhance method: %s"%
                                          self.enhance_method.value)
        elif self.method == SUPPRESS:
            if image.has_mask:
                result = opening(image.pixel_data, radius, image.mask)
            else:
                result = opening(image.pixel_data, radius)
        else:
            raise ValueError("Unknown filtering method: %s"%self.method)
        result_image = cpi.Image(result, parent_image=image)
        workspace.image_set.add(self.filtered_image_name.value, result_image)
        
        if self.show_window:
            workspace.display_data.image = image.pixel_data
            workspace.display_data.result = result

    def display(self, workspace, figure):
        image = workspace.display_data.image
        result = workspace.display_data.result
        figure.set_subplots((2, 1))
        figure.subplot_imshow_grayscale(0, 0, image,
                                        "Original: %s" % self.image_name.value)
        figure.subplot_imshow_grayscale(1, 0, result,
                                        "Filtered: %s" % self.filtered_image_name.value,
                                        sharexy = figure.subplot(0, 0))
        
    def upgrade_settings(self, setting_values, variable_revision_number,
                             module_name, from_matlab):
        '''Adjust setting values if they came from a previous revision
        
        setting_values - a sequence of strings representing the settings
                         for the module as stored in the pipeline
        variable_revision_number - the variable revision number of the
                         module at the time the pipeline was saved. Use this
                         to determine how the incoming setting values map
                         to those of the current module version.
        module_name - the name of the module that did the saving. This can be
                      used to import the settings from another module if
                      that module was merged into the current module
        from_matlab - True if the settings came from a Matlab pipeline, False
                      if the settings are from a CellProfiler 2.0 pipeline.
        
        Overriding modules should return a tuple of setting_values,
        variable_revision_number and True if upgraded to CP 2.0, otherwise
        they should leave things as-is so that the caller can report
        an error.
        '''
        if not from_matlab and variable_revision_number == 1:
            #
            # V1 -> V2, added enhance method and hole size
            #
            setting_values = setting_values + [E_SPECKLES, "1,10"]
            variable_revision_number = 2
        if not from_matlab and variable_revision_number == 2:
            #
            # V2 -> V3, added texture and DIC
            #
            setting_values = setting_values + [ "2.0", "0", ".95"]
            variable_revision_number = 3
        if not from_matlab and variable_revision_number == 3:
            setting_values = setting_values + [N_GRADIENT]
            variable_revision_number = 4
        return setting_values, variable_revision_number, from_matlab

EnhanceOrSuppressSpeckles = EnhanceOrSuppressFeatures
