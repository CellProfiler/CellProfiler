'''<b>Example2a</b> - An example of an image processing function.
<hr>
This example deconvolves the image with a Gaussian. Given a point spread
function that is an accurate representation of how optics aberations map
a point to pixels in an image, the deconvolution with that point spread
function uses the information in the blurred image to reconstruct the
true intensity.

Here, we implement the Richardson/Lucy algorithm. The choice of a Gaussian
as the point-spread function is an arbitrary one; in reality, a PSF could
be determined from an image with true point sources, such as a well-localized
fluorophore.

References:
http://en.wikipedia.org/wiki/Deconvolution#Optics_and_other_imaging

The code is taken from the following page:
http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
'''
import numpy as np
#
# We use scipy.stats.norm to make the Gaussian
#
from scipy.stats import norm
#
# We use scipy.signal.convolve2d to perform the convolution
#
from scipy.signal import convolve2d

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi

class Example2a(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Example2a"
    category = "Image Processing"
    
    def create_settings(self):
        #
        # Put your ImageNameProvider and ImageNameSubscriber here
        #
        # use self.input_image_name as the ImageNameSubscriber
        # use self.output_image_name as the ImageNameProvider
        #
        # Those are the names that are expected by the unit tests.
        #
        self.input_image_name = cps.ImageNameSubscriber("Input image")
        self.output_image_name = cps.ImageNameProvider("Output image", 
                                                       "Sharpened")
        self.scale = cps.Float(
            "Scale", .5, 0,
        doc="""This is the sigma of the Gaussian used as the point spread function""")
        #
        # We use a number of iterations to perform. An alternative or adjunct
        # would be to have the loop exit based on the estimated error reaching
        # a certain value
        #
        self.iterations = cps.Integer(
            "Iterations", 10, 1,
            doc = """The number of times to iterate toward maximum likelihood
            estimate.""")
    
    def settings(self):
        #
        # Add your ImageNameProvider and ImageNameSubscriber to the
        # settings that are returned.
        #
        return [self.input_image_name, self.output_image_name,
                self.scale, self.iterations]
    
    def run(self, workspace):
        image_set = workspace.image_set
        #
        # Get your image from the image set using the ImageNameProvider
        # Get the pixel data from the image
        # Do something creative if you'd like
        # Make a cpi.Image using the transformed pixel data
        # put your image back in the image set.
        #
        # If you need help figuring out which methods to use,
        # you can always use the Python help:
        #
        # help(cpi.Image)
        # help(cpi.ImageSet)
        #
        input_image = image_set.get_image(self.input_image_name.value)
        #
        # Get the observed image from the pixel_data
        #
        observed = input_image.pixel_data
        if observed.ndim == 3:
            #
            # This converts a color image to grayscale. A hint: you
            # can do
            # image_set.get_Image(self.input_image_name.value,
            #                     must_be_grayscale=True)
            #
            # instead.
            #
            observed = np.mean(observed, 2)
        #
        ###################
        #
        # Compute the Gaussian
        #
        scale = self.scale.value
        #
        # We pick a radius of 6x scale or at least 4 for the kernel.
        #
        r = max(scale * 6, 4)
        #
        # Make a grid going from -r to r inclusive
        #
        ig, jg = np.mgrid[-r:(r+1), -r:(r+1)].astype(float)
        #
        # The kernel is the pdf of the normal random variable evaluated
        # at the distance from the center.
        #
        rv = norm(scale = scale)
        psf = rv.pdf(np.sqrt(ig*ig + jg*jg))
        #
        # Normalize the probabilities to sum to 1
        #
        psf = psf / np.sum(psf)
        #
        # number of times to execute the refinement loop
        #
        iterations = self.iterations.value
        #
        # This is the code borrowed from 
        # http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
        #
        #function latent_est = RL_deconvolution(observed, psf, iterations)
        #   % to utilise the conv2 function we must make sure the inputs are double
        #   observed = double(observed);
        #   psf      = double(psf);
        #   % initial estimate is arbitrary - uniform 50% grey works fine
        #   latent_est = 0.5*ones(size(observed));
        latent_est = 0.5 * np.ones(observed.shape)
        #   % create an inverse psf
        #   psf_hat = psf(end:-1:1,end:-1:1);
        psf_hat = psf[::-1, ::-1] # Technically not necessary since symmetric
        #   % iterate towards ML estimate for the latent image
        #   for i= 1:iterations
        for _ in range(iterations):
            #   est_conv      = conv2(latent_est,psf,'same');
            est_conv = convolve2d(latent_est, psf, 
                                  mode='same', # return an array of the same size
                                  boundary='symm') # reflect the image at boundary
            #   relative_blur = observed./est_conv;
            relative_blur = observed / est_conv
            #   error_est     = conv2(relative_blur,psf_hat,'same'); 
            error_est = convolve2d(relative_blur, psf_hat, 'same', 'symm')
            #   latent_est    = latent_est.* error_est;
            latent_est = latent_est * error_est
            #end
        #
        # Save it in the image set
        #
        output_image = cpi.Image(latent_est, parent_image=input_image)
        image_set.add(self.output_image_name.value,
                      output_image)
        #
        # Display the image
        #
        if workspace.show_frame:
            # Put the original image and the final one into display_data
            workspace.display_data.input_image = input_image.pixel_data
            workspace.display_data.output_image = latent_est
            
    #
    # The display interface is changing / has changed.
    # This is a recipe to make yours work with both
    #
    def display(self, workspace, figure=None):
        if figure is None:
            figure = workspace.create_or_find_figure(subplots=(2, 1))
        else:
            figure.set_subplots((2, 1))
        figure.subplot_imshow_grayscale(
            0, 0, workspace.display_data.input_image,
            title = self.input_image_name.value)
        figure.subplot_imshow_grayscale(
            1, 0, workspace.display_data.output_image,
            title = "Sharpened image")        