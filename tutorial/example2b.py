'''<b>Example2</b> - an image processing module
<hr>
This is the boilerplate for an image processing module.
'''
import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi

class Example2b(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Example2b"
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
                                                       "SmileyFace")
    
    def settings(self):
        #
        # Add your ImageNameProvider and ImageNameSubscriber to the
        # settings that are returned.
        #
        return [self.input_image_name, self.output_image_name]
    
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
        # Remember to *copy* the image, otherwise you'll change the original!
        #
        pixel_data = input_image.pixel_data.copy()
        if pixel_data.ndim == 3:
            #
            # This converts a color image to grayscale.
            #
            pixel_data = np.mean(pixel_data, 2)
        #
        # I'm going to try and draw a smiley face on the image
        #
        # First, find a center that will work.
        #
        min_dim = np.min(pixel_data.shape)
        #
        # If the image is woefully small, don't do anything
        #
        if min_dim < 20:
            pass
        else:
            center = min_dim / 2
            #
            # Pick a radius for the face
            #
            face_radius = center * 2 / 3
            #
            # Pick a radius for the smile
            #
            smile_radius = face_radius * 3 / 4
            #
            # Pick eye points
            #
            eye_i_center = center * 2 / 3
            eye_j_centers = np.array([ center * 3 / 4, center * 5 / 4])
            eye_radius = face_radius / 8
            #
            # Now I grid the image. i is the row # per pixel and j is the column
            #
            i, j = np.mgrid[-center:(pixel_data.shape[0] - center),
                            -center:(pixel_data.shape[1] - center)]
            #
            # The face is a circle at the face_radius + / 2
            #
            # What I do is find the distance, subtract from the face radius
            # to get the distance from the ideal and create a binary mask
            # on the image for the points within the criteria
            #
            d = np.sqrt(i * i + j * j)
            mask = np.abs(d - face_radius) <= 2
            #
            # OK, add the smile. I could use a bigger radius and a higher
            # center to make it look better, but I've only budgeted myself
            # 15 minutes to write this thing.
            #
            # I restrict the smile circle to the bottom half of the image
            #
            smile_mask = (np.abs(d - smile_radius) <= 2) & (i > center / 4)
            mask = mask | smile_mask
            #
            # The eyes... let's just make one and blit it
            #
            ii, jj = np.mgrid[-eye_radius:(eye_radius+1),
                              -eye_radius:(eye_radius+1)]
            eye_mask = ii*ii + jj*jj < eye_radius*eye_radius
            for eye_j_center in eye_j_centers:
                mask[(eye_i_center - eye_radius):
                     (eye_i_center + eye_radius + 1),
                     (eye_j_center - eye_radius):
                     (eye_j_center + eye_radius + 1)] = eye_mask
            #######################################
            #
            # Respect the image's mask
            #
            #######################################
            if input_image.has_mask:
                #
                # Only invert stuff that's in the input mask
                #
                mask = mask & input_image.mask
            #
            # Now invert everything in the mask
            #
            pixel_data[mask] = np.max(pixel_data) - pixel_data[mask]
        #
        # Save it in the image set
        #
        output_image = cpi.Image(pixel_data, parent_image=input_image)
        image_set.add(self.output_image_name.value,
                      output_image)
        #
        # oooo gotta see it, right?
        #
        if workspace.show_frame:
            # Put the original image and the final one into display_data
            workspace.display_data.input_image = input_image.pixel_data
            workspace.display_data.output_image = pixel_data
            
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
            title = "CellProfiler image modules are fun")        