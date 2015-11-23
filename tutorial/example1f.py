from centrosome.filter import sobel
from scipy.ndimage import gaussian_filter

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

S_GAUSSIAN = "Gassian"
S_SOBEL = "Sobel"
class Example1f(cpm.CPModule):
    module_name = "Example1f"
    variable_revision_number = 1
    category = "Image Processing"
    def create_settings(self):
        self.filter_choice = cps.Choice("Filter choice", 
                                        [S_GAUSSIAN, S_SOBEL])
        self.input_image_name = cps.ImageNameSubscriber("Input image")
        self.output_image_name = cps.ImageNameProvider("Output image", "Filtered")
        
    def settings(self):
        return [self.filter_choice, self.input_image_name, self.output_image_name]
    
    def run(self, workspace):
        image_set = workspace.image_set
        image = image_set.get_image(self.input_image_name.value)
        pixel_data = image.pixel_data
        if self.filter_choice == S_GAUSSIAN:
            pixel_data = gaussian_filter(pixel_data, sigma=1)
        else:
            pixel_data = sobel(pixel_data)
        output = cpi.Image(pixel_data, parent_image = image)
        image_set.add(self.output_image_name.value, output)
        if self.show_window:
            workspace.display_data.input_image = image.pixel_data
            workspace.display_data.output_image = pixel_data
        
    def display(self, workspace, figure=None):
        if figure is None:
            figure = workspace.create_or_find_figure(subplots=(2, 1))
        else:
            figure.set_subplots((2, 1))
        ax = figure.subplot_imshow(0, 0, workspace.display_data.input_image)
        figure.subplot_imshow(1, 0, workspace.display_data.output_image,
                              sharex = ax,
                              sharey = ax)