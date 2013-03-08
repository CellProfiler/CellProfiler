import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi

from scipy.ndimage import gaussian_filter
from cellprofiler.cpmath.filter import sobel

S_GAUSSIAN = "Gaussian"
S_GAUSSIAN_V1 = "Gassian"
S_SOBEL = "Sobel"
class Example1f(cpm.CPModule):
    module_name = "Example1f"
    variable_revision_number = 2
    category = "Image Processing"
    def create_settings(self):
        self.filter_choice = cps.Choice("Filter choice", 
                                        [S_GAUSSIAN, S_SOBEL])
        self.input_image_name = cps.ImageNameSubscriber("Input image")
        self.output_image_name = cps.ImageNameProvider("Output image", "Filtered")
        self.sigma = cps.Float("Sigma", 1)
        
    def settings(self):
        return [self.filter_choice, self.input_image_name, 
                self.output_image_name, self.sigma]
    
    def run(self, workspace):
        image_set = workspace.image_set
        image = image_set.get_image(self.input_image_name.value)
        pixel_data = image.pixel_data
        if self.filter_choice == S_GAUSSIAN:
            pixel_data = gaussian_filter(pixel_data, sigma=self.sigma.value)
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
    
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        if variable_revision_number == 1:
            filter_choice = setting_values[0]
            if filter_choice == S_GAUSSIAN_V1:
                filter_choice = S_GAUSSIAN
            setting_values = [filter_choice, setting_values[1], "1"]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab