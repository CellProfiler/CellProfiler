# In Example 1f, we add a new setting to an existing module and make the new
# version capable of loading a pipeline containing the old version.
# 
# We've taken quite a bit of care to ensure that old, even CellProfiler 1.0,
# pipelines can be loaded by every subsequent version, in almost all cases
# without any modification. The key to this process is the 
# variable_revision_number. We change the variable_revision_number for the
# module whenever we make a change to how the pipeline might be saved.
# This happens when a setting is added or changed to a different type or when
# the choice text for a choice setting changes (in some cases, we've added
# code to handle both the old and new text without changing the revision).
#
# Here, variable_revision_number 2 of the module lets the user specify
# the sigma for the Gaussian and, in version 1, someone had misspelled
# "Gaussian" as "Gassian". The exercise is to add the new sigma setting
# and construct the method, "upgrade_settings", which is responsible
# for converting the settings of a version 1 module to a version 2 module.
#
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi

from scipy.ndimage import gaussian_filter
from centrosome.filter import sobel

S_GAUSSIAN = "Gaussian"
S_GAUSSIAN_V1 = "Gassian"
S_SOBEL = "Sobel"
class Example1f(cpm.CPModule):
    module_name = "Example1f"
    #
    # Change the variable_revision_number to 2
    #
    variable_revision_number = 1
    ##variable_revision_number = 2
    category = "Image Processing"
    
    def create_settings(self):
        self.filter_choice = cps.Choice("Filter choice", 
                                        [S_GAUSSIAN, S_SOBEL])
        self.input_image_name = cps.ImageNameSubscriber("Input image")
        self.output_image_name = cps.ImageNameProvider("Output image", "Filtered")
        #
        # Add the new sigma setting
        #
        ##self.sigma = cps.Float("Sigma", 1)
        
    def settings(self):
        return [self.filter_choice, self.input_image_name, 
                self.output_image_name, 
        #
        # Remember to add the new sigma setting to "settings"
        #
        ##      self.sigma
        ]
    
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
    
    #
    # upgrade_settings does most of the work of converting from version 1
    # to version 2.
    #
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
        #
        # Check if the settings were saved with variable_revision_number 1
        # of this module
        #
        if variable_revision_number == 1:
            #
            # If so, try to figure out if we need to change the text of
            # the filter_choice setting to the correct spelling. The filter_choice
            # is the first setting in settings().
            #
            filter_choice = setting_values[0]
            if filter_choice == S_GAUSSIAN_V1:
                filter_choice = S_GAUSSIAN
            #
            # Version 1 had two settings: the filter choice and the input
            # image name. Version 2 has three - the Gaussian's sigma is
            # the last one at the end. Whatever we put here will be the default
            # setting that's used for an upgrade of a module saved by Version 1.
            # We choose a Gaussian of 1.0 by default.
            #
            setting_values = [filter_choice, setting_values[1], "1"]
            #
            # Remember to change the variable_revision_number to be returned
            # to the current one.
            #
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab