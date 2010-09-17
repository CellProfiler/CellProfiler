'''<b>RunImageJ</b> runs an ImageJ command.
<hr>

ImageJ is an image processing and analysis program (http://rsbweb.nih.gov/ij/).
It operates by processing commands that operate on one or more images,
possibly modifying the images. ImageJ has a macro language which can
be used to program its operation and customize its operation, similar to
CellProfiler pipelines. ImageJ maintains a current image and most commands
operate on this image, but it's possible to load multiple images into
ImageJ and operate on them together.

The <b>RunImageJ</b> module runs one ImageJ command or macro per cycle. It first
loads the images you want to process into ImageJ, then runs the command, 
then retrieves images you want to process further in CellProfiler.'''

__version__ = "$Revision: 1 %"

import sys

import bioformats

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs

CM_COMMAND = "Command"
CM_MACRO = "Macro"
CM_NOTHING = "Nothing"

D_FIRST_IMAGE_SET = "FirstImageSet"
D_LAST_IMAGE_SET = "LastImageSet"

class RunImageJ(cpm.CPModule):
    module_name = "RunImageJ"
    variable_revision_number = 2
    category = "Image Processing"
    
    def create_settings(self):
        '''Create the settings for the module'''
        self.command_or_macro = cps.Choice(
            "Command or macro?", [CM_COMMAND, CM_MACRO],
            doc = """This setting determines whether <b>RunImageJ</b> runs
            a command, selected from the list of available commands, or
            a macro that you write yourself.""")
        #
        # Load the commands in visible_settings so that we don't call
        # ImageJ unless someone tries the module
        #
        def get_command_choices(pipeline):
            if len(self.command.choices) > 0:
                return self.command.choices
            import cellprofiler.utilities.jutil as J
            import imagej.ijbridge as ijbridge
            ijb = ijbridge.get_ij_bridge()
            return sorted(ijb.get_commands())
            
        self.command = cps.Choice(
            "Command:", [], value="None", choices_fn = get_command_choices,
            doc = """The command to execute when the module runs.""")
        self.macro = cps.Text(
            "Macro:", 'run("Invert");',
            multiline = True,
            doc="""This is the ImageJ macro to be executed. For help on
            writing macros, see http://rsb.info.nih.gov/ij/developer/macro/macros.html""")
        self.options = cps.Text(
            "Options:", "",
            doc = """Use this setting to provide options to the command or
            macro.""")
        self.wants_to_set_current_image = cps.Binary(
            "Set the current image?", True,
            doc="""Check this setting if you want to set the current
            ImageJ image using an image from a previous module. Leave it
            unchecked to use ImageJ's current image.""")
        self.current_input_image_name = cps.ImageNameSubscriber(
            "Current image:",
            doc="""This is the image that will become ImageJ's current image.
            ImageJ commands and macros will perform their operations on this
            image. Choose an image produced by a previous module.""")
        self.wants_to_get_current_image = cps.Binary(
            "Get the current image?", True,
            doc="""Check this setting if you want to retrieve ImageJ's
            current image after running the command or macro. Leave
            the setting unchecked if the pipeline does not need to access
            the current ImageJ image.""")
        self.current_output_image_name = cps.ImageNameProvider(
            "Final image:", "ImageJImage",
            doc="""This is the name for ImageJ's current image after
            processing by the command or macro. The image will be a
            snapshot of the current image after the command has run.""")
        self.pause_before_proceeding = cps.Binary(
            "Wait for ImageJ?", False,
            doc = """Some ImageJ commands and macros are interactive; you
            may want to adjust the image in ImageJ before continuing. Check
            this box to stop CellProfiler while you adjust the image in
            ImageJ. Leave the box unchecked to immediately use the image.
            <br>
            This command will not wait if CellProfiler is executed in
            batch mode.""")
        self.prepare_group_choice = cps.Choice(
            "Run before each group?", [CM_NOTHING, CM_COMMAND, CM_MACRO],
            doc="""You can run an ImageJ macro or a command before each group of
            images. This can be useful in order to set up ImageJ before
            processing a stack of images. Choose <i>%(CM_NOTHING)s</s> if
            you do not want to run a command or macro, <i>%(CM_COMMAND)s</i>
            to choose a command to run or <i>%(CM_MACRO)s</i> to run a macro.
            """ % globals())
        self.prepare_group_command = cps.Choice(
            "Command:", [], value="None", choices_fn = get_command_choices,
            doc = """The command to execute before processing a group of images.""")
        self.prepare_group_macro = cps.Text(
            "Macro:", 'run("Invert");',
            multiline = True,
            doc="""This is the ImageJ macro to be executed before processing
            a group of images. For help on writing macros, see 
            http://rsb.info.nih.gov/ij/developer/macro/macros.html""")
        self.prepare_group_options = cps.Text(
            "Options:", "",
            doc = """Use this setting to provide options to the command or
            macro.""")
        self.post_group_choice = cps.Choice(
            "Run after each group?", [CM_NOTHING, CM_COMMAND, CM_MACRO],
            doc="""You can run an ImageJ macro or a command after each group of
            images. This can be used to do some sort of operation on a whole
            stack of images that have been accumulated by the group operation.
            Choose <i>%(CM_NOTHING)s</s> if you do not want to run a command or 
            macro, <i>%(CM_COMMAND)s</i> to choose a command to run or 
            <i>%(CM_MACRO)s</i> to run a macro.
            """ % globals())
        self.post_group_command = cps.Choice(
            "Command:", [], value="None", choices_fn = get_command_choices,
            doc = """The command to execute after processing a group of images.""")
        self.post_group_macro = cps.Text(
            "Macro:", 'run("Invert");',
            multiline = True,
            doc="""This is the ImageJ macro to be executed after processing
            a group of images. For help on writing macros, see 
            http://rsb.info.nih.gov/ij/developer/macro/macros.html""")
        self.post_group_options = cps.Text(
            "Options:", "",
            doc = """Use this setting to provide options to the command or
            macro.""")
        self.wants_post_group_image = cps.Binary(
            "Save the selected image?", False,
            doc="""You can save the image that is currently selected in ImageJ
            at the end of macro processing and use it later in CellProfiler.
            The image will only be available during the last cycle of the
            group. Check this setting to use the selected image in CellProfiler
            or leave it unchecked if you do not want to use the selected image.
            """)
        self.post_group_output_image = cps.ImageNameProvider(
            "Image name:", "ImageJGroupImage",
            doc="""This setting names the output image produced by the
            ImageJ command or macro that CellProfiler runs after processing
            all images in the group. The image is only available at the
            last cycle in the group""",
            provided_attributes={cps.AGGREGATE_IMAGE_ATTRIBUTE: True,
                                 cps.AVAILABLE_ON_LAST_ATTRIBUTE: True } )
           
        self.show_imagej_button = cps.DoSomething(
            "Show ImageJ", "Show", self.on_show_imagej,
            doc="""Press this button to show the ImageJ user interface.
            You can use the user interface to run ImageJ commands or
            set up ImageJ before a CellProfiler run.""")
        
    def settings(self):
        '''The settings as loaded or stored in the pipeline'''
        return [self.command_or_macro, self.command, self.macro,
                self.options, self.wants_to_set_current_image,
                self.current_input_image_name,
                self.wants_to_get_current_image, self.current_output_image_name,
                self.pause_before_proceeding,
                self.prepare_group_choice, self.prepare_group_command,
                self.prepare_group_macro, self.prepare_group_options,
                self.post_group_choice, self.post_group_command,
                self.post_group_macro, self.post_group_options,
                self.wants_post_group_image, self.post_group_output_image]
    
    def visible_settings(self):
        '''The settings as seen by the user'''
        result = [self.command_or_macro]
        if self.command_or_macro == CM_COMMAND:
            result += [self.command, self.options]
        else:
            result += [self.macro]
        result += [self.wants_to_set_current_image]
        if self.wants_to_set_current_image:
            result += [self.current_input_image_name]
        result += [self.wants_to_get_current_image]
        if self.wants_to_get_current_image:
            result += [self.current_output_image_name]
        result += [ self.prepare_group_choice]
        if self.prepare_group_choice == CM_MACRO:
            result += [self.prepare_group_macro]
        elif self.prepare_group_choice == CM_COMMAND:
            result += [self.prepare_group_command, self.prepare_group_options]
        result += [self.post_group_choice]
        if self.post_group_choice == CM_MACRO:
            result += [self.post_group_macro]
        elif self.post_group_choice == CM_COMMAND:
            result += [self.post_group_command, self.post_group_options]
        if self.post_group_choice != CM_NOTHING:
            result += [self.wants_post_group_image]
            if self.wants_post_group_image:
                result += [self.post_group_output_image]
        result += [self.pause_before_proceeding, self.show_imagej_button]
        return result
    
    def on_show_imagej(self):
        '''Show the ImageJ user interface
        
        This method shows the ImageJ user interface when the user presses
        the Show ImageJ button.
        '''
        import imagej.ijbridge as ijbridge
        ijb = ijbridge.get_ij_bridge()
        ijb.show_imagej()
        
    def is_interactive(self):
        # On Mac, run in main thread for stability
        return sys.platform == 'darwin'
    
    def prepare_group(self, pipeline, image_set_list, grouping,
                      image_numbers):
        '''Prepare to run a group
        
        RunImageJ remembers the image number of the first and last image
        for later processing.
        '''
        d = self.get_dictionary(image_set_list)
        d[D_FIRST_IMAGE_SET] = image_numbers[0]
        d[D_LAST_IMAGE_SET] = image_numbers[-1]
        
    def run(self, workspace):
        '''Run the imageJ command'''
        import imagej.ijbridge as ijbridge
        import cellprofiler.utilities.jutil as J
        
        image_set = workspace.image_set
        assert isinstance(image_set, cpi.ImageSet)
        d = self.get_dictionary(workspace.image_set_list)
        if self.wants_to_set_current_image:
            input_image_name = self.current_input_image_name.value
            img = image_set.get_image(input_image_name,
                                      must_be_grayscale = True)
        else:
            img = None
        
        ijb = ijbridge.get_ij_bridge()
        
        #
        # Run a command or macro on the first image of the set
        #
        if d[D_FIRST_IMAGE_SET] == image_set.number + 1:
            if self.prepare_group_choice == CM_COMMAND:
                ijb.execute_command(self.prepare_group_command.value,
                                    self.prepare_group_options.value)
            elif self.prepare_group_choice == CM_MACRO:
                macro = workspace.measurements.apply_metadata(
                            self.prepare_group_macro.value)
                ijb.execute_macro(macro)
            if (self.prepare_group_choice != CM_NOTHING and 
                (not cpprefs.get_headless()) and 
                self.pause_before_proceeding):
                import wx
                wx.MessageBox("Please edit the image in ImageJ and hit OK to proceed",
                              "Waiting for ImageJ")
        #
        # Install the input image as the current image
        #
        if img is not None:
            ijb.inject_image(img.pixel_data, input_image_name)

        #
        # Do the per-imageset macro or command
        #
        if self.command_or_macro == CM_COMMAND:
            ijb.execute_command(self.command.value, self.options.value)
        else:
            macro = workspace.measurements.apply_metadata(self.macro.value)
            ijb.execute_macro(macro)
        if (not cpprefs.get_headless()) and self.pause_before_proceeding:
            import wx
            wx.MessageBox("Please edit the image in ImageJ and hit OK to proceed",
                          "Waiting for ImageJ")
        #
        # Get the output image
        #
        if self.wants_to_get_current_image:
            output_image_name = self.current_output_image_name.value
            pixel_data = ijb.get_current_image()
            image = cpi.Image(pixel_data)
            image_set.add(output_image_name, image)
        #
        # Execute the post-group macro or command
        #
        if d[D_LAST_IMAGE_SET] == image_set.number + 1:
            if self.post_group_choice == CM_COMMAND:
                ijb.execute_command(self.post_group_command.value, 
                                    self.post_group_options.value)
            elif self.post_group_choice == CM_MACRO:
                macro = workspace.measurements.apply_metadata(
                            self.post_group_macro.value)
                ijb.execute_macro(macro)
            if (self.post_group_choice != CM_NOTHING and 
                (not cpprefs.get_headless()) and 
                self.pause_before_proceeding):
                import wx
                wx.MessageBox("Please edit the image in ImageJ and hit OK to proceed",
                              "Waiting for ImageJ")
            #
            # Save the current ImageJ image after executing the post-group
            # command or macro
            #
            if (self.post_group_choice != CM_NOTHING and
                self.wants_post_group_image):
                output_image_name = self.post_group_output_image.value
                pixel_data = ijb.get_current_image()
                image = cpi.Image(pixel_data)
                image_set.add(output_image_name, image)
        if self.is_interactive():
            self.display(workspace)

        del(ijb)
            
    def display(self, workspace):
        figure = workspace.create_or_find_figure(title="RunImageJ, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(2,1))
        if self.wants_to_set_current_image:
            input_image_name = self.current_input_image_name.value
            img = workspace.image_set.get_image(input_image_name)
            pixel_data = img.pixel_data
            title = "Input image: %s" % input_image_name
            if pixel_data.ndim == 3:
                figure.subplot_imshow_color(0,0, pixel_data, title=title)
            else:
                figure.subplot_imshow_bw(0,0, pixel_data, title=title)
        else:
            figure.figure.text(.25, .5, "No input image",
                               verticalalignment='center',
                               horizontalalignment='center')
        
        if self.wants_to_get_current_image:
            output_image_name = self.current_output_image_name.value
            img = workspace.image_set.get_image(output_image_name)
            pixel_data = img.pixel_data
            title = "Output image: %s" % output_image_name
            if pixel_data.ndim == 3:
                figure.subplot_imshow_color(1,0, pixel_data, title=title,
                                            sharex = figure.subplot(0,0),
                                            sharey = figure.subplot(0,0))
            else:
                figure.subplot_imshow_bw(1,0, pixel_data, title=title,
                                         sharex = figure.subplot(0,0),
                                         sharey = figure.subplot(0,0))
        else:
            figure.figure.text(.75, .5, "No output image",
                               verticalalignment='center',
                               horizontalalignment='center')
    
    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if variable_revision_number == 1:
            setting_values = setting_values + [
                CM_NOTHING, "None",
                'print("Enter macro here")\n', "",
                CM_NOTHING, "None",
                'print("Enter macro here")\n', "",
                cps.NO, "AggregateImage"]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab
        
            
