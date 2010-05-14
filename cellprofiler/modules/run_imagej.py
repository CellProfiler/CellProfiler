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

import bioformats

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs

CM_COMMAND = "Command"
CM_MACRO = "Macro"

class RunImageJ(cpm.CPModule):
    module_name = "RunImageJ"
    variable_revision_number = 1
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
            from imagej.macros import get_commands
            J.attach()
            try:
                return sorted(get_commands())
            finally:
                J.detach()
            
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
                self.pause_before_proceeding]
    
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
        result += [self.pause_before_proceeding, self.show_imagej_button]
        return result
    
    def on_show_imagej(self):
        '''Show the ImageJ user interface
        
        This method shows the ImageJ user interface when the user presses
        the Show ImageJ button.
        '''
        from cellprofiler.utilities.jutil import attach, detach
        from imagej.macros import show_imagej
        attach()
        try:
            show_imagej()
        finally:
            detach()
        
    def is_interactive(self):
        return self.pause_before_proceeding.value
    
    def run(self, workspace):
        '''Run the imageJ command'''
        import cellprofiler.utilities.jutil as J
        from imagej.macros import execute_command, execute_macro
        import imagej.windowmanager as ijwm
        import imagej.imageprocessor as ijiproc
        import imagej.imageplus as ijip
        
        if self.wants_to_set_current_image:
            input_image_name = self.current_input_image_name.value
            img = workspace.image_set.get_image(input_image_name)
        else:
            img = None
        J.attach()
        try:
            if img is not None:
                ij_processor = ijiproc.make_image_processor(img.pixel_data * 255.0)
                image_plus = ijip.make_imageplus_from_processor(
                    input_image_name, ij_processor)
                ijwm.set_current_image(image_plus)
                current_image = image_plus
            else:
                current_image = ijwm.get_current_image()
            if self.command_or_macro == CM_COMMAND:
                execute_command(self.command.value, self.options.value)
            else:
                execute_macro(self.macro.value)
            if (not cpprefs.get_headless()) and self.pause_before_proceeding:
                import wx
                wx.MessageBox("Please edit the image in ImageJ and hit OK to proceed",
                              "Waiting for ImageJ")
            if self.wants_to_get_current_image:
                output_image_name = self.current_output_image_name.value
                image_plus = ijwm.get_current_image()
                ij_processor = image_plus.getProcessor()
                pixel_data = ijiproc.get_image(ij_processor) / 255.0
                image = cpi.Image(pixel_data)
                workspace.image_set.add(output_image_name, image)
        finally:
            J.detach()
        if self.is_interactive():
            self.display(workspace)
            
    def display(self, workspace):
        figure = workspace.create_or_find_figure(subplots=(2,1))
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
                figure.subplot_imshow_color(1,0, pixel_data, title=title)
            else:
                figure.subplot_imshow_bw(1,0, pixel_data, title=title)
        else:
            figure.figure.text(.75, .5, "No output image",
                               verticalalignment='center',
                               horizontalalignment='center')
        
            