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

__version__ = "$Revision$"

import numpy as np
import sys

import bioformats
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs
import imagej.macros as M
import imagej.parameterhandler as P
import cellprofiler.utilities.jutil as J

CM_COMMAND = "Command"
CM_MACRO = "Macro"
CM_NOTHING = "Nothing"

D_FIRST_IMAGE_SET = "FirstImageSet"
D_LAST_IMAGE_SET = "LastImageSet"

cached_commands = None

'''The index of the imageJ command in the settings'''
IDX_COMMAND_CHOICE = 0
IDX_COMMAND = 1
IDX_PRE_COMMAND_CHOICE = 9
IDX_PRE_COMMAND = 10
IDX_POST_COMMAND_CHOICE = 13
IDX_POST_COMMAND = 14

class RunImageJ(cpm.CPModule):
    module_name = "RunImageJ"
    variable_revision_number = 3
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
            
        self.command = cps.Choice(
            "Command:", [], value="None", choices_fn = self.get_command_choices,
            doc = """The command to execute when the module runs.""")
        self.command_settings_dictionary = {}
        self.command_settings = []
        self.command_settings_count = cps.HiddenCount(
            self.command_settings, "Command settings count")
        self.pre_command_settings_dictionary = {}
        self.pre_command_settings = []
        self.pre_command_settings_count = cps.HiddenCount(
            self.pre_command_settings, "Prepare group command settings count")
        self.post_command_settings_dictionary = {}
        self.post_command_settings = []
        self.post_command_settings_count = cps.HiddenCount(
            self.post_command_settings, "Post-group command settings count")
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
            processing a stack of images. Choose <i>%(CM_NOTHING)s</i> if
            you do not want to run a command or macro, <i>%(CM_COMMAND)s</i>
            to choose a command to run or <i>%(CM_MACRO)s</i> to run a macro.
            """ % globals())
        self.prepare_group_command = cps.Choice(
            "Command:", [], value="None", choices_fn = self.get_command_choices,
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
            Choose <i>%(CM_NOTHING)s</i> if you do not want to run a command or 
            macro, <i>%(CM_COMMAND)s</i> to choose a command to run or 
            <i>%(CM_MACRO)s</i> to run a macro.
            """ % globals())
        self.post_group_command = cps.Choice(
            "Command:", [], value="None", choices_fn = self.get_command_choices,
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
        
    def get_command_choices(self, pipeline):
        return sorted(self.get_cached_commands().keys())
    
    @staticmethod
    def get_cached_commands():
        global cached_commands
        if cached_commands is None:
            import cellprofiler.utilities.jutil as J
            import imagej.ijbridge as ijbridge
            ijb = ijbridge.get_ij_bridge()
            commands = ijb.get_commands()
            if hasattr(commands, 'values'):
                values = commands.values
            else:
                values = [None] * len(commands)
            cached_commands = {}
            for key, value in zip(commands, values):
                cached_commands[key] = value
        return cached_commands
        
    def get_command_settings(self, command, d):
        '''Get the settings associated with the current command
        
        d - the dictionary that persists the setting. None = regular
        '''
        cc = self.get_cached_commands()
        if (not cc.has_key(command)) or (cc[command] is None):
            return []
        if not d.has_key(command):
            classname = cc[command]
            try:
                plugin = M.get_plugin(classname)
            except:
                d[command] = []
                return []
            fp_in = P.get_input_fields_and_parameters(plugin)
            result = []
            for field, parameter in fp_in:
                field_type = P.get_field_type(field)
                label = parameter.label() or ""
                if field_type == P.FT_BOOL:
                    result += [cps.Binary(label, field.getBoolean(plugin))]
                elif field_type == P.FT_INTEGER:
                    result += [cps.Integer(label, field.getLong(plugin))]
                elif field_type == P.FT_FLOAT:
                    result += [cps.Float(label, field.getDouble(plugin))]
                elif field_type == P.FT_STRING:
                    result += [cps.Text(label, J.to_string(field.get(plugin)))]
                else:
                    assert field_type == P.FT_IMAGE
                    result += [cps.ImageNameSubscriber(label, "None")]
                    
            fp_out = P.get_output_fields_and_parameters(plugin)
            for field, parameter in fp_out:
                field_type = P.get_field_type(field)
                if field_type == P.FT_IMAGE:
                    result += [cps.ImageNameProvider(parameter.label() or "",
                                                     "Output")]
            d[command] = result
        else:
            result = d[command]
        return result
        
    def is_advanced(self, command, d):
        '''A command is an advanced command if there are settings for it'''
        return len(self.get_command_settings(command, d)) > 0
    
    def settings(self):
        '''The settings as loaded or stored in the pipeline'''
        return ([
            self.command_or_macro, self.command, self.macro,
            self.options, self.wants_to_set_current_image,
            self.current_input_image_name,
            self.wants_to_get_current_image, self.current_output_image_name,
            self.pause_before_proceeding,
            self.prepare_group_choice, self.prepare_group_command,
            self.prepare_group_macro, self.prepare_group_options,
            self.post_group_choice, self.post_group_command,
            self.post_group_macro, self.post_group_options,
            self.wants_post_group_image, self.post_group_output_image,
            self.command_settings_count, self.pre_command_settings_count,
            self.post_command_settings_count] + self.command_settings +
                self.pre_command_settings + self.post_command_settings)
    
    def visible_settings(self):
        '''The settings as seen by the user'''
        result = [self.command_or_macro]
        del self.command_settings[:]
        if self.command_or_macro == CM_COMMAND:
            result += [self.command]
            if not self.is_advanced(self.command.value,
                                    self.command_settings_dictionary):
                result += [self.options]
            else:
                cs = self.get_command_settings(self.command.value,
                                               self.command_settings_dictionary)
                result += cs
                self.command_settings += cs
        else:
            result += [self.macro]
        result += [self.wants_to_set_current_image]
        if self.wants_to_set_current_image:
            result += [self.current_input_image_name]
        result += [self.wants_to_get_current_image]
        if self.wants_to_get_current_image:
            result += [self.current_output_image_name]
        result += [ self.prepare_group_choice]
        del self.pre_command_settings[:]
        if self.prepare_group_choice == CM_MACRO:
            result += [self.prepare_group_macro]
        elif self.prepare_group_choice == CM_COMMAND:
            result += [self.prepare_group_command]
            if not self.is_advanced(self.prepare_group_command.value,
                                    self.pre_command_settings_dictionary):
                result += [self.prepare_group_options]
            else:
                cs = self.get_command_settings(self.prepare_group_command.value,
                                               self.pre_command_settings_dictionary)
                result += cs
                self.pre_command_settings += cs
        result += [self.post_group_choice]
        del self.post_command_settings[:]
        if self.post_group_choice == CM_MACRO:
            result += [self.post_group_macro]
        elif self.post_group_choice == CM_COMMAND:
            result += [self.post_group_command]
            if not self.is_advanced(self.post_group_command.value,
                                    self.post_command_settings_dictionary):
                result += [self.post_group_options]
            else:
                cs = self.get_command_settings(self.post_group_command.value,
                                               self.post_command_settings_dictionary)
                result += cs
                self.post_command_settings += cs
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
        
        J.attach()
        ijb = None
        try:
            ijb = ijbridge.get_ij_bridge()
            
            image_set = workspace.image_set
            assert isinstance(image_set, cpi.ImageSet)
            d = self.get_dictionary(workspace.image_set_list)
            if self.wants_to_set_current_image:
                input_image_name = self.current_input_image_name.value
                img = image_set.get_image(input_image_name,
                                          must_be_grayscale = True)
            else:
                img = None
            
            #
            # Run a command or macro on the first image of the set
            #
            if d[D_FIRST_IMAGE_SET] == image_set.number + 1:
                self.do_imagej(ijb, workspace, D_FIRST_IMAGE_SET)
            #
            # Install the input image as the current image
            #
            if img is not None:
                ijb.inject_image(img.pixel_data, input_image_name)
    
            self.do_imagej(ijb, workspace)
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
                self.do_imagej(ijb, workspace, D_LAST_IMAGE_SET)
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
        finally:
            if ijb is not None:
                del(ijb)
            J.detach()

    def do_imagej(self, ijb, workspace, when=None):
        if when == D_FIRST_IMAGE_SET:
            choice = self.prepare_group_choice.value
            command = self.prepare_group_command.value
            macro = self.prepare_group_macro.value
            options = self.prepare_group_options.value
            d = self.pre_command_settings_dictionary
        elif when == D_LAST_IMAGE_SET:
            choice = self.post_group_choice.value
            command = self.post_group_command.value
            macro = self.post_group_macro.value
            options = self.post_group_options.value
            d = self.pre_command_settings_dictionary
        else:
            choice = self.command_or_macro.value
            command = self.command.value
            macro  = self.macro.value
            options = self.options.value
            d = self.command_settings_dictionary
            
        if choice == CM_COMMAND:
            if self.is_advanced(command, d):
                self.execute_advanced_command(workspace, command, d)
            else:
                ijb.execute_command(command, options)
        elif choice == CM_MACRO:
            macro = workspace.measurements.apply_metadata(macro)
            ijb.execute_macro(macro)
        if (choice != CM_NOTHING and 
            (not cpprefs.get_headless()) and 
            self.pause_before_proceeding):
            import wx
            wx.MessageBox("Please edit the image in ImageJ and hit OK to proceed",
                          "Waiting for ImageJ")
    
    def execute_advanced_command(self, workspace, command, d):
        '''Execute an advanced command

        command - name of the command
        d - dictionary to be used to find settings
        '''
        from imagej.imageplus import make_imageplus_from_processor
        from imagej.imageplus import get_imageplus_wrapper
        from imagej.imageprocessor import make_image_processor
        from imagej.imageprocessor import get_image
        
        settings = d[command]
        classname = self.get_cached_commands()[command]
        plugin = M.get_plugin(classname)
        fp_in = P.get_input_fields_and_parameters(plugin)
        result = []
        image_set = workspace.image_set
        assert isinstance(image_set, cpi.ImageSet)
        wants_display = workspace.frame is not None
        if wants_display:
            workspace.display_data.input_images = input_images = []
            workspace.display_data.output_images = output_images = []
        for (field, parameter), setting in zip(fp_in, settings[:len(fp_in)]):
            field_type = P.get_field_type(field)
            label = parameter.label() or ""
            if field_type == P.FT_IMAGE:
                image_name = setting.value
                image = workspace.image_set.get_image(image_name,
                                                      must_be_grayscale = True)
                pixel_data = (image.pixel_data * 255.0).astype(np.float32)
                if wants_display:
                    input_images.append((image_name, pixel_data / 255.0))
                processor = make_image_processor(pixel_data)
                image_plus = make_imageplus_from_processor(image_name, processor)
                field.set(plugin, image_plus)
                del image_plus
                del processor
            elif field_type == P.FT_INTEGER:
                field.setInt(plugin, setting.value)
            elif field_type == P.FT_FLOAT:
                field.setFloat(plugin, setting.value)
            elif field_type == P.FT_BOOL:
                field.setBoolean(plugin, setting.value)
            else:
                field.set(plugin, setting.value)
        #
        # There are two ways to run this:
        # * Batch - just call plugin.run()
        # * Interactive - use PlugInFunctions.runInteractively
        #
        if self.pause_before_proceeding:
            J.static_call('imagej/plugin/PlugInFunctions', 'runInteractively',
                   '(Ljava/lang/Runnable)V', plugin)
        else:
            J.call(plugin, 'run', '()V')
        setting_idx = len(fp_in)
        fp_out = P.get_output_fields_and_parameters(plugin)
        for field, parameter in fp_out:
            field_type = P.get_field_type(field)
            if field_type == P.FT_IMAGE:
                image_name = settings[setting_idx].value
                setting_idx += 1
                image_plus = get_imageplus_wrapper(field.get(plugin))
                processor = image_plus.getProcessor()
                pixel_data = get_image(processor).astype(np.float32) / 255.0
                if wants_display:
                    output_images.append((image_name, pixel_data))
                image = cpi.Image(pixel_data)
                image_set.add(image_name, image)
                
    def display(self, workspace):
        if (self.command_or_macro == CM_COMMAND and 
              self.is_advanced(self.command.value,
                               self.command_settings_dictionary)):
            input_images = workspace.display_data.input_images
            output_images = workspace.display_data.output_images
            primary = None
            if len(input_images) == 0:
                if len(output_images) == 0:
                    figure = workspace.create_or_find_figure(title="RunImageJ, image cycle #%d"%(
                            workspace.measurements.image_set_number))
                    figure.figure.text(.25, .5, "No input image",
                                       verticalalignment='center',
                                       horizontalalignment='center')
                    return
                else:
                    nrows = 1
                    output_images = [ 
                        (name, img, i, 0) 
                        for i, (name, img) in enumerate(output_images)]
                    ncols = len(output_images)
            else:
                input_images = [ 
                    (name, img, i, 0) 
                    for i, (name, img) in enumerate(input_images)]
                ncols = len(input_images)
                if len(output_images) == 0:
                    nrows = 1
                else:
                    nrows = 2
                    output_images = [ 
                        (name, img, i, 1) 
                        for i, (name, img) in enumerate(output_images)]
                    ncols = max(ncols, len(output_images))
            figure = workspace.create_or_find_figure(
                title="RunImageJ, image cycle #%d" % 
                (workspace.measurements.image_set_number), 
                subplots = (ncols, nrows))
            for title, pixel_data, x, y in input_images + output_images:
                if pixel_data.ndim == 3:
                    mimg = figure.subplot_imshow_color(x, y, pixel_data, 
                                                       title=title, 
                                                       sharex = primary,
                                                       sharey = primary)
                else:
                    mimg = figure.subplot_imshow_bw(x, y, pixel_data, 
                                                    title=title,
                                                    sharex = primary,
                                                    sharey = primary)
                if primary is None:
                    primary = mimg
            return
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

    def prepare_settings(self, setting_values):
        '''Prepare the settings for loading
        
        set up the advanced settings for the commands
        '''
        for command_settings, idx_choice, idx_cmd, d in (
            (self.command_settings, IDX_COMMAND_CHOICE, IDX_COMMAND, 
             self.command_settings_dictionary),
            (self.pre_command_settings, IDX_PRE_COMMAND_CHOICE, IDX_PRE_COMMAND, 
             self.pre_command_settings_dictionary),
            (self.post_command_settings, IDX_POST_COMMAND_CHOICE, 
             IDX_POST_COMMAND, self.post_command_settings_dictionary)):
            del command_settings[:]
            if setting_values[idx_choice] == CM_COMMAND:
                command_settings += self.get_command_settings(
                    setting_values[idx_cmd], d)
        
            
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
        if variable_revision_number == 2:
            # Added advanced commands
            setting_values = setting_values + ['0','0','0']
            variable_revision_number = 3
        return setting_values, variable_revision_number, from_matlab
        
            
