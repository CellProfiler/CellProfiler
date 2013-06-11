'''<b>RunImageJ</b> runs an ImageJ command.
<hr>

<a href="http://rsbweb.nih.gov/ij/">ImageJ</a> is an image processing and analysis program.
It operates by processing commands that operate on one or more images,
possibly modifying the images. ImageJ has a macro language which can
be used to program its operation and customize its operation, similar to
CellProfiler pipelines. ImageJ maintains a current image and most commands
operate on this image, but it's possible to load multiple images into
ImageJ and operate on them together.

The <b>RunImageJ</b> module runs one ImageJ command or macro per cycle. It first
loads the images you want to process into ImageJ, then runs the command, and, if
desired, retrieves images you want to process further in CellProfiler.'''

__version__ = "$Revision$"

import numpy as np
import sys

import bioformats
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs
from cellprofiler.gui.help import BATCH_PROCESSING_HELP_REF
if bioformats.USE_IJ2:
    import imagej.imagej2 as IJ2
else:
    import imagej.macros as M
    import imagej.parameterhandler as P
import cellprofiler.utilities.jutil as J

CM_COMMAND = "Command"
CM_MACRO = "Macro"
CM_NOTHING = "Nothing"

D_FIRST_IMAGE_SET = "FirstImageSet"
D_LAST_IMAGE_SET = "LastImageSet"

cached_commands = None
cached_choice_tree = None
ij2_module_service = None

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
            "Run an ImageJ command or macro?", [CM_COMMAND, CM_MACRO],
            doc = """This setting determines whether <b>RunImageJ</b> runs either a:
            <ul>
            <li><i>Command:</i> Select from a list of available ImageJ commands
            (those items contained in the ImageJ menus); or</li>
            <li><i>Macro:</i> A series of ImageJ commands/plugins that you write yourself.</li>
            </ul>""")
        #
        # Load the commands in visible_settings so that we don't call
        # ImageJ unless someone tries the module
        #
        self.command = self.make_command_choice(
            "Command",
            doc = """<i>(Used only if running a command)</i><br>
            The command to execute when the module runs.""")
                                                
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
            "Macro", 'run("Invert");',
            multiline = True,
            doc="""<i>(Used only if running a macro)</i><br>
            This is the ImageJ macro to be executed. For help on
            writing macros, see <a href="http://rsb.info.nih.gov/ij/developer/macro/macros.html">here</a>.""")
        
        self.options = cps.Text(
            "Options", "",
            doc = """<i>(Used only if running a command)</i><br>
            Use this setting to provide options to the command.""")

        self.wants_to_set_current_image = cps.Binary(
            "Input the currently active image in ImageJ?", True,
            doc="""<p>Check this setting if you want to set the currently 
            active ImageJ image using an image from a 
            prior CellProfiler module.</p>
            <p>Leave it unchecked to use the currently 
            active image in ImageJ. You may want to do this if you
            have an output image from a prior <b>RunImageJ</b>
            that you want to perform further operations upon
            before retrieving the final result back to CellProfiler.</p>""")

        self.current_input_image_name = cps.ImageNameSubscriber(
            "Select the input image",
            doc="""<i>(Used only if setting the currently active image)</i><br>
            This is the CellProfiler image that will become 
            ImageJ's currently active image.
            The ImageJ commands and macros in this module will perform 
            their operations on this image. You may choose any image produced
            by a prior CellProfiler module.""")

        self.wants_to_get_current_image = cps.Binary(
            "Retrieve the currently active image from ImageJ?", True,
            doc="""Check this setting if you want to retrieve ImageJ's
            currently active image after running the command or macro. 
            <p>Leave
            the setting unchecked if the pipeline does not need to access
            the current ImageJ image. For example, you might want to run
            further ImageJ operations with additional <b>RunImageJ</b>
            upon the current image
            prior to retrieving the final image back to CellProfiler.</p>""")

        self.current_output_image_name = cps.ImageNameProvider(
            "Name the current output image", "ImageJImage",
            doc="""<i>(Used only if retrieving the currently active image)</i><br>
            This is the CellProfiler name for ImageJ's current image after
            processing by the command or macro. The image will be a
            snapshot of the current image after the command has run, and
            will be available for processing by subsequent CellProfiler modules.""")
        
        self.pause_before_proceeding = cps.Binary(
            "Wait for ImageJ before continuing?", False,
            doc = """Some ImageJ commands and macros are interactive; you
            may want to adjust the image in ImageJ before continuing. Check
            this box to stop CellProfiler while you adjust the image in
            ImageJ. Leave the box unchecked to immediately use the image.
            <br>
            This command will not wait if CellProfiler is executed in
            batch mode. See <i>%(BATCH_PROCESSING_HELP_REF)s</i> for more
            details on batch processing."""%globals())
        
        self.prepare_group_choice = cps.Choice(
            "Run a command or macro before each group of images?", [CM_NOTHING, CM_COMMAND, CM_MACRO],
            doc="""You can run an ImageJ macro or a command <i>before</i> each group of
            images. This can be useful in order to set up ImageJ before
            processing a stack of images. Choose <i>%(CM_NOTHING)s</i> if
            you do not want to run a command or macro, <i>%(CM_COMMAND)s</i>
            to choose a command to run or <i>%(CM_MACRO)s</i> to run a macro.
            """ % globals())
        
        self.prepare_group_command = self.make_command_choice(
            "Command", 
            doc = """<i>(Used only if running a command before an image group)</i><br>
            The command to execute before processing a group of images.""")

        self.prepare_group_macro = cps.Text(
            "Macro", 'run("Invert");',
            multiline = True,
            doc="""<i>(Used only if running a macro before an image group)</i><br>
            This is the ImageJ macro to be executed before processing
            a group of images. For help on writing macros, see 
            <a href="http://rsb.info.nih.gov/ij/developer/macro/macros.html">here</a>.""")
        
        self.prepare_group_options = cps.Text(
            "Options", "",
            doc = """<i>(Used only if running a command before an image group)</i><br>
            Use this setting to provide options to the command.""")
        
        self.post_group_choice = cps.Choice(
            "Run a command or macro after each group of images?", [CM_NOTHING, CM_COMMAND, CM_MACRO],
            doc="""You can run an ImageJ macro or a command <i>after</i> each group of
            images. This can be used to do some sort of operation on a whole
            stack of images that have been accumulated by the group operation.
            Choose <i>%(CM_NOTHING)s</i> if you do not want to run a command or 
            macro, <i>%(CM_COMMAND)s</i> to choose a command to run or 
            <i>%(CM_MACRO)s</i> to run a macro.
            """ % globals())
        
        self.post_group_command = self.make_command_choice(
            "Command", 
            doc = """
            <i>(Used only if running a command after an image group)</i><br>
            The command to execute after processing a group of images.""")
        
        self.post_group_macro = cps.Text(
            "Macro", 'run("Invert");',
            multiline = True,
            doc="""<i>(Used only if running a macro after an image group)</i><br>
            This is the ImageJ macro to be executed after processing
            a group of images. For help on writing macros, see 
            <a href="http://rsb.info.nih.gov/ij/developer/macro/macros.html">here</a>.""")
        
        self.post_group_options = cps.Text(
            "Options", "",
            doc = """<i>(Used only if running a command after an image group)</i><br>
            Use this setting to provide options to the command or
            macro.""")
        
        self.wants_post_group_image = cps.Binary(
            "Retrieve the image output by the group operation?", False,
            doc="""You can retrieve the image that is currently active in ImageJ
            at the end of macro processing and use it later in CellProfiler.
            The image will only be available during the last cycle of the
            image group. Check this setting to use the active image in CellProfiler
            or leave it unchecked if you do not want to use the active image.
            """)
        
        self.post_group_output_image = cps.ImageNameProvider(
            "Name the group output image", "ImageJGroupImage",
            doc="""<i>(Used only if retrieving an image after an image group operation)</i><br>
            This setting names the output image produced by the
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
        
    def make_command_choice(self, label, doc):
        '''Make a version-appropriate command chooser setting
        
        label - the label text for the setting
        doc - its documentation
        '''
        if bioformats.USE_IJ2:
            return cps.TreeChoice(label, "None", self.get_choice_tree, doc = doc)
        else:
            return cps.Choice(label, [], value="None", 
                              choices_fn = self.get_command_choices,
                              doc = doc)
        
    def get_command_choices(self, pipeline):
        return sorted(self.get_cached_commands().keys())
    
    @staticmethod
    def get_context():
        import imagej.ijbridge as ijbridge
        ijb = ijbridge.get_ij_bridge()
        return ijb.context
    
    @staticmethod
    def get_cached_commands():
        global cached_commands
        global ij2_module_service
        if cached_commands is None:
            import imagej.ijbridge as ijbridge
            ijb = ijbridge.get_ij_bridge()
            import cellprofiler.utilities.jutil as J
            commands = ijb.get_commands()
            if hasattr(commands, 'values'):
                values = commands.values
            else:
                values = [None] * len(commands)
            cached_commands = {}
            for key, value in zip(commands, values):
                cached_commands[key] = value
        return cached_commands

    def get_choice_tree(self):
        '''Get the ImageJ command choices for the TreeChoice control
        
        The menu items are augmented with a third tuple entry which is
        the ModuleInfo for the command.
        '''
        global cached_choice_tree
        if cached_choice_tree is not None:
            return cached_choice_tree
        context = RunImageJ.get_context()
        ij2_module_service = IJ2.get_module_service(context)
        tree = []
        
        for module_info in ij2_module_service.getModules():
            menu_path = module_info.getMenuPath()
            items = [IJ2.wrap_menu_entry(x)
                     for x in J.iterate_collection(menu_path)]
            if len(items) == 0:
                continue
            current_tree = tree
            for item in items:
                name = item.getName()
                weight = item.getWeight()
                matches = [node for node in current_tree
                           if node[0] == name]
                if len(matches) > 0:
                    current_node = matches[0]
                else:
                    current_node = [name, [], module_info, weight]
                    current_tree.append(current_node)
                current_tree = current_node[1]
            # mark the leaf.
            current_node[1] = None
        def sort_tree(tree):
            '''Recursively sort a tree in-place'''
            for node in tree:
                if node[1] is not None:
                    sort_tree(node[1])
            tree.sort(lambda node1, node2: cmp(node1[-1], node2[-1]))
        sort_tree(tree)
        cached_choice_tree = tree
        return cached_choice_tree
        
    def get_command_settings(self, command, d):
        '''Get the settings associated with the current command
        
        d - the dictionary that persists the setting. None = regular
        '''
        key = command.get_unicode_value()
        if not d.has_key(key):
            if bioformats.USE_IJ2:
                try:
                    module_info = command.get_selected_leaf()[2]
                except cps.ValidationError:
                    return []
                result = []
                inputs = module_info.getInputs()
                module = module_info.createModule()
                implied_outputs = []
                for module_item in inputs:
                    field_type = module_item.getType()
                    label = module_item.getLabel()
                    value = module_item.getValue(module)
                    minimum = module_item.getMinimumValue()
                    maximum = module_item.getMaximumValue()
                    description = module_item.getDescription()
                    if field_type == IJ2.FT_BOOL:
                        setting = cps.Binary(
                            label,
                            J.call(value, "booleanValue", "()Z"),
                            doc = description)
                    elif field_type == IJ2.FT_INTEGER:
                        if minimum is not None:
                            minimum = J.call(minimum, "intValue", "()I")
                        if maximum is not None:
                            maximum = J.call(maximum, "intValue", "()I")
                        setting = cps.Integer(
                            label,
                            J.call(value, "intValue", "()I"),
                            minval = minimum,
                            maxval = maximum,
                            doc = description)
                    elif field_type == IJ2.FT_FLOAT:
                        if minimum is not None:
                            minimum = J.call(minimum, "floatValue", "()F")
                        if maximum is not None:
                            maximum = J.call(maximum, "floatValue", "()F")
                        setting = cps.Float(
                            label,
                            J.call(value, "floatValue", "()F"),
                            minval = minimum,
                            maxval = maximum,
                            doc = description)
                    elif field_type == IJ2.FT_STRING:
                        choices = module_item.getChoices()
                        value = J.to_string(value)
                        if choices is not None:
                            choices = [J.to_string(choice) 
                                       for choice 
                                       in J.iterate_collection(choices)]
                            setting = cps.Choice(
                                label, choices, value, doc = description)
                        else:
                            setting = cps.Text(
                                label, value, doc = description)
                    elif field_type == IJ2.FT_COLOR:
                        if value is not None:
                            value = IJ2.color_rgb_to_html(value)
                        else:
                            value = "#ffffff"
                        setting = cps.Color(label, value, doc = description)
                    elif field_type == IJ2.FT_IMAGE:
                        setting = cps.ImageNameSubscriber(
                            label, "InputImage",
                            doc = description)
                        #
                        # This is a Display for ij2 - the plugin typically
                        # scribbles all over the display's image. So
                        # we list it as an output too.
                        #
                        implied_outputs.append((
                            cps.ImageNameProvider(
                                label, "OutputImage",
                                doc = description), module_item))
                    elif field_type == IJ2.FT_OVERLAY:
                        setting = cps.ObjectNameSubscriber(
                            label, "ImageJObject",
                            doc = description)
                    else:
                        continue
                    result.append((setting, module_item))
                for output in module_info.getOutputs():
                    field_type = output.getType()
                    if field_type == IJ2.FT_IMAGE:
                        result.append((cps.ImageNameProvider(
                            label, "ImageJImage",
                            doc = description), output))
                result += implied_outputs
                d[key] = result
                return [setting for setting, module_info in result]
            else:
                cc = self.get_cached_commands()
                if (not cc.has_key(key)) or (cc[key] is None):
                    return []
                classname = cc[key]
                try:
                    plugin = M.get_plugin(classname)
                except:
                    d[key] = []
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
            d[key] = result
        elif bioformats.USE_IJ2:
            result = [setting for setting, module_info in d[key]]
        else:
            result = d[key]
        return result
        
    def is_advanced(self, command, d):
        '''A command is an advanced command if there are settings for it'''
        if bioformats.USE_IJ2:
            return True
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
            if not self.is_advanced(self.command,
                                    self.command_settings_dictionary):
                result += [self.options]
            else:
                cs = self.get_command_settings(self.command,
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
            if not self.is_advanced(self.prepare_group_command,
                                    self.pre_command_settings_dictionary):
                result += [self.prepare_group_options]
            else:
                cs = self.get_command_settings(self.prepare_group_command,
                                               self.pre_command_settings_dictionary)
                result += cs
                self.pre_command_settings += cs
        result += [self.post_group_choice]
        del self.post_command_settings[:]
        if self.post_group_choice == CM_MACRO:
            result += [self.post_group_macro]
        elif self.post_group_choice == CM_COMMAND:
            result += [self.post_group_command]
            if not self.is_advanced(self.post_group_command,
                                    self.post_command_settings_dictionary):
                result += [self.post_group_options]
            else:
                cs = self.get_command_settings(self.post_group_command,
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
        if bioformats.USE_IJ2:
            J.run_in_main_thread(
                lambda :
                RunImageJ.get_context().loadService("imagej.ui.UIService"),
                True)
        else:
            import imagej.ijbridge as ijbridge
            ijb = ijbridge.get_ij_bridge()
            ijb.show_imagej()
        
    def is_interactive(self):
        return False
    
    def prepare_group(self, workspace, grouping, image_numbers):
        '''Prepare to run a group
        
        RunImageJ remembers the image number of the first and last image
        for later processing.
        '''
        d = self.get_dictionary(workspace.image_set_list)
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
            command = self.prepare_group_command
            macro = self.prepare_group_macro.value
            options = self.prepare_group_options.value
            d = self.pre_command_settings_dictionary
        elif when == D_LAST_IMAGE_SET:
            choice = self.post_group_choice.value
            command = self.post_group_command
            macro = self.post_group_macro.value
            options = self.post_group_options.value
            d = self.pre_command_settings_dictionary
        else:
            choice = self.command_or_macro.value
            command = self.command
            macro  = self.macro.value
            options = self.options.value
            d = self.command_settings_dictionary
            
        if choice == CM_COMMAND:
            if self.is_advanced(command, d):
                self.execute_advanced_command(workspace, command, d)
            else:
                ijb.execute_command(command.value, options)
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
        wants_display = workspace.frame is not None
        if wants_display:
            workspace.display_data.input_images = input_images = []
            workspace.display_data.output_images = output_images = []
        key = command.get_unicode_value()
        if bioformats.USE_IJ2:
            node = command.get_selected_leaf()
            module_info = node[2]
            module = IJ2.wrap_module(module_info.createModule())
            context = self.get_context()
            display_service = IJ2.get_display_service(context)
                
            display_dictionary = {}
            for setting, module_item in d[key]:
                field_type = module_item.getType()
                if isinstance(setting, cps.ImageNameProvider):
                    continue
                if field_type == IJ2.FT_BOOL:
                    value = J.make_instance("java/lang/Boolean",
                                            "(Z)V", setting.value)
                elif field_type == IJ2.FT_INTEGER:
                    value = J.make_instance("java/lang/Integer",
                                            "(I)V", setting.value)
                elif field_type == IJ2.FT_FLOAT:
                    value = J.make_instance("java/lang/Double",
                                            "(D)V", setting.value)
                elif field_type == IJ2.FT_STRING:
                    value = setting.value
                elif field_type == IJ2.FT_COLOR:
                    value = IJ2.make_color_rgb_from_html(setting.value)
                elif field_type == IJ2.FT_IMAGE:
                    image_name = setting.value
                    image = workspace.image_set.get_image(image_name)
                    dataset = IJ2.create_dataset(image.pixel_data,
                                                 setting.value)
                    display = display_service.createDisplay(dataset)
                    if image.has_mask:
                        overlay = IJ2.create_overlay(image.mask)
                        display.displayOverlay(overlay)
                    value = display
                    display_dictionary[module_item.getName()] = display
                    if wants_display:
                        input_images.append((image_name, image.pixel_data))
                module.setInput(module_item.getName(), value)
            module_service = IJ2.get_module_service(context)
            module_service.run(module)
            for setting, module_item in d[key]:
                if isinstance(setting, cps.ImageNameProvider):
                    name = module_item.getName()
                    output_name = setting.value
                    if display_dictionary.has_key(name):
                        display = display_dictionary[name]
                    else:
                        display = IJ2.wrap_display(module.getOutput(name))
                    ds = display_service.getActiveDataset(display)
                    pixel_data = ds.get_pixel_data()
                    image = cpi.Image(pixel_data)
                    workspace.image_set.add(output_name, image)
                    if wants_display:
                        output_images.append((output_name, pixel_data))
            for display in display_dictionary.values():
                panel = IJ2.wrap_display_panel(display.getDisplayPanel())
                panel.close()
        else:
            from imagej.imageplus import make_imageplus_from_processor
            from imagej.imageplus import get_imageplus_wrapper
            from imagej.imageprocessor import make_image_processor
            from imagej.imageprocessor import get_image
            
            command = command.value
            settings = d[command]
            classname = self.get_cached_commands()[command]
            plugin = M.get_plugin(classname)
            fp_in = P.get_input_fields_and_parameters(plugin)
            result = []
            image_set = workspace.image_set
            assert isinstance(image_set, cpi.ImageSet)
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
                J.execute_runnable_in_main_thread(J.run_script(
                    """new java.lang.Runnable() { run:function() {
                        importClass(Packages.imagej.plugin.PlugInFunctions);
                        PlugInFunctions.runInteractively(plugin);
                    }};""", dict(plugin=plugin)), True)
            else:
                J.execute_runnable_in_main_thread(plugin, True)
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
              self.is_advanced(self.command,
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
                command = self.make_command_choice("", "")
                command.set_value_text(setting_values[idx_cmd])
                command_settings += self.get_command_settings(
                    command, d)
        
            
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
        
            
