'''<b>Run ImageJ</b> runs an ImageJ command.
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
desired, retrieves images you want to process further in CellProfiler.

<h4>Technical notes</h4>
<p>ImageJ runs using Java, and as such, relies on proper handling of the Java memory requirements.
When ImageJ starts, the Java Virtual Machine (JVM) allocates a portion of memory for its
own use from the operating system; this memory is called the <i>java heap memory</i>. If you
encounter JVM memory errors, you can tell CellProfiler to increase the size of the Java heap memory 
on startup.</p>
<p>To do this, run CellProfiler from the command line with the following argument:
<code>--jvm-heap-size=JVM_HEAP_SIZE</code><br>
where <code>JVM_HEAP_SIZE</code> is the amount of memory to be reserved for the JVM. Example formats
for <code>JVM_HEAP_SIZE</code> include <i>512000k</i>, <i>512m</i>, <i>1g</i>, etc. For example,
to increase the JVM heap memory to 2GB, use <code>--jvm-heap-size=2g</code></p>
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

import logging
logger = logging.getLogger(__name__)
import numpy as np
import sys
import uuid

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
from cellprofiler.settings import YES, NO
import cellprofiler.preferences as cpprefs
from cellprofiler.gui.help import BATCH_PROCESSING_HELP_REF
import imagej.imagej2 as ij2
from imagej.imagej2 import get_context
import imagej.windowmanager as ijwm
import imagej.imageprocessor as ijiproc
import imagej.imageplus as ijip
import imagej.macros as ijmacros
import cellprofiler.utilities.jutil as J

CM_COMMAND = "Command"
CM_SCRIPT = "Script"
CM_MACRO = "Macro"
CM_NOTHING = "Nothing"

D_FIRST_IMAGE_SET = "FirstImageSet"
D_LAST_IMAGE_SET = "LastImageSet"

cached_commands = None
cached_choice_tree = None

'''The index of the imageJ command in the settings'''
IDX_COMMAND_CHOICE = 0
IDX_COMMAND = 1
IDX_COMMAND_COUNT = 2
IDX_PRE_COMMAND_COUNT = 3
IDX_POST_COMMAND_COUNT = 4
IDX_PRE_COMMAND_CHOICE = 9
IDX_PRE_COMMAND = 10
IDX_POST_COMMAND_CHOICE = 12
IDX_POST_COMMAND = 13

'''ImageJ images are scaled from 0 to 255'''
IMAGEJ_SCALE = 255.0

class RunImageJ(cpm.CPModule):
    module_name = "RunImageJ"
    variable_revision_number = 4
    category = "Image Processing"
    do_not_check=True
    
    def create_settings(self):
        '''Create the settings for the module'''
        logger.debug("Creating RunImageJ module settings")
        J.activate_awt()
        logger.debug("Activated AWT")
        
        self.command_or_macro = cps.Choice(
            "Run an ImageJ command or macro?", 
            [CM_COMMAND, CM_SCRIPT, CM_MACRO],doc = """
            This setting determines whether <b>RunImageJ</b> runs either a:
            <ul>
            <li><i>%(CM_COMMAND)s:</i> Select from a list of available ImageJ commands
            (those items contained in the ImageJ menus); or</li>
            <li><i>%(CM_SCRIPT)s:</i> A script written in one of ImageJ 2.0's
            supported scripting languages.</li>
            <li><i>%(CM_MACRO)s:</i> An ImageJ 1.x macro, written in the
            ImageJ 1.x macro language. <b>Run_ImageJ</b> runs ImageJ in 1.x
            compatability mode.</li>
            </ul>"""%globals())
        #
        # Load the commands in visible_settings so that we don't call
        # ImageJ unless someone tries the module
        #
        self.command = self.make_command_choice(
            "Command",doc = """
            <i>(Used only if running a %(CM_COMMAND)s)</i><br>
            The command to execute when the module runs."""%globals())
                                                
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
            "Macro", 
            """import imagej.command.CommandService;
cmdSvcClass = CommandService.class;
cmdSvc = ImageJ.getService(cmdSvcClass);
cmdSvc.run("imagej.core.commands.assign.InvertDataValues", new Object [] {"allPlanes", true}).get();""",
            multiline = True,doc="""
            <i>(Used only if running a %(CM_MACRO)s)</i><br>
            This is the ImageJ macro to be executed. The syntax for ImageJ
            macros depends on the scripting language engine chosen.
            We suggest that you use the Beanshell scripting language
            <a href="http://www.beanshell.org/manual/contents.html">
            (Beanshell documentation)</a>."""%globals())
        
        all_engines = ij2.get_script_service(get_context()).getLanguages()
        self.language_dictionary = dict(
            [(engine.getLanguageName(), engine) for engine in all_engines])
            
        self.macro_language = cps.Choice(
            "Macro language",
            choices = self.language_dictionary.keys(),doc = """
            This setting chooses the scripting language used to execute
            any macros in this module""")
        
        self.run_group_divider = cps.Divider()
        self.wants_to_set_current_image = cps.Binary(
            "Input the currently active image in ImageJ?", True,doc="""
            Select <i>%(YES)s</i> if you want to set the currently 
            active ImageJ image using an image from a 
            prior CellProfiler module.
            <p>Select <i>%(NO)s</i> to use the currently 
            active image in ImageJ. You may want to do this if you
            have an output image from a prior <b>RunImageJ</b>
            that you want to perform further operations upon
            before retrieving the final result back to CellProfiler.</p>"""%globals())

        self.current_input_image_name = cps.ImageNameSubscriber(
            "Select the input image",doc="""
            <i>(Used only if setting the currently active image)</i><br>
            This is the CellProfiler image that will become 
            ImageJ's currently active image.
            The ImageJ commands and macros in this module will perform 
            their operations on this image. You may choose any image produced
            by a prior CellProfiler module.""")

        self.wants_to_get_current_image = cps.Binary(
            "Retrieve the currently active image from ImageJ?", True,doc="""
            Select <i>%(YES)s</i> if you want to retrieve ImageJ's
            currently active image after running the command or macro. 
            <p>Select <i>%(NO)s</i> if the pipeline does not need to access
            the current ImageJ image. For example, you might want to run
            further ImageJ operations with additional <b>RunImageJ</b>
            upon the current image prior to retrieving the final image 
            back to CellProfiler.</p>"""%globals())

        self.current_output_image_name = cps.ImageNameProvider(
            "Name the current output image", "ImageJImage",doc="""
            <i>(Used only if retrieving the currently active image from ImageJ)</i><br>
            This is the CellProfiler name for ImageJ's current image after
            processing by the command or macro. The image will be a
            snapshot of the current image after the command has run, and
            will be available for processing by subsequent CellProfiler modules.""")
        
        self.pause_before_proceeding = cps.Binary(
            "Wait for ImageJ before continuing?", False,doc = """
            Some ImageJ commands and macros are interactive; you
            may want to adjust the image in ImageJ before continuing. 
            Select <i>%(YES)s</i> to stop CellProfiler while you adjust the image in
            ImageJ. Select <i>%(NO)s</i> to immediately use the image.
            <p>This command will not wait if CellProfiler is executed in
            batch mode. See <i>%(BATCH_PROCESSING_HELP_REF)s</i> for more
            details on batch processing.</p>"""%globals())
        
        self.prepare_group_choice = cps.Choice(
            "Function to run before each group of images?", 
            [CM_NOTHING, CM_COMMAND, CM_SCRIPT, CM_MACRO],doc="""
            You can run an ImageJ 2.0 script, an ImageJ 1.x macro or a command <i>before</i> each group of
            images. This can be useful in order to set up ImageJ before
            processing a stack of images. Choose <i>%(CM_NOTHING)s</i> if
            you do not want to run a command or macro, <i>%(CM_COMMAND)s</i>
            to choose a command to run, <i>%(CM_SCRIPT)s</i> to run an
            ImageJ 2.0 script or <i>%(CM_MACRO)s</i> to run an ImageJ 1.x
            macro in ImageJ 1.x compatibility mode.
            """ % globals())
        
        logger.debug("Finding ImageJ commands")
        
        self.prepare_group_command = self.make_command_choice(
            "Command", doc = """
            <i>(Used only if running a command before an image group)</i><br>
            Select the command to execute before processing a group of images.""")

        self.prepare_group_macro = cps.Text(
            "Macro", 'run("Invert");',
            multiline = True,doc="""
            <i>(Used only if running a macro before an image group)</i><br>
            This is the ImageJ macro to be executed before processing
            a group of images. For help on writing macros, see 
            <a href="http://rsb.info.nih.gov/ij/developer/macro/macros.html">here</a>.""")
        
        self.prepare_group_divider = cps.Divider()
        
        self.post_group_choice = cps.Choice(
            "Function to run after each group of images?", 
            [CM_NOTHING, CM_COMMAND, CM_SCRIPT, CM_MACRO],doc="""
            You can run an ImageJ 2.0 script, an ImageJ macro or a command <i>after</i> each group of
            images. This can be used to do some sort of operation on a whole
            stack of images that have been accumulated by the group operation.
            Choose <i>%(CM_NOTHING)s</i> if
            you do not want to run a command or macro, <i>%(CM_COMMAND)s</i>
            to choose a command to run, <i>%(CM_SCRIPT)s</i> to run an
            ImageJ 2.0 script or <i>%(CM_MACRO)s</i> to run an ImageJ 1.x
            macro in ImageJ 1.x compatibility mode.
            """ % globals())
        
        self.post_group_command = self.make_command_choice(
            "Command", doc = """
            <i>(Used only if running a command after an image group)</i><br>
            The command to execute after processing a group of images.""")
        
        self.post_group_macro = cps.Text(
            "Macro", 'run("Invert");',
            multiline = True,doc="""
            <i>(Used only if running a macro after an image group)</i><br>
            This is the ImageJ macro to be executed after processing
            a group of images. For help on writing macros, see 
            <a href="http://rsb.info.nih.gov/ij/developer/macro/macros.html">here</a>.""")
        
        self.post_group_divider = cps.Divider()
        
        self.wants_post_group_image = cps.Binary(
            "Retrieve the image output by the group operation?", False,doc="""
            You can retrieve the image that is currently active in ImageJ
            at the end of macro processing and use it later in CellProfiler.
            The image will only be available during the last cycle of the
            image group. 
            <p>Select <i>%(YES)s</i> to retrieve the active image for use in CellProfiler.
            Select <i>%(NO)s</i> if you do not want to retrieve the active image.</p>
            """%globals())
        
        self.post_group_output_image = cps.ImageNameProvider(
            "Name the group output image", "ImageJGroupImage",doc="""
            <i>(Used only if retrieving an image after an image group operation)</i><br>
            This setting names the output image produced by the
            ImageJ command or macro that CellProfiler runs after processing
            all images in the group. The image is only available at the
            last cycle in the group""",
            provided_attributes={cps.AGGREGATE_IMAGE_ATTRIBUTE: True,
                                 cps.AVAILABLE_ON_LAST_ATTRIBUTE: True } )
           
        self.show_imagej_button = cps.DoSomething(
            "Show ImageJ", "Show", self.on_show_imagej,doc="""
            Press this button to show the ImageJ user interface.
            You can use the user interface to run ImageJ commands or
            set up ImageJ before a CellProfiler run.""")
        
        logger.debug("Finished creating settings")

    @staticmethod
    def is_leaf(node):
        '''Return True if a tree node holds a command'''
        return len(node) >= 2 and node[2] is not None
    
    def make_command_choice(self, label, doc):
        '''Make a version-appropriate command chooser setting
        
        label - the label text for the setting
        doc - its documentation
        '''
        return cps.TreeChoice(
            label, "None", self.get_choice_tree, 
            fn_is_leaf=self.is_leaf, doc = doc)
        
    def get_choice_tree(self):
        '''Get the ImageJ command choices for the TreeChoice control
        
        The menu items are augmented with a third tuple entry which is
        the ModuleInfo for the command.
        '''
        global cached_choice_tree
        global cached_commands
        if cached_choice_tree is not None:
            return cached_choice_tree
        tree = []
        context = get_context()
        module_service = ij2.get_module_service(context)
        
        for module_info in module_service.getModules():
            if module_info.getMenuRoot() != "app":
                continue
            logger.info("Processing module %s" % module_info.getTitle())
            menu_path = module_info.getMenuPath()
            if menu_path is None or J.call(menu_path, "size", "()I") == 0:
                continue
            current_tree = tree
            #
            # The menu path is a collection of MenuEntry
            #
            for item in J.iterate_collection(menu_path):
                menu_entry = ij2.wrap_menu_entry(item)
                name = menu_entry.getName()
                weight = menu_entry.getWeight()
                matches = [node for node in current_tree
                           if node[0] == name]
                if len(matches) > 0:
                    current_node = matches[0]
                else:
                    current_node = [name, [], None, weight, None]
                    current_tree.append(current_node)
                current_tree = current_node[1]
            # mark the leaf.
            current_node[2] = module_info
            
        def sort_tree(tree):
            '''Recursively sort a tree in-place'''
            for node in tree:
                if node[1] is not None:
                    sort_tree(node[1])
            tree.sort(lambda node1, node2: cmp(node1[-1], node2[-1]))
        sort_tree(tree)
        cached_choice_tree = tree
        return cached_choice_tree
    
    @staticmethod
    def __get_module_info_from_command(command):
        leaf = command.get_selected_leaf()
        module_info = leaf[2]
        if leaf[4] is None:
            # Initialize the module at this point
            module = module_info.createModule()
            J.call(get_context().getContext(), "inject", 
                   "(Ljava/lang/Object;)V", module.o)
            try:
                module.initialize()
            except:
                logger.warn("Failed to initialize %s command." %
                            module_info.getTitle())
            leaf[4] = module
        else:
            module = leaf[4]
        return ij2.wrap_module_info(module.getInfo())
        
    def get_command_settings(self, command, d):
        '''Get the settings associated with the current command
        
        d - the dictionary that persists the setting. None = regular
        '''
        key = command.get_unicode_value()
        if not d.has_key(key):
            try:
                module_info = RunImageJ.__get_module_info_from_command(command)
            except cps.ValidationError:
                logger.info("Could not find command %s" % key)
                return []
            inputs = module_info.getInputs()
            result = []
            for module_item in inputs:
                field_type = module_item.getType()
                label = module_item.getLabel()
                if label is None or len(label) == 0:
                    label = module_item.getName()
                if module_item.isOutput():
                    # if both, qualify which is for input and which for output
                    label = "%s (Input)" % label
                minimum = module_item.getMinimumValue()
                maximum = module_item.getMaximumValue()
                default = module_item.loadValue()
                description = module_item.getDescription()
                if field_type == ij2.FT_BOOL:
                    value = (J.is_instance_of(default, 'java/lang/Boolean') and
                             J.call(default, "booleanValue", "()Z"))
                    if description == None or len(description) == 0:
                        description = """
                        This setting is an input parameter to the ImageJ
                        command that you selected. The parameter enables or
                        disables some feature of the command, but is otherwise
                        undocumented by ImageJ. Please see ImageJ's
                        documentation for further information."""
                    setting = cps.Binary(
                        label,
                        value = value,
                        doc = description)
                elif field_type == ij2.FT_INTEGER:
                    if description == None or len(description) == 0:
                        description = """
                        This setting is an input parameter to the ImageJ
                        command that you selected. The parameter supplies
                        an integer value, but is otherwise undocumented.
                        Please see ImageJ's documentation for further
                        information.
                        """
                    minimum, maximum = [
                        None if x is None else J.call(x, "intValue", "()I")
                        for x in minimum, maximum]
                               
                    if J.is_instance_of(default, 'java/lang/Number'):
                        value = J.call(default, "intValue", "()I")
                    elif minimum is not None:
                        value = minimum
                    elif maximum is not None:
                        value = maximum
                    else:
                        value = 0
                    setting = cps.Integer(
                        label,
                        value = value,
                        minval=minimum,
                        maxval=maximum,
                        doc = description)
                elif field_type == ij2.FT_FLOAT:
                    if description == None or len(description) == 0:
                        description = """
                        This setting is an input parameter to the ImageJ
                        command that you selected. The parameter supplies
                        a numeric value, but is otherwise undocumented.
                        Please see ImageJ's documentation for further
                        information.
                        """
                    minimum, maximum = [
                        None if x is None else J.call(x, "doubleValue", "()D")
                        for x in minimum, maximum]
                    if J.is_instance_of(default, 'java/lang/Number'):
                        value = J.call(default, "doubleValue", "()D")
                    elif minimum is not None:
                        value = minimum
                    elif maximum is not None:
                        value = maximum
                    else:
                        value = 0
                    setting = cps.Float(
                        label,
                        value=value,
                        minval=minimum,
                        maxval=maximum,
                        doc = description)
                elif field_type == ij2.FT_STRING:
                    choices = module_item.getChoices()
                    value = J.to_string(default)
                    if choices is not None and\
                       J.call(choices, "size", "()I") > 0:
                        if description == None or len(description) == 0:
                            description = """
                            This setting is an input parameter to the ImageJ
                            command that you selected. The parameter lets you
                            choose from among several options, but is otherwise
                            undocumented. Please see ImageJ's documentation for
                            further information.
                            """
                        choices = list(J.get_collection_wrapper(
                            choices, J.to_string))
                        setting = cps.Choice(
                            label, choices, value, doc = description)
                    else:
                        if description == None or len(description) == 0:
                            description = """
                            This setting is an input parameter to the ImageJ
                            command that you selected. The parameter supplies
                            a text item to the command, but is otherwise
                            undocumented. Please see ImageJ's documentation for
                            further information.
                            """
                        setting = cps.Text(
                            label, value, doc = description)
                elif field_type == ij2.FT_COLOR:
                    if description == None or len(description) == 0:
                        description = """
                        This setting is an input parameter to the ImageJ
                        command that you selected. The parameter lets you
                        choose a color for the command, but is otherwise
                        undocumented. Please see ImageJ's documentation for
                        further information.
                        """
                    value = "#ffffff"
                    setting = cps.Color(label, value, doc = description)
                elif field_type == ij2.FT_IMAGE:
                    if description == None or len(description) == 0:
                        description = """
                        This setting supplies an input image to the
                        command that you selected, but is otherwise
                        undocumented. Please see ImageJ's documentation for
                        further information.
                        """
                    setting = cps.ImageNameSubscriber(
                        label, "InputImage",
                        doc = description)
                elif field_type == ij2.FT_TABLE:
                    setting = IJTableSubscriber(label, "InputTable",
                                                doc=description)
                elif field_type == ij2.FT_FILE:
                    if description == None or len(description) == 0:
                        description = """
                        This setting is an input parameter to the ImageJ
                        command that you selected. The parameter lets you
                        choose a file for the command, but is otherwise
                        undocumented. Please see ImageJ's documentation for
                        further information.
                        """
                    if default is None:
                        default = ""
                    elif not isinstance(default, basestring):
                        default = J.to_string(default)
                    setting = cps.Pathname(
                        label, default, doc = description)
                elif field_type == ij2.FT_PLUGIN:
                    if description == None or len(description) == 0:
                        description = """
                        This setting is an input parameter to the ImageJ
                        command that you selected. The parameter lets you
                        choose from among several options, but is otherwise
                        undocumented. Please see ImageJ's documentation for
                        further information.
                        """
                    klass = J.call(module_item.o, "getType", 
                                   "()Ljava/lang/Class;")
                    choices = [
                        J.to_string(x) for x in 
                        ij2.get_object_service(get_context()).getObjects(klass)]
                    if len(choices) < 2:
                        continue
                    value = J.to_string(default)
                    setting = cps.Choice(label, choices, value,
                                         doc = description)
                else:
                    continue
                result.append((setting, module_item))
            for output in module_info.getOutputs():
                field_type = output.getType()
                label = output.getLabel()
                if label is None or len(label) == 0:
                    label = output.getName()
                if output.isInput():
                    # if both, qualify which is for input and which for output
                    label = "%s (Output)" % label
                if field_type == ij2.FT_IMAGE:
                    if description == None or len(description) == 0:
                        description = """
                        This setting names the output (or one of the outputs)
                        of the ImageJ command that you selected but is otherwise
                        undocumented. Please see ImageJ's documentation for
                        further information.
                        """
                    result.append((cps.ImageNameProvider(
                        label, "ImageJImage",
                        doc = description), output))
                elif field_type == ij2.FT_TABLE:
                    result.append((IJTableProvider(
                        label, "ImageJTable", doc=description), output))
            d[key] = result
        else:
            result = d[key]
        return [setting for setting, module_info in result]
        
    def is_advanced(self, command, d):
        '''A command is an advanced command if there are settings for it'''
        return True
    
    def is_aggregation_module(self):
        '''RunImageJ is an aggregation module if it performs a prepare or post group command'''
        return self.prepare_group_choice != CM_NOTHING and \
               self.post_group_choice != CM_NOTHING
    
    def settings(self):
        '''The settings as loaded or stored in the pipeline'''
        return ([
            self.command_or_macro, self.command, self.macro,
            self.macro_language,
            self.wants_to_set_current_image,
            self.current_input_image_name,
            self.wants_to_get_current_image, self.current_output_image_name,
            self.pause_before_proceeding,
            self.prepare_group_choice, self.prepare_group_command,
            self.prepare_group_macro, 
            self.post_group_choice, self.post_group_command,
            self.post_group_macro, 
            self.wants_post_group_image, self.post_group_output_image,
            self.command_settings_count, self.pre_command_settings_count,
            self.post_command_settings_count] + self.command_settings +
                self.pre_command_settings + self.post_command_settings)
    
    def on_setting_changed(self, setting, pipeline):
        '''Respond to a setting change
        
        We have to update the ImageJ module settings in response to a
        new choice.
        '''
        for command_choice, command_setting, module_settings, d in (
            (self.command_or_macro, 
             self.command, 
             self.command_settings, 
             self.command_settings_dictionary),
            (self.prepare_group_choice,
             self.prepare_group_command, 
             self.pre_command_settings, 
             self.pre_command_settings_dictionary),
            (self.post_group_choice, 
             self.post_group_command, 
             self.post_command_settings, 
             self.post_command_settings_dictionary)):
            if ((id(setting) == id(command_setting)) or
                (id(setting) == id(command_choice) and 
                 command_choice == CM_COMMAND)):
                del module_settings[:]
                module_settings.extend(self.get_command_settings(command_setting, d))
            elif id(setting) == id(command_choice):
                del module_settings[:]
                
    def visible_settings(self):
        '''The settings as seen by the user'''
        uses_macros = ((self.command_or_macro == CM_SCRIPT) or
                       (self.prepare_group_choice == CM_SCRIPT) or
                       (self.post_group_choice == CM_SCRIPT))
        result = [self.command_or_macro]
        if self.command_or_macro == CM_COMMAND:
            result += [self.command]
            result += self.command_settings
        else:
            result += [self.macro]
        if uses_macros:
            result += [self.macro_language]
        result += [self.run_group_divider, self.wants_to_set_current_image]
        if self.wants_to_set_current_image:
            result += [self.current_input_image_name]
        result += [self.wants_to_get_current_image]
        if self.wants_to_get_current_image:
            result += [self.current_output_image_name]
        result += [ self.prepare_group_choice]
        if self.prepare_group_choice in (CM_SCRIPT, CM_MACRO):
            result += [self.prepare_group_macro]
        elif self.prepare_group_choice == CM_COMMAND:
            result += [self.prepare_group_command]
            result += self.pre_command_settings
        if self.prepare_group_choice != CM_NOTHING:
            result += [self.prepare_group_divider]
        result += [self.post_group_choice]
        if self.post_group_choice in (CM_SCRIPT, CM_MACRO):
            result += [self.post_group_macro]
        elif self.post_group_choice == CM_COMMAND:
            result += [self.post_group_command]
            result += self.post_command_settings
        if self.post_group_choice != CM_NOTHING:
            result += [self.post_group_divider, self.wants_post_group_image]
            if self.wants_post_group_image:
                result += [self.post_group_output_image]
        result += [self.pause_before_proceeding, self.show_imagej_button]
        return result
    
    def on_show_imagej(self):
        '''Show the ImageJ user interface
        
        This method shows the ImageJ user interface when the user presses
        the Show ImageJ button.
        '''
        logger.debug("Starting ImageJ UI")
        ui_service = ij2.get_ui_service(get_context())
        if ui_service is not None and not ui_service.isVisible():
            if cpprefs.get_headless():
                # Silence the auto-updater in the headless preferences
                #
                ij2.update_never_remind()
                
            ui_service.createUI()
        elif ui_service is not None:
            ui = ui_service.getDefaultUI()
            J.execute_runnable_in_main_thread(J.run_script(
                """new java.lang.Runnable() {
                run: function() { 
                    ui.getApplicationFrame().setVisible(true); }}""",
                dict(ui=ui)), True)
        
    def prepare_group(self, workspace, grouping, image_numbers):
        '''Prepare to run a group
        
        RunImageJ remembers the image number of the first and last image
        for later processing.
        '''
        d = self.get_dictionary(workspace.image_set_list)
        d[D_FIRST_IMAGE_SET] = image_numbers[0]
        d[D_LAST_IMAGE_SET] = image_numbers[-1]
        if self.wants_to_set_current_image or self.wants_to_get_current_image:
            # For ImageJ 1.0 and some scripting, the UI has to be open
            # in order to get or set the current image.
            #
            self.on_show_imagej()
        
    def run(self, workspace):
        '''Run the imageJ command'''
        image_set = workspace.image_set
        d = self.get_dictionary(workspace.image_set_list)
        if self.wants_to_set_current_image:
            input_image_name = self.current_input_image_name.value
            img = image_set.get_image(input_image_name,
                                      must_be_grayscale = True)
            if self.show_window:
                workspace.display_data.image_sent_to_ij = img.pixel_data
        else:
            img = None
        display_service = ij2.get_display_service(get_context())
        #
        # Run a command or macro on the first image of the set
        #
        if d.get(D_FIRST_IMAGE_SET) == image_set.image_number:
            self.do_imagej(workspace, D_FIRST_IMAGE_SET)
        ij1_mode = self.command_or_macro == CM_MACRO
        #
        # Install the input image as the current image
        #
        if img is not None:
            ijpixels = img.pixel_data * IMAGEJ_SCALE
            if not ij1_mode:
                dataset = ij2.create_dataset(get_context(), 
                                             ijpixels,
                                             input_image_name)
                display = display_service.createDisplay(
                    input_image_name, dataset)
                display_service.setActiveDisplay(display)
        else:
            ijpixels = None

        self.do_imagej(workspace, input_image=ijpixels)
        #
        # Get the output image
        #
        if self.wants_to_get_current_image and not ij1_mode:
            output_image_name = self.current_output_image_name.value
            for attempt in range(4):
                display = display_service.getActiveImageDisplay()
                if display.o is not None:
                    break
                #
                # Possible synchronization problem with ImageJ 1.0
                # Possible error involving user changing window focus
                #
                import time
                time.sleep(.25)
            else:
                raise ValueError("Failed to retrieve active display")
            pixels = self.save_display_as_image(
                workspace, display, output_image_name)
            if self.show_window:
                workspace.display_data.image_acquired_from_ij = pixels
        #
        # Execute the post-group macro or command
        #
        if d.get(D_LAST_IMAGE_SET) == image_set.image_number:
            self.do_imagej(workspace, D_LAST_IMAGE_SET)
            #
            # Save the current ImageJ image after executing the post-group
            # command or macro
            #
            if (self.post_group_choice != CM_NOTHING and
                self.wants_post_group_image):
                output_image_name = self.post_group_output_image.value
                if not ij1_mode:
                    image_plus = ijwm.get_current_image()
                    ij_processor = image_plus.getProcessor()
                    pixels = ijiproc.get_image(ij_processor).\
                        astype('float32') / IMAGEJ_SCALE
                    image = cpi.Image(pixels, mask=mask)
                    workspace.image_set.add(output_image_name, image)
                else:
                    display = display_service.getActiveImageDisplay()
                    self.save_display_as_image(workspace, display, output_image_name)
                
    def save_display_as_image(self, workspace, display, image_name):
        '''Convert an ImageJ display to an image and save in the image set
        
        workspace - current workspace
        display - an ImageJ Display
        image_name - save the image in the image set using this name.
        '''
        display_view = display.getActiveView()
        dataset = ij2.wrap_dataset(display_view.getData())
        pixel_data = dataset.get_pixel_data() / IMAGEJ_SCALE
        mask = ij2.create_mask(display)
        image = cpi.Image(pixel_data, mask=mask)
        workspace.image_set.add(image_name, image)
        return pixel_data

    def save_dataset_as_image(self, workspace, dataset, image_name):
        '''Convert an ImageJ dataset to an image and save in the image set
        
        workspace - current workspace
        dataset - an ImageJ dataset
        image_name - save the image in the image set using this name.
        '''
        pixel_data = dataset.get_pixel_data() / IMAGEJ_SCALE
        image = cpi.Image(pixel_data)
        workspace.image_set.add(image_name, image)
        return pixel_data

    def do_imagej(self, workspace, when=None, input_image=None):
        if when == D_FIRST_IMAGE_SET:
            choice = self.prepare_group_choice.value
            command = self.prepare_group_command
            macro = self.prepare_group_macro.value
            d = self.pre_command_settings_dictionary
        elif when == D_LAST_IMAGE_SET:
            choice = self.post_group_choice.value
            command = self.post_group_command
            macro = self.post_group_macro.value
            d = self.pre_command_settings_dictionary
        else:
            choice = self.command_or_macro.value
            command = self.command
            macro  = self.macro.value
            d = self.command_settings_dictionary
            
        if choice == CM_COMMAND:
            self.execute_advanced_command(workspace, command, d)
        elif choice == CM_SCRIPT:
            macro = workspace.measurements.apply_metadata(macro)
            script_service = ij2.get_script_service(get_context())
            factory = script_service.getByName(self.macro_language.value)
            engine = factory.getScriptEngine()
            engine.put("ImageJ", get_context())
            result = engine.evalS(macro)
        elif choice == CM_MACRO:
            macro = workspace.measurements.apply_metadata(macro)
            if when is None and\
               (self.wants_to_set_current_image or
                self.wants_to_get_current_image):
                if input_image is None:
                    input_image = np.zeros((16,16), np.float32)
                ij_processor = ijiproc.make_image_processor(
                    input_image.astype('float32'))
                image_plus = ijip.make_imageplus_from_processor(
                    self.current_input_image_name.value, ij_processor)
                if self.wants_to_set_current_image:
                    ijwm.set_current_image(image_plus)
                image_plus = ijip.get_imageplus_wrapper(
                    ijmacros.run_batch_macro(macro, image_plus.o))
                ijwm.set_current_image(image_plus)
                if self.wants_to_get_current_image:
                    ij_processor = image_plus.getProcessor()
                    pixels = ijiproc.get_image(ij_processor).\
                        astype('float32') / IMAGEJ_SCALE
                    image = cpi.Image(pixels)
                    workspace.image_set.add(
                        self.current_output_image_name.value, image)
                    if self.show_window:
                        workspace.display_data.image_acquired_from_ij = pixels
            else:
                ijmacros.execute_macro(macro)
            
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
        context = get_context()
        self.get_command_settings(command, d)
        wants_display = self.show_window
        if wants_display:
            workspace.display_data.input_images = input_images = []
            workspace.display_data.output_images = output_images = []
        key = command.get_unicode_value()
        module_info = RunImageJ.__get_module_info_from_command(command)
        
        input_dictionary = J.get_map_wrapper(
            J.make_instance('java/util/HashMap', "()V"))
        display_dictionary = {}
        display_service = ij2.get_display_service(context)
        for setting, module_item in d[key]:
            if isinstance(setting, cps.ImageNameProvider):
                continue
            field_name = module_item.getName()
            field_type = module_item.getType()
            raw_type = J.call(module_item.o, "getType", "()Ljava/lang/Class;")
            if field_type in (ij2.FT_BOOL, ij2.FT_INTEGER, ij2.FT_FLOAT,
                              ij2.FT_STRING):
                input_dictionary.put(field_name, J.box(setting.value, raw_type))
            elif field_type == ij2.FT_COLOR:
                assert isinstance(setting, cps.Color)
                red, green, blue = setting.to_rgb()
                jobject = J.make_instance(
                    "imagej/util/ColorRGB", "(III)V", red, green, blue)
                input_dictionary.put(field_name, jobject)
            elif field_type == ij2.FT_IMAGE:
                data_class = J.call(module_item.o, "getType", "()Ljava/lang/Class;")
                image_name = setting.value
                image = workspace.image_set.get_image(image_name)
                pixel_data = image.pixel_data * IMAGEJ_SCALE
                
                dataset = ij2.create_dataset(
                    context, pixel_data, image_name)
                if J.call(data_class, "isAssignableFrom",
                          "(Ljava/lang/Class;)Z",
                          J.class_for_name("imagej.data.Dataset")):
                    o = dataset.o
                else:
                    display = display_service.createDisplay(image_name, dataset)
                    display_dictionary[module_item.getName()] = display 
                    if image.has_mask:
                        #overlay_name = "X" + uuid.uuid4().get_hex()
                        #image_dictionary[overlay_name] = image.mask
                        overlay = ij2.create_overlay(context, image.mask)
                        overlay_service = ij2.get_overlay_service(context)
                        overlay_service.addOverlays(
                            display.o, J.make_list([overlay]))
                        ij2.select_overlay(display.o, overlay)
                    if J.call(data_class, "isAssignableFrom",
                              "(Ljava/lang/Class;)Z",
                              J.class_for_name("imagej.data.display.DatasetView")):
                        o = display.getActiveView().o
                    else:
                        o = display.o
                input_dictionary.put(field_name, o)
                if wants_display:
                    input_images.append((image_name, image.pixel_data))
            elif field_type == ij2.FT_TABLE:
                table_name = setting.value
                table = workspace.object_set.get_type_instance(
                    IJ_TABLE_TYPE, table_name)
                input_dictionary.put(field_name, table)
            elif field_type == ij2.FT_FILE:
                jfile = J.make_instance(
                    "java/io/File", "(Ljava/lang/String;)V", setting.value)
                input_dictionary.put(field_name, jfile)
            elif field_type == ij2.FT_PLUGIN:
                klass = J.call(module_item.o, "getType", 
                               "()Ljava/lang/Class;")
                
                oo = ij2.get_object_service(get_context()).getObjects(klass)
                if len(oo) == 0:
                    input_dictionary.put(field_name, None)
                elif len(oo) == 1:
                    input_dictionary.put(field_name, oo[0])
                else:
                    for o in oo:
                        choice = J.to_string(o)
                        if setting.value == choice:
                            input_dictionary.put(field_name, o)
                            break
                    else:
                        input_dictionary.put(field_name, None)
            elif module_item.isRequired():
                input_dictionary.put(field_name, None)
                
        module = module_info.createModule()
        J.call(get_context().getContext(), "inject", 
               "(Ljava/lang/Object;)V", module.o)
        try:
            module.initialize()
            filter_pre = True
        except:
            filter_pre = False
        context = get_context()
        module_service = ij2.get_module_service(context)
        #
        # Filter out the init preprocessor if we already initialized
        #
        pluginService = context.getService("org.scijava.plugin.PluginService")
        preprocessors, postprocessors = [
            J.call(pluginService, "createInstancesOfType", 
                   "(Ljava/lang/Class;)Ljava/util/List;",
                   J.class_for_name(x))
            for x in ("imagej.module.process.PreprocessorPlugin",
                      "imagej.module.process.PostprocessorPlugin")]
        if filter_pre:
            good_preprocessors = []
            for preprocessor in J.get_collection_wrapper(preprocessors):
                if not J.is_instance_of(
                    preprocessor, "imagej/module/process/InitPreprocessor"):
                    good_preprocessors.append(preprocessor)
            preprocessors = J.make_list(good_preprocessors).o
        #
        # Now run the module
        #
        jfuture = J.call(
            module_service.o, "run", 
            "(Limagej/module/Module;Ljava/util/List;Ljava/util/List;Ljava/util/Map;)Ljava/util/concurrent/Future;",
            module.o, preprocessors, postprocessors, input_dictionary.o)
        future = J.get_future_wrapper(jfuture, ij2.wrap_module)
        module = future.get()
        for setting, module_item in d[key]:
            if isinstance(setting, cps.ImageNameProvider):
                name = module_item.getName()
                output_name = setting.value
                if display_dictionary.has_key(name):
                    display = display_dictionary[name]
                    pixel_data = self.save_display_as_image(
                        workspace, display, output_name)
                else:
                    o = module.getOutput(name)
                    if J.is_instance_of(o, "imagej/data/display/ImageDisplay"):
                        display = ij2.wrap_display(o)
                        pixel_data = self.save_display_as_image(
                            workspace, display, output_name)
                    elif J.is_instance_of(o, "imagej/data/display/DatasetView"):
                        pixel_data = self.save_dataset_as_image(
                            workspace,
                            ij2.wrap_dataset(ij2.wrap_data_view(o).getData()),
                            output_name)
                    else:
                        # Assume it's a dataset.
                        pixel_data = self.save_dataset_as_image(
                            workspace, ij2.wrap_dataset(o), output_name)
                
                if wants_display:
                    output_images.append((output_name, pixel_data))
        # Close any displays that we created.
        for display in display_dictionary.values():
            display.close()
                
    def display(self, workspace, figure):
        if (self.command_or_macro == CM_COMMAND and 
              self.is_advanced(self.command,
                               self.command_settings_dictionary)):
            input_images = workspace.display_data.input_images
            output_images = workspace.display_data.output_images
            primary = None
            if len(input_images) == 0:
                if len(output_images) == 0:
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
            figure.set_subplots((ncols, nrows))
            for title, pixel_data, x, y in input_images + output_images:
                if pixel_data.ndim == 3:
                    mimg = figure.subplot_imshow_color(x, y, pixel_data, 
                                                       title=title, 
                                                       sharexy = primary)
                else:
                    mimg = figure.subplot_imshow_bw(x, y, pixel_data, 
                                                    title=title,
                                                    sharexy = primary)
                if primary is None:
                    primary = mimg
            return
        figure.set_subplots((2, 1))
        if self.wants_to_set_current_image:
            input_image_name = self.current_input_image_name.value
            pixel_data = workspace.display_data.image_sent_to_ij
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
            pixel_data = workspace.display_data.image_acquired_from_ij
            title = "Output image: %s" % output_image_name
            if pixel_data.ndim == 3:
                figure.subplot_imshow_color(1,0, pixel_data, title=title,
                                            sharexy = figure.subplot(0,0))
            else:
                figure.subplot_imshow_bw(1,0, pixel_data, title=title,
                                         sharexy = figure.subplot(0,0))
        else:
            figure.figure.text(.75, .5, "No output image",
                               verticalalignment='center',
                               horizontalalignment='center')

    def prepare_settings(self, setting_values):
        '''Prepare the settings for loading
        
        set up the advanced settings for the commands
        '''
        for command_settings, idx_choice, idx_cmd, idx_count, d in (
            (self.command_settings, IDX_COMMAND_CHOICE, IDX_COMMAND,
             IDX_COMMAND_COUNT, self.command_settings_dictionary),
            (self.pre_command_settings, IDX_PRE_COMMAND_CHOICE, IDX_PRE_COMMAND, 
             IDX_PRE_COMMAND_COUNT, self.pre_command_settings_dictionary),
            (self.post_command_settings, IDX_POST_COMMAND_CHOICE, 
             IDX_POST_COMMAND, IDX_POST_COMMAND_COUNT,
             self.post_command_settings_dictionary)):
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
        if variable_revision_number == 3:
            # Removed options, added macro language
            (command_or_macro, command, macro,
             options, wants_to_set_current_image,
             current_input_image_name,
             wants_to_get_current_image, current_output_image_name,
             pause_before_proceeding,
             prepare_group_choice, prepare_group_command,
             prepare_group_macro, prepare_group_options,
             post_group_choice, post_group_command,
             post_group_macro, post_group_options,
             wants_post_group_image, 
             post_group_output_image) = setting_values[:19]
            command_or_macro, command, macro = self.upgrade_settings_from_v3(
                command_or_macro, command, macro)
            prepare_group_choice, prepare_group_command, prepare_group_macro = \
                self.upgrade_settings_from_v3(
                    prepare_group_choice, prepare_group_command, 
                    prepare_group_macro)
            post_group_choice, post_group_command, post_group_macro = \
                self.upgrade_settings_from_v3(
                    post_group_choice, post_group_command, 
                    post_group_macro)
            setting_values = [
                command_or_macro, command, macro, "ECMAScript",
                wants_to_set_current_image, current_input_image_name,
                wants_to_get_current_image, current_output_image_name,
                pause_before_proceeding, prepare_group_choice,
                prepare_group_command, prepare_group_macro, 
                post_group_choice, post_group_command,
                post_group_macro, wants_post_group_image, 
                post_group_output_image] + setting_values[19:]
            variable_revision_number = 4
            
        return setting_values, variable_revision_number, from_matlab
    
    def upgrade_settings_from_v3(self, command_or_macro, command, macro):
        '''Upgrade settings from ImageJ 1.x to ImageJ 2.x
        
        command_or_macro: either CM_COMMAND or CM_MACRO
        
        command: the ImageJ 1.x menu command to execute
        
        macro: the ImageJ 1.x macro to execute
        
        Returns adjusted command_or_macro, command and macro
        '''
        if command_or_macro == CM_COMMAND:
            command_or_macro = CM_MACRO
            macro = 'run("%s");' % command
        return command_or_macro, command, macro
        
        
IJ_TABLE_SETTING_GROUP = "ijtable"
IJ_TABLE_TYPE = "imagej.data.table.TableDisplay"

class IJTableProvider(cps.NameProvider):
    '''A setting provider of ImageJ table names'''
    def __init__(self, text, *args, **kwargs):
        super(IJTableProvider, self).__init__(
            text, IJ_TABLE_SETTING_GROUP, *args, **kwargs)
        
class IJTableSubscriber(cps.NameSubscriber):
    '''A setting subscriber to ImageJ table names'''
    def __init__(self, text, *args, **kwargs):
        super(IJTableSubscriber, self).__init__(
            text, IJ_TABLE_SETTING_GROUP, *args, **kwargs)
        