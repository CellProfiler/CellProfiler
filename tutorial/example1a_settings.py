# example1.py
#
# This is the minimum skeleton for a CellProfiler module.
#

# We get the CPModule class from here. "cpm" is the standard alias for the
# Python module, "cellprofiler.cpmodule".

import cellprofiler.cpmodule as cpm

# This is where all settings are defined. See below for explanation.

import cellprofiler.settings as cps

#
# This is the module class definition. Each module is a Python class
# whose base class is CPModule. All that means is that anything that
# CellProfiler expects to find in a module will use whatever is in
# CPModule unless you define an override of it in your module
#
class Example1a(cpm.CPModule):
    #
    # Every CellProfiler module must have a variable_revision_number,
    # module_name and category. variable_revision_number will be explained
    # in an advanced section - you can leave it as-is for now without
    # understanding it further.
    #
    # module_name is the name that will be used to display your module,
    #             for instance in the menus
    #
    # category is the category it will fall into. Categories are used
    #          in the menus as well to group modules of similar function.
    #          You can make up your own category, but we have some suggested
    #          standards:
    #
    # File Processing - your module's purpose is primarily to load or store
    #                   files or access a database or similar
    #
    # Image Processing - your module transforms input images to produce
    #                    derived images. An example might be a module that
    #                    enhanced a feature, such as edges, in an image
    #
    # Object Processing - your module is primarily concerned with "Objects"
    #                     It might segment an image, modify existing
    #                     objects or determine relationships between them.
    #
    # Measurement - your module quantifies images and/or object shapes and
    #               produces measurements for them.
    #
    # Data Tools - these will be explained in a later tutorial
    #
    # Other - if it doesn't fit into one of the above.
    # 
    variable_revision_number = 1
    module_name = "Example1a"
    category = "Other"
    
    #
    # The next thing that every module must have is a create_settings method.
    # A setting is a variable that influences the behavior of your module.
    # There are settings that let you enter text and numbers, make choices
    # off of lists (custom text and images, objects and measurements that are 
    # available for your module to use) and image and object names for
    # images and objects produced by your module
    #
    # This module has none, so this is the "do-nothing" version, so far.
    #
    def create_settings(self): # "self" refers to the module's class attributes
        '''Create a fresh set of module settings'''
        ##self.text_setting = cps.Text("Text setting", "suggested value")
        ##self.choice_setting = cps.Choice(
        ##    "Choice setting", ["Choice 1", "Choice 2", "Choice 3"])
        ##self.binary_setting = cps.Binary("Binary setting", False)
        ##self.integer_setting = cps.Integer("Integer setting", 15)
        ##self.float_setting = cps.Float("Float setting", 1.5)
    #
    # You need to be able to tell CellProfiler about the settings in your
    # module. The "settings" method returns the settings in the order that
    # they will be loaded and saved from your pipeline file.
    #
    # The method also tells CellProfiler the display order for your settings...
    # unless you use "visible_settings" which we'll go over later
    #
    def settings(self):
        '''Return these settings to CellProfiler'''
        ##return [self.text_setting,
        ##        self.choice_setting,
        ##       self.binary_setting,
        ##        self.integer_setting,
        ##        self.float_setting]
    #
    # Finally, you need a run method. This is executed when your pipeline
    # is run by CellProfiler.
    #
    # The workspace contains all of the state of your analysis. Later
    # tutorials will go over all of those pieces.
    #
    def run(self, workspace):
        '''Execute the module's code on the current image set'''
        ##integer_value = self.integer_setting.value
        ##float_value = self.float_setting.value
        ##print "%d + %f = %f" % (integer_value,
        ##                        float_value,
        ##                        integer_value + float_value)
        
    #
    # We'll cover the display in Example # 1e, but here's a quick display
    # to give you something to look at when you execute the module.
    #
    # We've changed the way the display method works for the upcoming release
    # to separate the UI from the part of the program that executes the
    # pipeline. That lets us run the pipeline in a separate thread, a separate
    # process or even on a separate machine and still do the display.
    #
    def display(self, workspace, frame=None):
        if frame is not None:
            #
            # New style: tell the figure frame that we want to break the frame
            # into a 1 x 1 grid of axes.
            #
            # Use the *new* version of subplot table to make a table that's
            # much prettier than the old one.
            #
            frame.set_subplots((1,1))
            frame.subplot_table(
                0, 0,
                [[setting.text, setting.value_text] for setting in self.settings()],
                col_labels=["Setting", "Value"])
        else:
            #
            # The old version
            #
            frame = workspace.create_or_find_figure(subplots=(1,1))
            frame.subplot_table(
                0, 0,
                [[setting.text, setting.value_text] for setting in self.settings()])
    
    #
    # Prior to the current release, a module had to tell CellProfiler whether
    # it interacted with the user interface inside the "run" method and by
    # default, a module was marked interactive just in case it did use the
    # user interface.
    # 
    # CellProfiler would use the indicator to figure out whether "run" had
    # to be run in the user interface thread (CellProfiler would crash under
    # OS-X otherwise).
    #
    # In the upcoming release, "run" isn't allowed to interact with the user
    # interface directly, so you will not need to override is_interactive
    # in the future.
    #
    # We'll cover the new UI interaction mechanism in example2c.
    #
    def is_interactive(self):
        return False