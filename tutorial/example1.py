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
class Example1(cpm.CPModule):
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
    module_name = "Example1"
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
        pass # You need at least one statement, so this is a do-nothing statement
    
    #
    # You need to be able to tell CellProfiler about the settings in your
    # module. The "settings" method returns the settings in the order that
    # they will be loaded and saved from your pipeline file.
    #
    # The method also tells CellProfiler the display order for your settings...
    # unless you use "visible_settings" which we'll go over later
    #
    def settings(self):
        # [] is an empty list. You should put settings in between the brackets
        # if you have any, for instance "[self.foo, self.bar]" if those were
        # your settings.
        return []
    #
    # Finally, you need a run method. This is executed when your pipeline
    # is run by CellProfiler.
    #
    # The workspace contains all of the state of your analysis. Later
    # tutorials will go over all of those pieces.
    #
    def run(self, workspace):
        pass
    
    