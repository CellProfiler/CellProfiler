'''<b>Example1e</b> demonstrates SettingsGroup
<hr>
There are many circumstances where it would be useful to let a user specify
an arbitrary number of a group of settings. For instance, you might want to
sum an arbitrary number of images together in your module or perform the
same operation on every listed image or object. This is done using
cps.SettingsGroup and by overriding the prepare_settings method.
'''

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

class Example1e(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Example1e"
    category = "Other"
    
    def create_settings(self):
        self.groups = []
        #
        # A hidden count setting saves the number of items in a list
        # to the pipeline. When you load the module, you can read the
        # hidden count and make sure your module has the correct number of
        # groups before the settings are transferred.
        #
        self.group_count = cps.HiddenCount(self.groups)
        #
        # A DoSomething setting looks like a button. When you press the button,
        # it does something: it calls the callback function which, in this case,
        # is "self.add_group()"
        #
        self.add_group_button = cps.DoSomething(
            "Add another group", "Add", self.add_group,
            doc = "Press this button to add another group")
        #
        # We call "add_group()" with a False argument. This tells add_group
        # to make sure and not add a button that would allow the user to
        # remove the first group
        #
        self.add_group(False)
        
    def add_group(self, can_delete = True):
        '''Add a group to the list of groups
        
        can_delete - if true, add a button that removes the entry
        '''
        #
        # Make a new settings group to hold the settings
        #
        group = cps.SettingsGroup()
        #
        # if you can delete, that means there's a setting above this one,
        # so it's nice to add a divider in that case.
        #
        if can_delete:
            group.append("divider", cps.Divider())
        #
        # Each instance has an addend and a multiplicand. We run through
        # them all, first adding, then multiplying to get our final answer
        #
        # group.append takes two arguments. The first is the name for the
        # attribute. In this case, we used "addend" so if you want to get
        # the addend, you say, "group.addend.value"
        #
        group.append("addend", cps.Float("Addend", 0))
        group.append("multiplicand", cps.Float("Multiplicand", 1))
        #
        # Only add the RemoveSettingButton if we can delete
        #
        if can_delete:
            group.append("remover", cps.RemoveSettingButton(
                "Remove this entry", "Remove",
                self.groups, group))
        #
        # Add the new group to the list.
        #
        self.groups.append(group)
        
    def settings(self):
        result = [self.group_count]
        #
        # Loop over all the elements in the group
        #
        for group in self.groups:
            assert isinstance(group, cps.SettingsGroup)
            #
            # Add the settings that go into the pipeline. SettingsGroup is
            # smart enough to know that DoSomething and Divider are UI elements
            # and don't go in the pipeline.
            #
            result += group.pipeline_settings()
        return result
    
    def visible_settings(self):
        #
        # Don't put in the hidden count...doh it's HIDDEN!
        result = []
        for group in self.groups:
            assert isinstance(group, cps.SettingsGroup)
            #
            # Add the visible settings for each group member
            #
            result += group.visible_settings()
        #
        # Put the add button at the end
        #
        result.append(self.add_group_button)
        return result
    #
    # by convention, "run" goes next.
    # Let's add up the values and print to the console
    #
    def run(self, workspace):
        accumulator = 0
        for group in self.groups:
            accumulator += group.addend.value
            accumulator += group.multiplicand.value
        #
        # You can put strings, numbers, lists, tuples, dictionaries and
        # numpy arrrays into workspace_display data as well as lists,
        # tuples and dictionaries of the above.
        #
        # The workspace will magically transfer these to itself when
        # display() is called - this might happen in a different process
        # or possibly on a different machine.
        #
        workspace.display_data.accumulator = accumulator
        
    def display(self, workspace, figure = None):
        #
        # We added the figure argument for the FileUI version. This is
        # a recipe for a display method that works with both.
        #
        if figure is None:
            #
            # In the old UI, you'd tell the workspace to make you a figure
            # or to find the old one.
            # subplots tells you how many subplots in the x and y directions
            #
            figure = workspace.create_or_find_figure(subplots = (1, 1))
        else:
            #
            # In the new UI, the figure is created for you and you set the
            # number of subplots like this
            #
            figure.set_subplots((1, 1))
        #
        # retrieve the accumulator value
        #
        accumulator = workspace.display_data.accumulator
        #
        # This is a Matplotlib Axes instance for you to draw on.
        # Google for Matplotlib's documentation on what super-special stuff
        # you can do with them. Also see examples - we do some special handing
        # for imshow which displays images on axes. I bet you didn't see that
        # coming ;-)
        axes = figure.subplot(0, 0)
        #
        # This keeps Matplotlib from drawing table on top of table.
        #
        axes.clear()
        #
        # We use a list into the table to organize it in columns. A header
        # and then the values.
        axes.table(cellText = [["Property", "Value"],
                               ["accumulator", str(accumulator)]], 
                   loc='center')
        #
        # You have to do this in order to get rid of the plot display
        #
        axes.set_frame_on(False)
        axes.set_axis_off()
        
    #
    # Finally, there's prepare_settings. setting_values are stored as unicode
    # strings (after a fashion) in your pipeline. Your module gets to have
    # as many as it returns from its settings() method. If you have settings
    # groups, you have to make sure that settings() returns the correct number
    # of settings... so you need to cheat and look at the setting values
    # before settings() is called so you can know how many to return.
    #
    # We thought about this a *little* and this loosey-goosey way of doing it
    # is typically pythonic in that it gives the implementor a lot of 
    # flexibility, but there's less structure to keep things from going wrong.
    #
    def prepare_settings(self, setting_values):
        # The first setting tells how many are present
        count = int(setting_values[0])
        #
        # Delete all but the first group. Python won't choke if there is
        # no self.groups[1]
        #
        del self.groups[1:]
        #
        # add count-1 new groups
        #
        for _ in range(1, count):
            self.add_group()
            