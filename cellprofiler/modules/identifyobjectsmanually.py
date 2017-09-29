# coding=utf-8

"""
IdentifyObjectsManually
=======================

**IdentifyObjectsManually** allows you to identify objects in an image
by hand rather than automatically.

This module lets you outline the objects in an image using the mouse.

The user interface has several mouse tools:

-  *Outline:* Lets you draw an outline around an object. Press the left
   mouse button at the start of the outline and draw the outline around
   your object. The tool will close your outline when you release the
   left mouse button.
-  *Zoom in:* Lets you draw a rectangle and zoom the display to within
   that rectangle.
-  *Zoom out:* Reverses the effect of the last zoom-in.
-  *Erase:* Erases an object if you click on it.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

"""

import numpy as np
from centrosome.cpmorphology import draw_line
from centrosome.cpmorphology import fill_labeled_holes
from centrosome.outline import outline

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.object as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.setting as cps
import identify as I

TOOL_OUTLINE = "Outline"
TOOL_ZOOM_IN = "Zoom in"
TOOL_ERASE = "Erase"


class IdentifyObjectsManually(I.Identify):
    category = "Object Processing"
    module_name = "IdentifyObjectsManually"
    variable_revision_number = 2

    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
                "Select the input image", cps.NONE, doc="""Choose the name of the image to display in the object selection user interface.""")

        self.objects_name = cps.ObjectNameProvider(
                "Name the objects to be identified", "Cells", doc="""\
What do you want to call the objects that you identify using this module? You can use this name to
refer to your objects in subsequent modules.""")

    def settings(self):
        return [
            self.image_name,
            self.objects_name
        ]

    def visible_settings(self):
        return [
            self.image_name,
            self.objects_name
        ]

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        '''This module cannot be used in a batch context'''
        raise ValueError("The IdentifyObjectsManually module cannot be run in batch mode")

    def run(self, workspace):
        image_name = self.image_name.value
        objects_name = self.objects_name.value
        image = workspace.image_set.get_image(image_name)
        pixel_data = image.pixel_data

        labels = workspace.interaction_request(
                self, pixel_data, workspace.measurements.image_set_number)
        if labels is None:
            # User cancelled. Soldier on as best we can.
            workspace.cancel_request()
            labels = np.zeros(pixel_data.shape[:2], int)
        objects = cpo.Objects()
        objects.segmented = labels
        workspace.object_set.add_objects(objects, objects_name)

        ##################
        #
        # Add measurements
        #
        m = workspace.measurements
        #
        # The object count
        #
        object_count = np.max(labels)
        I.add_object_count_measurements(m, objects_name, object_count)
        #
        # The object locations
        #
        I.add_object_location_measurements(m, objects_name, labels)

        workspace.display_data.labels = labels
        workspace.display_data.pixel_data = pixel_data

    def display(self, workspace, figure):
        objects_name = self.objects_name.value
        labels = workspace.display_data.labels
        pixel_data = workspace.display_data.pixel_data
        figure.set_subplots((1, 1))
        cplabels = [
            dict(name=objects_name, labels=[labels])]
        if pixel_data.ndim == 3:
            figure.subplot_imshow_color(
                    0, 0, pixel_data, title=objects_name, cplabels=cplabels)
        else:
            figure.subplot_imshow_grayscale(
                    0, 0, pixel_data, title=objects_name, cplabels=cplabels)

    def handle_interaction(self, pixel_data, image_set_number):
        '''Display a UI for editing'''
        from cellprofiler.gui.editobjectsdlg import EditObjectsDialog
        from wx import OK
        title = "%s #%d, image cycle #%d: " % (self.module_name,
                                               self.module_num,
                                               image_set_number)
        title += "Create, remove and edit %s. \n" % self.objects_name.value
        title += 'Press "F" to being freehand drawing.\n'
        title += "Click Help for full instructions."
        with EditObjectsDialog(
                pixel_data, [np.zeros(pixel_data.shape[:2], np.uint32)], False,
                title) as dialog_box:
            result = dialog_box.ShowModal()
            if result != OK:
                return None
            return dialog_box.labels[0]

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 2:
            image_name, object_name, max_resolution, save_outlines = setting_values
            wants_outlines = \
                (cps.YES if save_outlines.lower() == cps.DO_NOT_USE.lower()
                 else cps.NO)
            setting_values = [image_name, object_name, wants_outlines,
                              save_outlines]
            variable_revision_number = 1
            from_matlab = False

        if variable_revision_number == 1:
            setting_values = setting_values[:-2]

            variable_revision_number = 2

        return setting_values, variable_revision_number, from_matlab

    def get_measurement_columns(self, pipeline):
        '''Return database info on measurements made in module

        pipeline - pipeline being run

        Return a list of tuples of object name, measurement name and data type
        '''
        result = I.get_object_measurement_columns(self.objects_name.value)
        return result

    @property
    def measurement_dictionary(self):
        '''Return the dictionary to be used in get_object_categories/measurements

        Identify.get_object_categories and Identify.get_object_measurements
        use a dictionary to match against the objects produced. We
        return a dictionary whose only key is the object name and
        whose value (the parents) is an empty list.
        '''
        return {self.objects_name.value: []}

    def get_categories(self, pipeline, object_name):
        '''Return a list of categories of measurements made by this module

        pipeline - pipeline being run
        object_name - find categories of measurements made on this object
        '''
        return self.get_object_categories(pipeline, object_name,
                                          self.measurement_dictionary)

    def get_measurements(self, pipeline, object_name, category):
        '''Return a list of features measured on object & category

        pipeline - pipeline being run
        object_name - name of object being measured
        category - category of measurement being queried
        '''
        return self.get_object_measurements(pipeline, object_name, category,
                                            self.measurement_dictionary)
