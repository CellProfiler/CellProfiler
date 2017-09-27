# coding=utf-8

import cellprofiler.measurement
import matplotlib
import matplotlib.cm
import numpy as np
from centrosome.outline import outline

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.setting as cps
import identify as cpmi
from cellprofiler.setting import YES, NO
import _help

__doc__ = """\
IdentifyTertiaryObjects
=======================

**IdentifyTertiaryObjects** identifies tertiary objects (e.g.,
cytoplasm) by removing smaller primary objects (e.g., nuclei) from larger
secondary objects (e.g., cells), leaving a ring shape.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also **IdentifyPrimaryObjects** and **IdentifySecondaryObjects**
modules.

What is a tertiary object?
^^^^^^^^^^^^^^^^^^^^^^^^^^

{DEFINITION_OBJECT}

We define an
object as *tertiary* when it is identified using prior primary and
secondary objects.

As an example, you can find nuclei using **IdentifyPrimaryObjects** and
cell bodies using **IdentifySecondaryObjects**. Use the
**IdentifyTertiaryObjects** module to define the
cytoplasm, the region outside the nucleus but within the cell body, as a
new object which can be measured in downstream **Measure** modules.

What do I need as input?
^^^^^^^^^^^^^^^^^^^^^^^^

This module will take the smaller identified objects and remove them
from the larger identified objects. For example, “subtracting” the
nuclei from the cells will leave just the cytoplasm, the properties of
which can then be measured by downstream **Measure** modules. The larger
objects should therefore be equal in size or larger than the smaller
objects and must completely contain the smaller objects;
**IdentifySecondaryObjects** will produce objects that satisfy this
constraint. Ideally, both inputs should be objects produced by prior
**Identify** modules.

What do I get as output?
^^^^^^^^^^^^^^^^^^^^^^^^

A set of objects are produced by this module, which can be used
in downstream modules for measurement purposes or other operations.
Because each tertiary object is produced from primary and secondary
objects, there will always be at most one tertiary object for each
larger object. See the section "Measurements made by this module" below for
the measurements that are produced by this module.

Note that if the smaller objects are not completely contained within the
larger objects, creating subregions using this module can result in objects
with a single label (that is, identity) that nonetheless are not contiguous.
This may lead
to unexpected results when running measurement modules such as
**MeasureObjectSizeShape** because calculations of the perimeter, aspect
ratio, solidity, etc. typically make sense only for contiguous objects.
Other modules, such as **MeasureImageIntensity**, are not affected and
will yield expected results.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Image measurements:**

-  *Count:* The number of tertiary objects identified.

**Object measurements:**

-  *Parent:* The identity of the primary object and secondary object
   associated with each tertiary object.

-  *Location\_X, Location\_Y:* The pixel (X,Y) coordinates of the center
   of mass of the identified tertiary objects.

""".format(**{
    "DEFINITION_OBJECT": _help.DEFINITION_OBJECT
})

'''The parent object relationship points to the secondary / larger objects'''
R_PARENT = "Parent"
'''The removed object relationship points to the primary / smaller objects'''
R_REMOVED = "Removed"


class IdentifyTertiaryObjects(cpm.Module):
    module_name = "IdentifyTertiaryObjects"
    variable_revision_number = 3
    category = "Object Processing"

    def create_settings(self):
        """Create the settings for the module

        Create the settings for the module during initialization.
        """
        self.secondary_objects_name = cps.ObjectNameSubscriber(
                "Select the larger identified objects", cps.NONE, doc="""\
Select the larger identified objects. This will usually be an object
previously identified by an **IdentifySecondaryObjects** module.""")

        self.primary_objects_name = cps.ObjectNameSubscriber(
                "Select the smaller identified objects", cps.NONE, doc="""\
Select the smaller identified objects. This will usually be an object
previously identified by an **IdentifyPrimaryObjects** module.""")

        self.subregion_objects_name = cps.ObjectNameProvider(
                "Name the tertiary objects to be identified", "Cytoplasm", doc="""\
Enter a name for the new tertiary objects. The tertiary objects
will consist of the smaller object subtracted from the larger object.""")

        self.shrink_primary = cps.Binary(
                "Shrink smaller object prior to subtraction?", True, doc="""\
Select *%(YES)s* to shrink the smaller objects by 1 pixel before
subtracting them from the larger objects. this approach will ensure that
there is always a tertiary object produced, even if it is only 1 pixel wide.
If you need alternate amounts of shrinking, use the **ExpandOrShrink**
module prior to **IdentifyTertiaryObjects**.

Select *%(NO)s* to subtract the objects directly, which will ensure that
no pixels are shared between the primary/secondary/tertiary objects and
hence measurements for all three sets of objects will not use the same
pixels multiple times. However, this may result in the creation of
objects with no area. Measurements can still be made on such objects,
but the results will be zero or not-a-number (NaN).
""" % globals())

    def settings(self):
        return [
            self.secondary_objects_name,
            self.primary_objects_name,
            self.subregion_objects_name,
            self.shrink_primary
        ]

    def visible_settings(self):
        return [
            self.secondary_objects_name,
            self.primary_objects_name,
            self.subregion_objects_name,
            self.shrink_primary
        ]

    def run(self, workspace):
        """Run the module on the current data set

        workspace - has the current image set, object set, measurements
                    and the parent frame for the application if the module
                    is allowed to display. If the module should not display,
                    workspace.frame is None.
        """
        #
        # The object set holds "objects". Each of these is a container
        # for holding up to three kinds of image labels.
        #
        object_set = workspace.object_set
        #
        # Get the primary objects (the centers to be removed).
        # Get the string value out of primary_object_name.
        #
        primary_objects = object_set.get_objects(self.primary_objects_name.value)
        #
        # Get the cleaned-up labels image
        #
        primary_labels = primary_objects.segmented
        #
        # Do the same with the secondary object
        secondary_objects = object_set.get_objects(self.secondary_objects_name.value)
        secondary_labels = secondary_objects.segmented
        #
        # If one of the two label images is smaller than the other, we
        # try to find the cropping mask and we apply that mask to the larger
        #
        try:
            if any([p_size < s_size
                    for p_size, s_size
                    in zip(primary_labels.shape, secondary_labels.shape)]):
                #
                # Look for a cropping mask associated with the primary_labels
                # and apply that mask to resize the secondary labels
                #
                secondary_labels = primary_objects.crop_image_similarly(secondary_labels)
                tertiary_image = primary_objects.parent_image
            elif any([p_size > s_size
                      for p_size, s_size
                      in zip(primary_labels.shape, secondary_labels.shape)]):
                primary_labels = secondary_objects.crop_image_similarly(primary_labels)
                tertiary_image = secondary_objects.parent_image
            elif secondary_objects.parent_image is not None:
                tertiary_image = secondary_objects.parent_image
            else:
                tertiary_image = primary_objects.parent_image
        except ValueError:
            # No suitable cropping - resize all to fit the secondary
            # labels which are the most critical.
            #
            primary_labels, _ = cpo.size_similarly(secondary_labels, primary_labels)
            if secondary_objects.parent_image is not None:
                tertiary_image = secondary_objects.parent_image
            else:
                tertiary_image = primary_objects.parent_image
                if tertiary_image is not None:
                    tertiary_image, _ = cpo.size_similarly(secondary_labels, tertiary_image)
        #
        # Find the outlines of the primary image and use this to shrink the
        # primary image by one. This guarantees that there is something left
        # of the secondary image after subtraction
        #
        primary_outline = outline(primary_labels)
        tertiary_labels = secondary_labels.copy()
        if self.shrink_primary:
            primary_mask = np.logical_or(primary_labels == 0,
                                         primary_outline)
        else:
            primary_mask = primary_labels == 0
        tertiary_labels[primary_mask == False] = 0
        #
        # Get the outlines of the tertiary image
        #
        tertiary_outlines = outline(tertiary_labels) != 0
        #
        # Make the tertiary objects container
        #
        tertiary_objects = cpo.Objects()
        tertiary_objects.segmented = tertiary_labels
        tertiary_objects.parent_image = tertiary_image
        #
        # Relate tertiary objects to their parents & record
        #
        child_count_of_secondary, secondary_parents = \
            secondary_objects.relate_children(tertiary_objects)
        if self.shrink_primary:
            child_count_of_primary, primary_parents = \
                primary_objects.relate_children(tertiary_objects)
        else:
            # Primary and tertiary don't overlap.
            # Establish overlap between primary and secondary and commute
            _, secondary_of_primary = \
                secondary_objects.relate_children(primary_objects)
            mask = secondary_of_primary != 0
            child_count_of_primary = np.zeros(mask.shape, int)
            child_count_of_primary[mask] = child_count_of_secondary[
                secondary_of_primary[mask] - 1]
            primary_parents = np.zeros(secondary_parents.shape,
                                       secondary_parents.dtype)
            primary_of_secondary = np.zeros(secondary_objects.count + 1, int)
            primary_of_secondary[secondary_of_primary] = \
                np.arange(1, len(secondary_of_primary) + 1)
            primary_of_secondary[0] = 0
            primary_parents = primary_of_secondary[secondary_parents]
        #
        # Write out the objects
        #
        workspace.object_set.add_objects(tertiary_objects,
                                         self.subregion_objects_name.value)
        #
        # Write out the measurements
        #
        m = workspace.measurements
        #
        # The parent/child associations
        #
        for parent_objects_name, parents_of, child_count, relationship in (
                (self.primary_objects_name, primary_parents,
                 child_count_of_primary, R_REMOVED),
                (self.secondary_objects_name, secondary_parents,
                 child_count_of_secondary, R_PARENT)):
            m.add_measurement(self.subregion_objects_name.value,
                              cellprofiler.measurement.FF_PARENT % parent_objects_name.value,
                              parents_of)
            m.add_measurement(parent_objects_name.value,
                              cellprofiler.measurement.FF_CHILDREN_COUNT % self.subregion_objects_name.value,
                              child_count)
            mask = parents_of != 0
            image_number = np.ones(np.sum(mask), int) * m.image_set_number
            child_object_number = np.argwhere(mask).flatten() + 1
            parent_object_number = parents_of[mask]
            m.add_relate_measurement(
                    self.module_num, relationship,
                    parent_objects_name.value, self.subregion_objects_name.value,
                    image_number, parent_object_number,
                    image_number, child_object_number)

        object_count = tertiary_objects.count
        #
        # The object count
        #
        cpmi.add_object_count_measurements(workspace.measurements,
                                           self.subregion_objects_name.value,
                                           object_count)
        #
        # The object locations
        #
        cpmi.add_object_location_measurements(workspace.measurements,
                                              self.subregion_objects_name.value,
                                              tertiary_labels)

        if self.show_window:
            workspace.display_data.primary_labels = primary_labels
            workspace.display_data.secondary_labels = secondary_labels
            workspace.display_data.tertiary_labels = tertiary_labels
            workspace.display_data.tertiary_outlines = tertiary_outlines

    def display(self, workspace, figure):
        primary_labels = workspace.display_data.primary_labels
        secondary_labels = workspace.display_data.secondary_labels
        tertiary_labels = workspace.display_data.tertiary_labels
        tertiary_outlines = workspace.display_data.tertiary_outlines
        #
        # Draw the primary, secondary and tertiary labels
        # and the outlines
        #
        figure.set_subplots((2, 2))

        figure.subplot_imshow_labels(0, 0, primary_labels,
                                     self.primary_objects_name.value)
        figure.subplot_imshow_labels(1, 0, secondary_labels,
                                     self.secondary_objects_name.value,
                                     sharexy=figure.subplot(0, 0))
        figure.subplot_imshow_labels(0, 1, tertiary_labels,
                                     self.subregion_objects_name.value,
                                     sharexy=figure.subplot(0, 0))
        figure.subplot_imshow_bw(1, 1, tertiary_outlines,
                                 "Outlines",
                                 sharexy=figure.subplot(0, 0))

    def is_object_identification_module(self):
        '''IdentifyTertiaryObjects makes tertiary objects sets so it's a identification module'''
        return True

    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        subregion_name = self.subregion_objects_name.value
        columns = cpmi.get_object_measurement_columns(subregion_name)
        for parent in (self.primary_objects_name.value,
                       self.secondary_objects_name.value):
            columns += [(parent,
                         cellprofiler.measurement.FF_CHILDREN_COUNT % subregion_name,
                         cpmeas.COLTYPE_INTEGER),
                        (subregion_name,
                         cellprofiler.measurement.FF_PARENT % parent,
                         cpmeas.COLTYPE_INTEGER)]
        return columns

    def upgrade_settings(self,
                         setting_values,
                         variable_revision_number,
                         module_name,
                         from_matlab):
        """Adjust the setting values to make old pipelines compatible with new

        This function allows the caller to adjust the setting_values
        (which are the text representation of the values of the settings)
        based on the variable_revision_number, the name of the module
        used to save the values (if two modules' functions were merged)
        and whether the values were saved by the Matlab or Python version
        of the module.

        setting_values - a list of string setting values in the order
                         specified by the "settings" function
        variable_revision_number - the variable revision number at the time
                                   of saving
        module_name - the name of the module that did the saving
        from_matlab - True if the matlab version of the module did the saving,
                      False if a Python module did the saving

        returns the modified setting_values, the corrected
                variable_revision_number and the corrected from_matlab flag
        """

        if from_matlab and variable_revision_number == 1:
            new_setting_values = list(setting_values)
            #
            # if the Matlab outlines name was "Do not use", turn
            # use_outlines off, otherwise turn it on
            #
            if new_setting_values[3] == cps.DO_NOT_USE:
                # The text value, "No", sets use_outlines to False
                new_setting_values.append(cps.NO)
            else:
                # The text value, "Yes", sets use_outlines to True
                new_setting_values.append(cps.YES)
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 1

        if (not from_matlab) and variable_revision_number == 1:
            setting_values = setting_values + [cps.YES]
            variable_revision_number = 2

        if variable_revision_number == 2:
            setting_values = setting_values[:3] + setting_values[5:]

            variable_revision_number = 3

        return setting_values, variable_revision_number, from_matlab

    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        categories = []
        if object_name == cpmeas.IMAGE:
            categories += ["Count"]
        elif (object_name == self.primary_objects_name or
                      object_name == self.secondary_objects_name):
            categories.append("Children")
        if object_name == self.subregion_objects_name:
            categories += ("Parent", "Location", "Number")
        return categories

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        result = []

        if object_name == cpmeas.IMAGE:
            if category == "Count":
                result += [self.subregion_objects_name.value]
        if (object_name in
                (self.primary_objects_name.value, self.secondary_objects_name.value)
            and category == "Children"):
            result += ["%s_Count" % self.subregion_objects_name.value]
        if object_name == self.subregion_objects_name:
            if category == "Location":
                result += ["Center_X", "Center_Y"]
            elif category == "Parent":
                result += [self.primary_objects_name.value,
                           self.secondary_objects_name.value]
            elif category == "Number":
                result += ["Object_Number"]
        return result


IdentifyTertiarySubregion = IdentifyTertiaryObjects
