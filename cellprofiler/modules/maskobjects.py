# coding=utf-8

"""
MaskObjects
===========

**MaskObjects** removes objects outside of a specified region or
regions.

This module allows you to delete the objects or portions of objects that
are outside of a region (mask) you specify. For example, after
identifying nuclei and tissue regions in previous **Identify** modules,
you might want to exclude all nuclei that are outside of a tissue
region.

If using a masking image, the mask is composed of the foreground (white
portions); if using a masking object, the mask is composed of the area
within the object. You can choose to remove only the portion of each
object that is outside of the region, remove the whole object if it is
partially or fully outside of the region, or retain the whole object
unless it is fully outside of the region.

|

============ ============
Supports 2D? Supports 3D?
============ ============
YES          NO
============ ============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Parent object measurements:**

-  *Count:* The number of new masked objects created from each parent
   object.

**Masked object measurements:**

-  *Parent:* The label number of the parent object.
-  *Location_X, Location_Y:* The pixel (X,Y) coordinates of the center
   of mass of the masked objects.
"""

import cellprofiler.measurement
import numpy as np
import scipy.ndimage as scind
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.outline import outline

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.setting as cps
import identify as I
from cellprofiler.setting import YES, NO

MC_OBJECTS = "Objects"
MC_IMAGE = "Image"

P_MASK = "Keep overlapping region"
P_REMOVE = "Remove"
P_KEEP = "Keep"
P_REMOVE_PERCENTAGE = "Remove depending on overlap"

R_RETAIN = "Retain"
R_RENUMBER = "Renumber"

# This dictionary is used by upgrade_settings to keep track of changes
# to the above names. If you change them, please put the text of the
# new names into the dictionary.
S_DICTIONARY = {
    "Objects": MC_OBJECTS,
    "Image": MC_IMAGE,
    "Keep overlapping region": P_MASK,
    "Remove": P_REMOVE,
    "Remove depending on overlap": P_REMOVE_PERCENTAGE,
    "Keep": P_KEEP,
    "Retain": R_RETAIN,
    "Renumber": R_RENUMBER
}


def s_lookup(x):
    '''Look up the current value for a setting choice w/backwards compatibility

    x - setting value from pipeline
    '''
    return S_DICTIONARY.get(x, x)


class MaskObjects(I.Identify):
    category = "Object Processing"
    module_name = "MaskObjects"
    variable_revision_number = 3

    def create_settings(self):
        '''Create the settings that control this module'''
        self.object_name = cps.ObjectNameSubscriber(
            "Select objects to be masked",
            cps.NONE,
            doc="""\
Select the objects that will be masked (that is, excluded in whole or in
part based on the other settings in the module). You can choose from any
objects created by a previous object processing module, such as
**IdentifyPrimaryObjects**, **IdentifySecondaryObjects** or
**IdentifyTertiaryObjects**.
"""
        )

        self.remaining_objects = cps.ObjectNameProvider(
            "Name the masked objects",
            "MaskedNuclei",
            doc="""\
Enter a name for the objects that remain after
the masking operation. You can refer to the masked objects in
subsequent modules by this name.
"""
        )

        self.mask_choice = cps.Choice(
            "Mask using a region defined by other objects or by binary image?",
            [
                MC_OBJECTS,
                MC_IMAGE
            ],
            doc="""\
You can mask your objects by defining a region using objects you
previously identified in your pipeline (*%(MC_OBJECTS)s*) or by
defining a region based on the white regions in a binary image
previously loaded or created in your pipeline (*%(MC_IMAGE)s*).
""" % globals()
        )

        self.masking_objects = cps.ObjectNameSubscriber(
            "Select the masking object",
            cps.NONE,
            doc="""\
*(Used only if mask is to be made from objects)*

Select the objects that will be used to define the masking region. You
can choose from any objects created by a previous object processing
module, such as **IdentifyPrimaryObjects**,
**IdentifySecondaryObjects**, or **IdentifyTertiaryObjects**.
"""
        )

        self.masking_image = cps.ImageNameSubscriber(
            "Select the masking image",
            cps.NONE,
            doc="""\
*(Used only if mask is to be made from an image)*

Select an image that was either loaded or created by a previous module.
The image should be a binary image where the white portion of the image
is the region(s) you will use for masking. Binary images can be loaded
from disk using the **NamesAndTypes** module by selecting “Binary mask”
for the image type. You can also create a binary image from a grayscale
image using **ApplyThreshold**.
"""
        )

        self.wants_inverted_mask = cps.Binary(
            "Invert the mask?",
            False,
            doc="""\
This option reverses the foreground/background relationship of the mask.

-  Select "*%(NO)s*" for the mask to be composed of the foregound (white
   portion) of the masking image or the area within the masking objects.
-  Select "*%(YES)s*" for the mask to instead be composed of the
   *background* (black portions) of the masking image or the area
   *outside* the masking objects.
   """ % globals()
        )

        self.overlap_choice = cps.Choice(
            "Handling of objects that are partially masked",
            [
                P_MASK,
                P_KEEP,
                P_REMOVE,
                P_REMOVE_PERCENTAGE
            ],
            doc="""\
An object might partially overlap the mask region, with pixels both
inside and outside the region. **MaskObjects** can handle this in one
of three ways:

-  *%(P_MASK)s:* Choosing this option will reduce the size of partially
   overlapping objects. The part of the object that overlaps the masking
   region will be retained. The part of the object that is outside of the
   masking region will be removed.
-  *%(P_KEEP)s:* If you choose this option, **MaskObjects** will keep
   the whole object if any part of it overlaps the masking region.
-  *%(P_REMOVE)s:* Objects that are partially outside of the masking
   region will be completely removed if you choose this option.
-  *%(P_REMOVE_PERCENTAGE)s:* Determine whether to remove or keep an
   object depending on how much of the object overlaps the masking
   region. **MaskObjects** will keep an object if at least a certain
   fraction (which you enter below) of the object falls within the
   masking region. **MaskObjects** completely removes the object if too
   little of it overlaps the masking region.""" % globals()
        )

        self.overlap_fraction = cps.Float(
            "Fraction of object that must overlap",
            .5,
            minval=0,
            maxval=1,
            doc="""\
*(Used only if removing based on overlap)*

Specify the minimum fraction of an object that must overlap the masking
region for that object to be retained. For instance, if the fraction is
0.75, then 3/4 of an object must be within the masking region for that
object to be retained.
"""
        )

        self.retain_or_renumber = cps.Choice(
            "Numbering of resulting objects",
            [
                R_RENUMBER,
                R_RETAIN
            ],
            doc="""\
Choose how to number the objects that remain after masking, which
controls how remaining objects are associated with their predecessors:

-  *%(R_RENUMBER)s:* The objects that remain will be renumbered using
   consecutive numbers. This is a good choice if you do not plan to use
   measurements from the original objects; your object measurements for
   the masked objects will not have gaps (where removed objects are
   missing).
-  *%(R_RETAIN)s:* The original labels for the objects will be
   retained. This allows any measurements you make from the masked
   objects to be directly aligned with measurements you might have made
   of the original, unmasked objects (or objects directly associated
   with them).
""" % globals()
        )

    def settings(self):
        '''The settings as they appear in the pipeline'''
        return [
            self.object_name,
            self.remaining_objects,
            self.mask_choice,
            self.masking_objects,
            self.masking_image,
            self.overlap_choice,
            self.overlap_fraction,
            self.retain_or_renumber,
            self.wants_inverted_mask
        ]

    def help_settings(self):
        '''The settings as they appear in the pipeline'''
        return [self.object_name, self.remaining_objects, self.mask_choice,
                self.masking_objects, self.masking_image,
                self.wants_inverted_mask,
                self.overlap_choice, self.overlap_fraction, self.retain_or_renumber]

    def visible_settings(self):
        '''The settings as they appear in the UI'''
        result = [self.object_name, self.remaining_objects, self.mask_choice,
                  self.masking_image if self.mask_choice == MC_IMAGE
                  else self.masking_objects, self.wants_inverted_mask,
                  self.overlap_choice]

        if self.overlap_choice == P_REMOVE_PERCENTAGE:
            result += [self.overlap_fraction]

        result += [self.retain_or_renumber]

        return result

    def run(self, workspace):
        '''Run the module on an image set'''

        object_name = self.object_name.value
        remaining_object_name = self.remaining_objects.value
        original_objects = workspace.object_set.get_objects(object_name)

        if self.mask_choice == MC_IMAGE:
            mask = workspace.image_set.get_image(self.masking_image.value,
                                                 must_be_binary=True)
            mask = mask.pixel_data
        else:
            masking_objects = workspace.object_set.get_objects(
                    self.masking_objects.value)
            mask = masking_objects.segmented > 0
        if self.wants_inverted_mask:
            mask = ~mask
        #
        # Load the labels
        #
        labels = original_objects.segmented.copy()
        nobjects = np.max(labels)
        #
        # Resize the mask to cover the objects
        #
        mask, m1 = cpo.size_similarly(labels, mask)
        mask[~m1] = False
        #
        # Apply the mask according to the overlap choice.
        #
        if nobjects == 0:
            pass
        elif self.overlap_choice == P_MASK:
            labels = labels * mask
        else:
            pixel_counts = fix(scind.sum(mask, labels,
                                         np.arange(1, nobjects + 1, dtype=np.int32)))
            if self.overlap_choice == P_KEEP:
                keep = pixel_counts > 0
            else:
                total_pixels = fix(scind.sum(np.ones(labels.shape), labels,
                                             np.arange(1, nobjects + 1, dtype=np.int32)))
                if self.overlap_choice == P_REMOVE:
                    keep = pixel_counts == total_pixels
                elif self.overlap_choice == P_REMOVE_PERCENTAGE:
                    fraction = self.overlap_fraction.value
                    keep = pixel_counts / total_pixels >= fraction
                else:
                    raise NotImplementedError("Unknown overlap-handling choice: %s",
                                              self.overlap_choice.value)
            keep = np.hstack(([False], keep))
            labels[~ keep[labels]] = 0
        #
        # Renumber the labels matrix if requested
        #
        if self.retain_or_renumber == R_RENUMBER:
            unique_labels = np.unique(labels[labels != 0])
            indexer = np.zeros(nobjects + 1, int)
            indexer[unique_labels] = np.arange(1, len(unique_labels) + 1)
            labels = indexer[labels]
            parent_objects = unique_labels
        else:
            parent_objects = np.arange(1, nobjects + 1)
        #
        # Add the objects
        #
        remaining_objects = cpo.Objects()
        remaining_objects.segmented = labels
        remaining_objects.unedited_segmented = original_objects.unedited_segmented
        workspace.object_set.add_objects(remaining_objects,
                                         remaining_object_name)
        #
        # Add measurements
        #
        m = workspace.measurements
        m.add_measurement(remaining_object_name,
                          cellprofiler.measurement.FF_PARENT % object_name,
                          parent_objects)
        if np.max(original_objects.segmented) == 0:
            child_count = np.array([], int)
        else:
            child_count = fix(scind.sum(labels, original_objects.segmented,
                                        np.arange(1, nobjects + 1, dtype=np.int32)))
            child_count = (child_count > 0).astype(int)
        m.add_measurement(object_name,
                          cellprofiler.measurement.FF_CHILDREN_COUNT % remaining_object_name,
                          child_count)
        if self.retain_or_renumber == R_RETAIN:
            remaining_object_count = nobjects
        else:
            remaining_object_count = len(unique_labels)
        I.add_object_count_measurements(m, remaining_object_name,
                                        remaining_object_count)
        I.add_object_location_measurements(m, remaining_object_name, labels)
        #
        # Save the input, mask and output images for display
        #
        if self.show_window:
            workspace.display_data.original_labels = original_objects.segmented
            workspace.display_data.final_labels = labels
            workspace.display_data.mask = mask

    def display(self, workspace, figure):
        '''Create an informative display for the module'''
        import matplotlib
        from cellprofiler.gui.tools import renumber_labels_for_display
        original_labels = workspace.display_data.original_labels
        final_labels = workspace.display_data.final_labels
        mask = workspace.display_data.mask
        #
        # Create a composition of the final labels and mask
        #
        final_labels = renumber_labels_for_display(final_labels)
        outlines = outline(original_labels) > 0

        cm = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())
        sm = matplotlib.cm.ScalarMappable(cmap=cm)
        #
        # Paint the labels in color
        #
        image = sm.to_rgba(final_labels)[:, :, :3]
        image[final_labels == 0, :] = 0
        #
        # Make the mask a dark gray
        #
        image[(final_labels == 0) & mask, :] = .25
        #
        # Make the outlines of the kept objects the primary color
        # and the outlines of removed objects red.
        #
        final_outlines = outline(final_labels) > 0
        original_color = np.array(cpprefs.get_secondary_outline_color(), float) / 255
        final_color = np.array(cpprefs.get_primary_outline_color(), float) / 255
        image[outlines, :] = original_color[np.newaxis, :]
        image[final_outlines, :] = final_color[np.newaxis, :]

        figure.set_subplots((2, 1))
        figure.subplot_imshow_labels(0, 0, original_labels,
                                     title=self.object_name.value)
        figure.subplot_imshow_color(1, 0, image,
                                    title=self.remaining_objects.value,
                                    sharexy=figure.subplot(0, 0))

    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''

        object_name = self.object_name.value
        remaining_object_name = self.remaining_objects.value
        columns = I.get_object_measurement_columns(self.remaining_objects.value)
        columns += [(object_name, cellprofiler.measurement.FF_CHILDREN_COUNT % remaining_object_name,
                     cpmeas.COLTYPE_INTEGER),
                    (remaining_object_name, cellprofiler.measurement.FF_PARENT % object_name,
                     cpmeas.COLTYPE_INTEGER)]
        return columns

    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        """

        object_dictionary = self.get_object_dictionary()
        return self.get_object_categories(pipeline, object_name,
                                          object_dictionary)

    def get_object_dictionary(self):
        '''Get the dictionary of parent child relationships

        see Identify.get_object_categories, Identify.get_object_measurements
        '''
        object_dictionary = {
            self.remaining_objects.value: [self.object_name.value]
        }
        return object_dictionary

    def get_measurements(self, pipeline, object_name, category):
        '''Return names of the measurements made by this module

        pipeline - pipeline being run
        object_name - object being measured (or Image)
        category - category of measurement, for instance, "Location"
        '''
        return self.get_object_measurements(pipeline, object_name, category,
                                            self.get_object_dictionary())

    def validate_module(self, pipeline):
        """Bypass Identify.validate_module"""
        pass

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 2:
            object_name, mask_region_name, remaining_object_name, \
            renumber, save_outlines, remove_overlapping = setting_values
            wants_outlines = (cps.NO if save_outlines.lower() ==
                                        cps.DO_NOT_USE.lower() else cps.YES)
            renumber = (R_RENUMBER if renumber == "Renumber"
                        else R_RETAIN if renumber == "Retain"
            else renumber)
            overlap_choice = (P_MASK if remove_overlapping == "Retain"
                              else P_REMOVE if remove_overlapping == "Remove"
            else remove_overlapping)

            setting_values = [
                object_name, remaining_object_name, MC_OBJECTS,
                mask_region_name, mask_region_name, overlap_choice,
                ".5", renumber, wants_outlines, save_outlines]
            from_matlab = False
            variable_revision_number = 1

        if variable_revision_number == 1 and not from_matlab:
            # Added "wants_inverted_mask"
            setting_values = setting_values + [cps.NO]
            variable_revision_number = 2

        if variable_revision_number == 2:
            setting_values = setting_values[:-3] + setting_values[-1:]

            variable_revision_number = 3

        setting_values = list(setting_values)
        setting_values[5] = s_lookup(setting_values[5])
        setting_values[7] = s_lookup(setting_values[7])
        return setting_values, variable_revision_number, from_matlab
