'''<b>MaskObjects</b> removes objects outside of specified region.
<hr>
This image analysis module allows you to delete the objects and portions
of objects that are outside of a region you specify (e.g. nuclei outside
of a tissue region).  The objects and the region should both result from
any <b>Identify</b> module (Primary, Secondary, or Tertiary). Alternatively, you
can use a binary image as a mask and remove objects outside of the mask.
You can transform a grayscale image into a binary image using the
<b>ApplyThreshold</b> module.<p>

You can choose to remove only the portions of objects that are outside of
your region or mask, remove the whole object if it is partially or fully
outside of the region or retain the whole object unless it is fully outside
of the region.
'''
__version__="$Revision$"
import numpy as np
import scipy.ndimage as scind

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import identify as I
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.cpmath.outline import outline
from cellprofiler.gui.cpfigure import renumber_labels_for_display
import cellprofiler.preferences as cpprefs

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
    variable_revision_number = 1
    
    def create_settings(self):
        '''Create the settings that control this module'''
        self.object_name = cps.ObjectNameSubscriber(
            "Select objects to be masked","None",
            doc="""These are the objects that will be masked by the masking
            objects or image. You can choose from any objects created by
            a previous module, such as <b>IdentifyPrimaryObjects</b>,
            <b>IdentifySecondaryObjects</b> or <b>IdentifyTertiaryObjects</b>""")
        
        self.remaining_objects = cps.ObjectNameProvider(
            "Name the masked objects", "MaskedNuclei",
            doc="""This setting gives a name to the objects that remain after
            the masking operation. You can refer to the masked objects in
            subsequent modules using this name.""")
        
        self.mask_choice = cps.Choice(
            "Mask using other objects or binary image?",
            [MC_OBJECTS, MC_IMAGE],
            doc="""You can either mask your objects using some other objects
            or you can mask them using a binary image.""")
        
        self.masking_objects = cps.ObjectNameSubscriber(
            "Select the masking object","None",
            doc="""These are the objects that will be used to define the
            masking region. Objects that do not overlap the masking objects
            will be masked out. You can choose from any objects created
            by a previous module, such as <b>IdentifyPrimaryObjects</b>""")
        
        self.masking_image = cps.ImageNameSubscriber(
            "Select the masking image","None",
            doc="""This is the name of an image that was either loaded or
            created by a previous module. The image should be a binary image.
            Images can be loaded from disk by <b>LoadImages</b> or
            <b>LoadData</b>. You can create a binary image from a grayscale
            image using <b>ApplyThreshold</b> and can modify that image
            using the <b>Morph</b> module. Objects that do not overlap
            the masking image will be masked out.""")
        
        self.overlap_choice = cps.Choice(
            "Handling of objects that are partially masked",
            [P_MASK, P_KEEP, P_REMOVE, P_REMOVE_PERCENTAGE],
            doc="""An object can partially overlap the mask region with
            pixels both inside and outside the region. <b>MaskObjects</b>
            can handle this in one of three ways:<br>
            <ul><li><i>Keep overlapping region</i> - choosing this option
            will reduce the size of partially overlapping objects. The part
            of the object that overlaps the region will be retained. The
            part of the object that is outside of the region will be removed.
            </li><li><i>Remove</i> - objects that are partially outside
            of the masking region will be completely removed if you choose
            this option.</li>
            <li><i>Remove depending on overlap</i> - determine whether to
            remove or keep an object depending on how much of the object
            overlaps the masking region. <b>MaskObjects</b> will keep an
            object if at least a certain fraction (which you enter below) of
            the object falls within the masking region. <b>MaskObjects</b>
            completely removes the object if too little of it overlaps
            the masking region.</li>
            <li><i>Keep</i> - if you choose this option, <b>MaskObjects</b> 
            will keep the whole object if any part of it overlaps the masking
            region.</li></ul>""")
        
        self.overlap_fraction = cps.Float(
            "Fraction of object that must overlap", .5, 
            minval = 0, maxval = 1,
            doc = """This setting specifies the minimum fraction of an object
            that must overlap the masking region for that object to be retained.
            For instance, if the fraction is 0.75, then 3/4 of an object
            must be within the masking region for that object to be retained.""")
        
        self.retain_or_renumber = cps.Choice(
            "Numbering of resulting objects",
            [R_RENUMBER, R_RETAIN],
            doc="""This setting controls the numbering of the objects that 
            remain after masking. If you choose, "Renumber",
            <b>MaskObjects</b> will number the objects that remain 
            using consecutive numbers. If you choose, "Retain",
            <b>MaskObjects</b> will retain the original numbers. Renumbering
            is a good choice if you will not use measurements from the
            original unmasked objects; your object measurements for the
            masked objects won't have gaps. Retaining makes it eaiser to
            analyze object measurement data if you want to measure objects
            associated with your original objects.""")

        self.wants_outlines = cps.Binary(
            "Save outlines of the resulting objects?", False,
            doc = """You can save the outlines of the objects that are
            produced after masking. These outlines can be displayed over
            images using the <b>OverlayOutlines</b> module.""")
        
        self.outlines_name = cps.OutlineNameProvider(
            "Name the outline image", "MaskedOutlines",
            doc = """This setting names the outline image. Subsequent modules
            can refer to the binary outline image using this name.""")
        
    def settings(self):
        '''The settings as they appear in the pipeline'''
        return [self.object_name, self.remaining_objects, self.mask_choice,
                self.masking_objects, self.masking_image, self.overlap_choice,
                self.overlap_fraction, self.retain_or_renumber, 
                self.wants_outlines, self.outlines_name]
    
    def visible_settings(self):
        '''The settings as they appear in the UI'''
        result = [self.object_name, self.remaining_objects, self.mask_choice,
                  self.masking_image if self.mask_choice == MC_IMAGE
                  else self.masking_objects, self.overlap_choice]

        if self.overlap_choice == P_REMOVE_PERCENTAGE:
            result += [self.overlap_fraction]
        
        result += [self.retain_or_renumber, self.wants_outlines]
        if self.wants_outlines.value:
            result += [self.outlines_name]
        return result
    
    def is_interactive(self):
        '''Tell CP that this module does not have an interactive UI'''
        return False
    
    def run(self, workspace):
        '''Run the module on an image set'''
        
        object_name = self.object_name.value
        remaining_object_name = self.remaining_objects.value
        original_objects = workspace.object_set.get_objects(object_name)
        
        if self.mask_choice == MC_IMAGE:
            mask = workspace.image_set.get_image(self.masking_image.value,
                                                 must_be_binary = True)
            mask = mask.pixel_data
        else:
            masking_objects = workspace.object_set.get_objects(
                self.masking_objects.value)
            mask = masking_objects.segmented > 0
        #
        # Apply the mask according to the overlap choice.
        #
        labels = original_objects.segmented.copy()
        nobjects = np.max(labels)
        if nobjects == 0:
            pass
        elif self.overlap_choice == P_MASK:
            labels = labels * mask
        else:
            pixel_counts = fix(scind.sum(mask, labels, 
                                         np.arange(1, nobjects+1)))
            if self.overlap_choice == P_KEEP:
                keep = pixel_counts > 0
            else:
                total_pixels = fix(scind.sum(np.ones(labels.shape), labels,
                                             np.arange(1, nobjects+1)))
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
            unique_labels = np.unique(labels[labels!=0])
            indexer = np.zeros(nobjects+1, int)
            indexer[unique_labels] = np.arange(1, len(unique_labels)+1)
            labels = indexer[labels]
            parent_objects = unique_labels
        else:
            parent_objects = np.arange(1, nobjects+1)
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
                          I.FF_PARENT % object_name,
                          parent_objects)
        if np.max(original_objects.segmented) == 0:
            child_count = np.array([],int)
        else:
            child_count = fix(scind.sum(labels, original_objects.segmented,
                                        np.arange(1, nobjects+1)))
            child_count = (child_count > 0).astype(int)
        m.add_measurement(object_name,
                          I.FF_CHILDREN_COUNT % remaining_object_name,
                          child_count)
        if self.retain_or_renumber == R_RETAIN:
            remaining_object_count = nobjects
        else:
            remaining_object_count = len(unique_labels)
        I.add_object_count_measurements(m, remaining_object_name,
                                        remaining_object_count)
        I.add_object_location_measurements(m, remaining_object_name, labels)
        #
        # Add an outline if asked to do so
        #
        if self.wants_outlines.value:
            outline_image = cpi.Image(outline(labels) > 0,
                                      parent_image = original_objects.parent_image)
            workspace.image_set.add(self.outlines_name.value, outline_image)
        #
        # Save the input, mask and output images for display
        #
        if workspace.frame is not None:
            workspace.display_data.original_labels = original_objects.segmented
            workspace.display_data.final_labels = labels
            workspace.display_data.mask = mask
            
    def display(self, workspace):
        '''Create an informative display for the module'''
        import matplotlib
        original_labels = workspace.display_data.original_labels
        final_labels = workspace.display_data.final_labels
        mask = workspace.display_data.mask
        #
        # Create a composition of the final labels and mask
        #
        final_labels = renumber_labels_for_display(final_labels)
        outlines = outline(original_labels) > 0
        
        cm = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())
        sm = matplotlib.cm.ScalarMappable(cmap = cm)
        #
        # Paint the labels in color
        #
        image = sm.to_rgba(final_labels)[:,:,:3]
        image[final_labels == 0,:] = 0
        #
        # Make the mask a dark gray
        #
        image[(final_labels == 0) & mask,:] = .25
        #
        # Make the outlines of the original objects red
        #
        image[outlines,0] = 1
        image[outlines,1] = 0
        image[outlines,2] = 0
        
        figure = workspace.create_or_find_figure(subplots=(2,1))
        figure.subplot_imshow_labels(0, 0, original_labels,
                                     title = self.object_name.value)
        figure.subplot_imshow_color(1, 0, image,
                                    title = self.remaining_objects.value)
        
    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        
        object_name = self.object_name.value
        remaining_object_name = self.remaining_objects.value
        columns = I.get_object_measurement_columns(self.remaining_objects.value)
        columns += [(object_name, I.FF_CHILDREN_COUNT % remaining_object_name,
                     cpmeas.COLTYPE_INTEGER),
                    (remaining_object_name, I.FF_PARENT % object_name,
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
            
        setting_values = list(setting_values)
        setting_values[5] = s_lookup(setting_values[5])
        setting_values[7] = s_lookup(setting_values[7])
        return setting_values, variable_revision_number, from_matlab
    