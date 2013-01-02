'''<b>Identify Tertiary Objects</b> identifies tertiary objects (e.g., cytoplasm) by removing smaller primary
objects (e.g. nuclei) from larger secondary objects (e.g., cells), leaving a ring shape
<hr>

This module will take the smaller identified objects and remove them from
the larger identified objects. For example, "subtracting" the nuclei from
the cells will leave just the cytoplasm, the properties of which can then
be measured by <b>Measure</b> modules. The larger objects should therefore be
equal in size or larger than the smaller objects and must completely
contain the smaller objects.  Both inputs should be objects produced by
<b>Identify</b> modules, not grayscale images.

<p><i>Note:</i> Creating subregions using this module can result in objects that
are not contiguous, which does not cause problems when running the
<b>MeasureImageIntensity</b> and <b>MeasureTexture</b> modules, but does cause 
problems when running the <b>MeasureObjectSizeShape</b> module because calculations 
of the perimeter, aspect ratio, solidity, etc. cannot be made for noncontiguous
objects.

<i>Special note on saving images:</i> You can use the settings in this module to pass object outlines along object
outlines can be passed along to the module <b>OverlayOutlines</b> and then
save them with the <b>SaveImages</b> module. You can also pass the identified objects themselves along to the object
processing module <b>ConvertToImage</b> and then save them with the <b>SaveImages</b> module.

<h4>Available measurements</h4>
<ul>
<li><i>Image features:</i>
<ul>
<li><i>Count:</i> The number of tertiary objects identified.</li>
</ul>
</li>         
<li><i>Object features:</i>
<ul>
<li><i>Parent:</i> The identity of the primary object and secondary object associated 
with each tertiary object.</li>
<li><i>Location_X, Location_Y:</i> The pixel (X,Y) coordinates of the center of mass of the 
identified tertiary objects.</li>
</ul>
</li>
</ul>

See also <b>IdentifyPrimaryObject</b> and <b>IdentifySecondaryObject</b> modules.
'''

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org


import numpy as np
import matplotlib
import matplotlib.cm

import identify as cpmi
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs
from cellprofiler.cpmath.outline import outline

class IdentifyTertiaryObjects(cpm.CPModule):

    module_name = "IdentifyTertiaryObjects"
    variable_revision_number = 1
    category = "Object Processing"
    
    def create_settings(self):
        """Create the settings for the module
        
        Create the settings for the module during initialization.
        """
        self.secondary_objects_name = cps.ObjectNameSubscriber("Select the larger identified objects","None",doc="""
            What did you call the larger identified objects?""")
        
        self.primary_objects_name = cps.ObjectNameSubscriber("Select the smaller identified objects","None",doc="""
            What did you call the smaller identified objects?""")
        
        self.subregion_objects_name = cps.ObjectNameProvider("Name the tertiary objects to be identified","Cytoplasm",doc="""
            What do you want to call the new subregions? The new tertiary subregion 
            will consist of the smaller object subtracted from the larger object.""")
        
        self.use_outlines = cps.Binary("Retain outlines of the tertiary objects?",False)
        
        self.outlines_name = cps.OutlineNameProvider("Name the outline image","CytoplasmOutlines", doc="""\
            <i>(Used only if outlines are to be retained for later use in the pipeline)</i><br>
            <p> Enter a name that will allow the outlines to be selected later in the pipeline.""") 

    def settings(self):
        """All of the settings to be loaded and saved in the pipeline file
        
        Returns a list of the settings in the order that they should be
        saved or loaded in the pipeline file.
        """
        return [self.secondary_objects_name, self.primary_objects_name,
                self.subregion_objects_name, self.outlines_name,
                self.use_outlines]
    
    def visible_settings(self):
        """The settings that should be visible on-screen
        
        Returns the list of settings in the order that they should be
        displayed in the module setting editor. These can be tailored
        to only display relevant settings (see use_outlines/outlines_name
        below)
        """
        result = [self.secondary_objects_name, self.primary_objects_name,
                  self.subregion_objects_name, self.use_outlines]
        #
        # display the name of the outlines image only if the user
        # has asked to use outlines
        #
        if self.use_outlines.value:
            result.append(self.outlines_name)
        return result
    
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
                    for p_size,s_size
                    in zip(primary_labels.shape, secondary_labels.shape)]):
                #
                # Look for a cropping mask associated with the primary_labels
                # and apply that mask to resize the secondary labels
                #
                secondary_labels = primary_objects.crop_image_similarly(secondary_labels)
                tertiary_image = primary_objects.parent_image
            elif any([p_size > s_size 
                    for p_size,s_size
                    in zip(primary_labels.shape, secondary_labels.shape)]):
                primary_labels = secondary_objects.crop_image_similarly(primary_labels)
                tertiary_image = secondary_objects.parent_image
            elif secondary_objects.parent_image != None:
                tertiary_image = secondary_objects.parent_image
            else:
                tertiary_image = primary_objects.parent_image
        except ValueError:
            # No suitable cropping - resize all to fit the secondary
            # labels which are the most critical.
            #
            primary_labels, _ = cpo.size_similarly(secondary_labels, primary_labels)
            if secondary_objects.parent_image != None:
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
        primary_mask = np.logical_or(primary_labels == 0,
                                     primary_outline)
        tertiary_labels[primary_mask == False] = 0
        #
        # Get the outlines of the tertiary image
        #
        tertiary_outlines = outline(tertiary_labels)!=0
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
        child_count_of_primary, primary_parents = \
            primary_objects.relate_children(tertiary_objects)
        
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
        for parent_objects_name, parents_of, child_count\
         in ((self.primary_objects_name, primary_parents,child_count_of_primary),
             (self.secondary_objects_name, secondary_parents, child_count_of_secondary)):
            m.add_measurement(self.subregion_objects_name.value,
                              cpmi.FF_PARENT%(parent_objects_name.value),
                              parents_of)
            m.add_measurement(parent_objects_name.value,
                              cpmi.FF_CHILDREN_COUNT%(self.subregion_objects_name.value),
                              child_count)
        object_count = np.max(tertiary_labels)
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
        #
        # The outlines
        #
        if self.use_outlines.value:
            out_img = cpi.Image(tertiary_outlines.astype(bool),
                                parent_image = tertiary_image)
            workspace.image_set.add(self.outlines_name.value, out_img)

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
                                       sharexy = figure.subplot(0, 0))
        figure.subplot_imshow_labels(0, 1, tertiary_labels,
                                       self.subregion_objects_name.value,
                                       sharexy = figure.subplot(0, 0))
        figure.subplot_imshow_bw(1, 1, tertiary_outlines,
                                   "Outlines",
                                   sharexy = figure.subplot(0, 0))

    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        subregion_name = self.subregion_objects_name.value
        columns = cpmi.get_object_measurement_columns(subregion_name)
        for parent in (self.primary_objects_name.value, 
                       self.secondary_objects_name.value):
            columns += [(parent,
                         cpmi.FF_CHILDREN_COUNT%subregion_name,
                         cpmeas.COLTYPE_INTEGER),
                        (subregion_name,
                         cpmi.FF_PARENT%parent,
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
        
        return setting_values,variable_revision_number,from_matlab
    
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
        if (object_name == self.subregion_objects_name):
            categories += ("Parent", "Location","Number")
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
                result += [ "Center_X","Center_Y"]
            elif category == "Parent":
                result += [ self.primary_objects_name.value,
                            self.secondary_objects_name.value]
            elif category == "Number":
                result += ["Object_Number"]
        return result
    
IdentifyTertiarySubregion = IdentifyTertiaryObjects
