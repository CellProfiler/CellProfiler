'''<b>Identify Tertiary Subregion</b> identifies (e.g. cytoplasm) by removing the primary
objects (e.g. nuclei) from secondary objects (e.g. cells) leaving a ring shape.
<hr>
This module will take the smaller identified objects and remove from them
the larger identified objects. For example, "subtracting" the nuclei from
the cells will leave just the cytoplasm, the properties of which can then
be measured by Measure modules. The larger objects should therefore be
equal in size or larger than the smaller objects and must completely
contain the smaller objects.  Both inputs should be objects produced by
identify modules, not images.

<p>Note: Creating subregions using this module can result in objects that
are not contiguous, which does not cause problems when running the
<b>Measure Intensity</b> and <b>Measure Texture</b> modules, but does cause 
problems when running the <b>Measure AreaShape</b> module because calculations 
of the perimeter, aspect ratio, solidity, etc. cannot be made for noncontiguous
objects.

See also <b>Identify Primary</b> and <b>Identify Secondary</b> modules.
'''

#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Developed by the Broad Institute
#Copyright 2003-2009
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org

__version__="$Revision$"

import numpy as np
import matplotlib
import matplotlib.cm

import identify as cpmi
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.gui.cpfigure as cpf
import cellprofiler.preferences as cpprefs
from cellprofiler.cpmath.outline import outline

class IdentifyTertiarySubregion(cpm.CPModule):

    module_name = "IdentifyTertiarySubregion"
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
        
        self.subregion_objects_name = cps.ObjectNameProvider("Name the identified subregion objects","Cytoplasm",doc="""
            What do you want to call the new subregions? The new tertiary subregion 
            will consist of the primary object subtracted from the secondary object.""")
        
        self.use_outlines = cps.Binary("Save outlines of the identified objects?",False)
        
        self.outlines_name = cps.OutlineNameProvider("Name the outline image","CytoplasmOutlines", doc="""\
            <i>(Only used if outlines are to be saved)</i>
            <p>The outlines of the identified objects may be used by modules downstream,
            by selecting them from any drop-down image list.""") 

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
        
        if workspace.frame != None:
            #
            # Draw the primary, secondary and tertiary labels
            # and the outlines
            #
            window_name = "CellProfiler(%s:%d)"%(self.module_name,self.module_num)
            my_frame=cpf.create_or_find(workspace.frame, 
                                        title="Identify tertiary subregion", 
                                        name=window_name, subplots=(2,2))
            
            title = "%s, cycle # %d"%(self.primary_objects_name.value,
                                      workspace.image_set.number+1)
            my_frame.subplot_imshow_labels(0,0,primary_labels,title)
            my_frame.subplot_imshow_labels(1,0,secondary_labels, 
                                           self.secondary_objects_name.value)
            my_frame.subplot_imshow_labels(0, 1,tertiary_labels, 
                                           self.subregion_objects_name.value)
            my_frame.subplot_imshow_bw(1,1,tertiary_outlines, 
                                       "Outlines")
            my_frame.Refresh()
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
                                tertiary_image)
            workspace.image_set.add(self.outlines_name.value, out_img)
            
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
    
