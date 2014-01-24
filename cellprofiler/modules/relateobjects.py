'''<b>Relate Objects</b> assigns relationships; all objects (e.g. speckles) within a
parent object (e.g. nucleus) become its children.
<hr>
This module allows you to associate <i>child</i> objects with <i>parent</i> objects. 
This is useful for counting the number of children associated with each parent,
and for calculating mean measurement values for all children that are
associated with each parent.

<p>An object will be considered a child even if the edge is the only part
touching a parent object. If an child object is touching multiple parent objects,
the object will be assigned as a child of all parents that it overlaps with.

<h4>Available measurements</h4>
<b>Parent object measurements:</b>
<ul>
<li><i>Count:</i> The number of child sub-objects for each parent object.</li>
<li><i>Mean measurements:</i> The mean of the child object measurements,
calculated for each parent object.</li>
<li><i>Distances:</i> The distance of each child object to its respective parent.</li>
</ul>

<b>Child object measurements:</b>
<ul>
<li><i>Parent:</i> The label number of the parent object, as assigned by an <b>Identify</b>
module.</li>
</ul>

See also: <b>ReassignObjectNumbers</b>.
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


import sys
import numpy as np
import re
import scipy.ndimage as scind

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
from cellprofiler.settings import YES, NO
from cellprofiler.modules.identify import C_PARENT, C_CHILDREN, R_PARENT, R_CHILD
from cellprofiler.modules.identify import FF_PARENT,FF_CHILDREN_COUNT
from cellprofiler.modules.identify import \
     M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y, M_NUMBER_OBJECT_NUMBER
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.cpmath.cpmorphology import centers_of_labels
from cellprofiler.cpmath.outline import outline

D_NONE = "None"
D_CENTROID = "Centroid"
D_MINIMUM = "Minimum"
D_BOTH = "Both"

D_ALL = [D_NONE, D_CENTROID, D_MINIMUM, D_BOTH]

C_MEAN = "Mean"

FF_MEAN = '%s_%%s_%%s' % C_MEAN

'''Distance category'''
C_DISTANCE = 'Distance'

'''Centroid distance feature'''
FEAT_CENTROID = 'Centroid'

'''Minimum distance feature'''
FEAT_MINIMUM = 'Minimum'

'''Centroid distance measurement (FF_DISTANCE % parent)'''
FF_CENTROID = '%s_%s_%%s' % (C_DISTANCE, FEAT_CENTROID)

'''Minimum distance measurement (FF_MINIMUM % parent)'''
FF_MINIMUM = '%s_%s_%%s' % (C_DISTANCE, FEAT_MINIMUM)

FIXED_SETTING_COUNT = 5
VARIABLE_SETTING_COUNT = 1

class RelateObjects(cpm.CPModule):

    module_name = 'RelateObjects'
    category = "Object Processing"
    variable_revision_number = 2

    def create_settings(self):
        self.sub_object_name = cps.ObjectNameSubscriber(
            'Select the input child objects', cps.NONE,doc="""
            Child objects are defined as those objects contained within the
            parent object. For example, when relating speckles to the
            nuclei that contains them, the speckles are the children.""")

        self.parent_name = cps.ObjectNameSubscriber(
            'Select the input parent objects',
            cps.NONE,doc="""
            Parent objects are defined as those objects which encompass the 
            child object. For example, when relating speckles to the
            nuclei that contains them, the nuclei are the parents.""")

        self.find_parent_child_distances = cps.Choice(
            "Calculate child-parent distances?",
            D_ALL,doc="""
            Choose the method to calculate distances of each child to its parent.
            <ul>
            <li><i>%(D_NONE)s:</i> Do not calculate any distances.</li>
            <li><i>%(D_MINIMUM)s:</i> The distance from the 
            centroid of the child object to the closest perimeter point on
            the parent object.</li>
            <li><i>%(D_CENTROID)s:</i> The distance from the
            centroid of the child object to the centroid of the parent. </li>
            <li><i>%(D_BOTH)s:</i> Calculate both the <i>%(D_MINIMUM)s</i> and 
            <i>%(D_CENTROID)s</i> distances.</li>
            </ul>"""%globals())
        
        self.wants_step_parent_distances = cps.Binary(
            "Calculate distances to other parents?", False,doc = """
            <i>(Used only if calculating distances)</i><br>
            Select <i>%(YES)s</i> to calculate the distances of the child objects to 
            some other objects. These objects must be either parents or
            children of your parent object in order for this module to
            determine the distances. For instance, you might find "Nuclei" using
            <b>IdentifyPrimaryObjects</b>, find "Cells" using
            <b>IdentifySecondaryObjects</b> and find "Cytoplasm" using
            <b>IdentifyTertiaryObjects</b>. You can use <b>Relate</b> to relate
            speckles to cells and then measure distances to nuclei and
            cytoplasm. You could not use <b>RelateObjects</b> to relate speckles to
            cytoplasm and then measure distances to nuclei, because nuclei is
            neither a direct parent or child of cytoplasm."""%globals())
        self.step_parent_names = []

        self.add_step_parent(can_delete = False)

        self.add_step_parent_button = cps.DoSomething("","Add another parent",
                                                      self.add_step_parent)

        self.wants_per_parent_means = cps.Binary(
            'Calculate per-parent means for all child measurements?',
            False,doc="""
            Select <i>%(YES)s</i> to calculate the per-parent mean values of every upstream 
            measurement made with the children objects and stores them as a
            measurement for the parent; the nomenclature of this new measurements is 
            "Mean_&lt;child&gt;_&lt;category&gt;_&lt;feature&gt;". 
            For this reason, this module should be placed <i>after</i> all <b>Measure</b>
            modules that make measurements of the children objects."""%globals())
        
    def add_step_parent(self, can_delete = True):
        group = cps.SettingsGroup()
        group.append("step_parent_name", cps.Choice(
            "Parent name", [cps.NONE], 
            choices_fn = self.get_step_parents, doc = """
            <i>(Used only if calculating distances to another parent)</i><br>
            Choose the name of the other parent. The <b>RelateObjects</b> module will 
            measure the distance from this parent to the child objects
            in the same manner as it does to the primary parents.
            You can only choose the parents or children of
            the parent object."""))
        
        if can_delete:
            group.append("remove", cps.RemoveSettingButton(
                "", "Remove this object", self.step_parent_names, group))
        self.step_parent_names.append(group)

    def get_step_parents(self, pipeline):
        '''Return the possible step-parents associated with the parent'''
        step_parents = set()
        parent_name = self.parent_name.value
        for module in pipeline.modules():
            if module.module_num == self.module_num:
                return list(step_parents)
            #
            # Objects that are the parent of the parents
            #
            grandparents = module.get_measurements(pipeline, parent_name,
                                                   C_PARENT)
            step_parents.update(grandparents)
            #
            # Objects that are the children of the parents
            #
            siblings = module.get_measurements(pipeline, parent_name,
                                               C_CHILDREN)
            for sibling in siblings:
                match = re.match("^([^_]+)_Count", sibling)
                if match is not None:
                    sibling_name = match.groups()[0]
                    if parent_name in module.get_measurements(pipeline,
                                                              sibling_name,
                                                              C_PARENT):
                        step_parents.add(sibling_name)
        return list(step_parents)

    @property
    def has_step_parents(self):
        '''True if there are possible step-parents for the parent object'''
        return (len(self.step_parent_names) > 0 and
                len(self.step_parent_names[0].step_parent_name.choices) > 0)
    
    def settings(self):
        result = [self.sub_object_name, self.parent_name, 
                  self.find_parent_child_distances, self.wants_per_parent_means, 
                  self.wants_step_parent_distances]
        result += [group.step_parent_name for group in self.step_parent_names]
        return result

    def visible_settings(self):
        result = [self.sub_object_name, self.parent_name,
                  self.wants_per_parent_means, 
                  self.find_parent_child_distances]
        if (self.find_parent_child_distances != D_NONE and 
            self.has_step_parents):
            result += [self.wants_step_parent_distances]
            if self.wants_step_parent_distances:
                for group in self.step_parent_names:
                    result += group.visible_settings()
                result += [self.add_step_parent_button]
        return result

    def run(self, workspace):
        parents = workspace.object_set.get_objects(self.parent_name.value)
        children = workspace.object_set.get_objects(self.sub_object_name.value)
        child_count, parents_of = parents.relate_children(children)
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        if self.wants_per_parent_means.value:
            parent_indexes = np.arange(np.max(parents.segmented))+1
            for feature_name in m.get_feature_names(self.sub_object_name.value):
                if not self.should_aggregate_feature(feature_name):
                    continue
                data = m.get_current_measurement(self.sub_object_name.value,
                                                 feature_name)
                if data is not None and len(data) > 0:
                    if len(parents_of) > 0:
                        means = fix(scind.mean(data.astype(float), 
                                               parents_of, parent_indexes))
                    else:
                        means = np.zeros((0,))
                else:
                    # No child measurements - all NaN
                    means = np.ones(len(parents_of)) * np.nan
                mean_feature_name = FF_MEAN%(self.sub_object_name.value,
                                             feature_name)
                m.add_measurement(self.parent_name.value, mean_feature_name,
                                  means)
        m.add_measurement(self.sub_object_name.value,
                          FF_PARENT%(self.parent_name.value),
                          parents_of)
        m.add_measurement(self.parent_name.value,
                          FF_CHILDREN_COUNT%(self.sub_object_name.value),
                          child_count)
        good_parents = parents_of[parents_of != 0]
        image_numbers = np.ones(len(good_parents), int) * m.image_set_number
        good_children = np.argwhere(parents_of != 0).flatten() + 1
        if np.any(good_parents):
            m.add_relate_measurement(self.module_num,
                                     R_PARENT, 
                                     self.parent_name.value,
                                     self.sub_object_name.value,
                                     image_numbers,
                                     good_parents,
                                     image_numbers,
                                     good_children)
            m.add_relate_measurement(self.module_num,
                                     R_CHILD, 
                                     self.sub_object_name.value,
                                     self.parent_name.value,
                                     image_numbers,
                                     good_children,
                                     image_numbers,
                                     good_parents)
        parent_names = self.get_parent_names()
        
        for parent_name in parent_names:
            if self.find_parent_child_distances in (D_BOTH, D_CENTROID):
                self.calculate_centroid_distances(workspace, parent_name)
            if self.find_parent_child_distances in (D_BOTH, D_MINIMUM):
                self.calculate_minimum_distances(workspace, parent_name)

        if self.show_window:
            workspace.display_data.parent_labels = parents.segmented
            workspace.display_data.parent_count = parents.count
            workspace.display_data.child_labels = children.segmented
            workspace.display_data.parents_of = parents_of

    def display(self, workspace, figure):
        if not self.show_window:
            return
        from cellprofiler.gui.cpfigure_tools import renumber_labels_for_display
        figure.set_subplots((2,2))
        renumbered_parent_labels = renumber_labels_for_display(
            workspace.display_data.parent_labels)
        child_labels = workspace.display_data.child_labels
        parents_of = workspace.display_data.parents_of
        #
        # discover the mapping so that we can apply it to the children
        #
        mapping = np.arange(workspace.display_data.parent_count+1)
        mapping[workspace.display_data.parent_labels.flatten()] = \
            renumbered_parent_labels.flatten()
        parent_labeled_children = np.zeros(child_labels.shape, int)
        mask = child_labels > 0
        parent_labeled_children[mask] = \
            mapping[parents_of[child_labels[mask] - 1]]
        
        figure.subplot_imshow_labels(
            0, 0, renumbered_parent_labels,
            title = self.parent_name.value,
            renumber=False)
        figure.subplot_imshow_labels(
            1, 0, child_labels,
            title = self.sub_object_name.value,
            sharex = figure.subplot(0,0),
            sharey = figure.subplot(0,0))
        figure.subplot_imshow_labels(
            0, 1, parent_labeled_children,
            "%s labeled by %s"%
            (self.sub_object_name.value,
             self.parent_name.value),
            renumber=False,
            sharex = figure.subplot(0,0),
            sharey = figure.subplot(0,0))
    
    def get_parent_names(self):
        '''Get the names of parents to be measured for distance'''
        parent_names = [self.parent_name.value]
        if self.wants_step_parent_distances.value:
            parent_names += [group.step_parent_name.value
                             for group in self.step_parent_names]
        return parent_names
    
    def calculate_centroid_distances(self, workspace, parent_name):
        '''Calculate the centroid-centroid distance between parent & child'''
        meas = workspace.measurements
        assert isinstance(meas,cpmeas.Measurements)
        sub_object_name = self.sub_object_name.value
        parents = workspace.object_set.get_objects(parent_name)
        children = workspace.object_set.get_objects(sub_object_name)
        parents_of = self.get_parents_of(workspace, parent_name)
        pcenters = centers_of_labels(parents.segmented).transpose()
        ccenters = centers_of_labels(children.segmented).transpose()
        if pcenters.shape[0] == 0 or ccenters.shape[0] == 0:
            dist = np.array([np.NaN] * len(parents_of))
        else:
            #
            # Make indexing of parents_of be same as pcenters
            #
            parents_of = parents_of - 1
            mask = (parents_of != -1) | (parents_of > pcenters.shape[0])
            dist = np.array([np.NaN] * ccenters.shape[0])
            dist[mask] = np.sqrt(np.sum((ccenters[mask,:] - 
                                         pcenters[parents_of[mask],:])**2,1))
        meas.add_measurement(sub_object_name, FF_CENTROID % parent_name, dist)

    def calculate_minimum_distances(self, workspace, parent_name):
        '''Calculate the distance from child center to parent perimeter'''
        meas = workspace.measurements
        assert isinstance(meas,cpmeas.Measurements)
        sub_object_name = self.sub_object_name.value
        parents = workspace.object_set.get_objects(parent_name)
        children = workspace.object_set.get_objects(sub_object_name)
        parents_of = self.get_parents_of(workspace, parent_name)
        if len(parents_of) == 0:
            dist = np.zeros((0,))
        elif np.all(parents_of == 0):
            dist = np.array([np.NaN] * len(parents_of))
        else:
            mask = parents_of > 0
            ccenters = centers_of_labels(children.segmented).transpose()
            ccenters = ccenters[mask,:]
            parents_of_masked = parents_of[mask] - 1
            pperim = outline(parents.segmented)
            #
            # Get a list of all points on the perimeter
            #
            perim_loc = np.argwhere(pperim != 0)
            #
            # Get the label # for each point
            #
            perim_idx = pperim[perim_loc[:,0],perim_loc[:,1]]
            #
            # Sort the points by label #
            #
            idx = np.lexsort((perim_loc[:,1],perim_loc[:,0],perim_idx))
            perim_loc = perim_loc[idx,:]
            perim_idx = perim_idx[idx]
            #
            # Get counts and indexes to each run of perimeter points
            #
            counts = fix(scind.sum(np.ones(len(perim_idx)),perim_idx,
                                   np.arange(1,perim_idx[-1]+1))).astype(np.int32)
            indexes = np.cumsum(counts) - counts
            #
            # For the children, get the index and count of the parent
            #
            ccounts = counts[parents_of_masked]
            cindexes = indexes[parents_of_masked]
            #
            # Now make an array that has an element for each of that child's
            # perimeter points
            #
            clabel = np.zeros(np.sum(ccounts), int)
            #
            # cfirst is the eventual first index of each child in the
            # clabel array
            #
            cfirst = np.cumsum(ccounts) - ccounts
            clabel[cfirst[1:]] += 1
            clabel = np.cumsum(clabel)
            #
            # Make an index that runs from 0 to ccounts for each
            # child label.
            #
            cp_index = np.arange(len(clabel)) - cfirst[clabel]
            #
            # then add cindexes to get an index to the perimeter point
            #
            cp_index += cindexes[clabel]
            #
            # Now, calculate the distance from the centroid of each label
            # to each perimeter point in the parent.
            #
            dist = np.sqrt(np.sum((perim_loc[cp_index,:] - 
                                   ccenters[clabel,:])**2,1))
            #
            # Finally, find the minimum distance per child
            #
            min_dist = fix(scind.minimum(dist, clabel, np.arange(len(ccounts))))
            #
            # Account for unparented children
            #
            dist = np.array([np.NaN] * len(mask))
            dist[mask] = min_dist
        meas.add_measurement(sub_object_name, FF_MINIMUM % parent_name, dist)
    
    def get_parents_of(self, workspace, parent_name):
        '''Return the parents_of measurment or equivalent
        
        parent_name - name of parent objects
        
        Return a vector of parent indexes to the given parent name using
        the Parent measurement. Look for a direct parent / child link first
        and then look for relationships between self.parent_name and the
        named parent.
        '''
        meas = workspace.measurements
        assert isinstance(meas, cpmeas.Measurements)
        parent_feature = FF_PARENT%(parent_name)
        primary_parent = self.parent_name.value
        sub_object_name = self.sub_object_name.value
        primary_parent_feature = FF_PARENT%(primary_parent)
        if parent_feature in meas.get_feature_names(sub_object_name):
            parents_of = meas.get_current_measurement(sub_object_name,
                                                      parent_feature)
        elif parent_feature in meas.get_feature_names(primary_parent):
            #
            # parent_name is the grandparent of the sub-object via
            # the primary parent.
            #
            primary_parents_of = meas.get_current_measurement(
                sub_object_name, primary_parent_feature)
            grandparents_of = meas.get_current_measurement(primary_parent,
                                                           parent_feature)
            mask = primary_parents_of != 0
            parents_of = np.zeros(primary_parents_of.shape[0], 
                                  grandparents_of.dtype)
            if primary_parents_of.shape[0] > 0:
                parents_of[mask] = grandparents_of[primary_parents_of[mask]-1]
        elif primary_parent_feature in meas.get_feature_names(parent_name):
            primary_parents_of = meas.get_current_measurement(
                sub_object_name, primary_parent_feature)
            primary_parents_of_parent = meas.get_current_measurement(
                parent_name, primary_parent_feature)
            #
            # There may not be a 1-1 relationship, but we attempt to
            # construct one
            #
            reverse_lookup_len = max(np.max(primary_parents_of)+1,
                                     len(primary_parents_of_parent))
            reverse_lookup = np.zeros(reverse_lookup_len, int)
            if primary_parents_of_parent.shape[0] > 0:
                reverse_lookup[primary_parents_of_parent] =\
                              np.arange(1,len(primary_parents_of_parent)+1)
            if primary_parents_of.shape[0] > 0:
                parents_of = reverse_lookup[primary_parents_of]
        else:
            raise ValueError("Don't know how to relate %s to %s" %
                             (primary_parent, parent_name))
        return parents_of

    ignore_features = set(M_NUMBER_OBJECT_NUMBER)
    def should_aggregate_feature(self, feature_name):
        '''Return True if aggregate measurements should be made on a feature
        
        feature_name - name of a measurement, such as Location_Center_X
        '''
        if feature_name.startswith(C_MEAN):
            return False
        if feature_name.startswith(C_PARENT):
            return False
        if feature_name in self.ignore_features:
            return False
        return True
        
    def validate_module(self, pipeline):
        '''Validate the module's settings
        
        Relate will complain if the children and parents are related
        by a prior module or if a step-parent is named twice'''
        for module in pipeline.modules():
            if module == self:
                break
            parent_features = module.get_measurements(
                pipeline, self.sub_object_name.value, "Parent")
            if self.parent_name.value in (parent_features):
                raise cps.ValidationError(
                    "%s and %s were related by the %s module"%
                    (self.sub_object_name.value, self.parent_name.value,
                     module.module_name),self.parent_name)
        if self.has_step_parents and self.wants_step_parent_distances:
            step_parents = set()
            for group in self.step_parent_names:
                if group.step_parent_name.value in step_parents:
                    raise cps.ValidationError(
                        "%s has already been chosen" %
                        group.step_parent_name.value,
                        group.step_parent_name)
                step_parents.add(group.step_parent_name.value)
    
    def get_child_columns(self, pipeline):
        child_columns = pipeline.get_measurement_columns(self)
        child_columns = [column
                         for column in child_columns
                         if column[0] == self.sub_object_name.value and
                         self.should_aggregate_feature(column[1])]
        return child_columns
        
    def get_measurement_columns(self, pipeline):
        '''Return the column definitions for this module's measurements'''
        columns = [(self.sub_object_name.value,
                    FF_PARENT%(self.parent_name.value),
                    cpmeas.COLTYPE_INTEGER),
                   (self.parent_name.value,
                    FF_CHILDREN_COUNT%self.sub_object_name.value,
                    cpmeas.COLTYPE_INTEGER)]
        if self.wants_per_parent_means.value:
            child_columns = self.get_child_columns(pipeline)
            columns += [(self.parent_name.value,
                         FF_MEAN%(self.sub_object_name.value, column[1]),
                         cpmeas.COLTYPE_FLOAT)
                        for column in child_columns]
        if self.find_parent_child_distances in (D_BOTH, D_CENTROID):
            for parent_name in self.get_parent_names():
                columns += [(self.sub_object_name.value,
                             FF_CENTROID % parent_name,
                             cpmeas.COLTYPE_INTEGER)]
        if self.find_parent_child_distances in (D_BOTH, D_MINIMUM):
            for parent_name in self.get_parent_names():
                columns += [(self.sub_object_name.value,
                             FF_MINIMUM % parent_name,
                             cpmeas.COLTYPE_INTEGER)]
        return columns
    
    def get_object_relationships(self, pipeline):
        '''Return the object relationships produced by this module'''
        parent_name = self.parent_name.value
        sub_object_name = self.sub_object_name.value
        return [(R_PARENT, parent_name, sub_object_name, 
                 cpmeas.MCA_AVAILABLE_EACH_CYCLE),
                (R_CHILD, sub_object_name, parent_name,
                 cpmeas.MCA_AVAILABLE_EACH_CYCLE)]

    def get_categories(self, pipeline, object_name):
        if object_name == self.parent_name.value:
            if self.wants_per_parent_means:
                return ["Mean_%s" % self.sub_object_name, "Children"]
            else:
                return ["Children"]
        elif object_name == self.sub_object_name.value:
            result = ["Parent"]
            if self.find_parent_child_distances != D_NONE:
                result += [C_DISTANCE]
            return result
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.parent_name.value:
            if category == "Mean_%s"%self.sub_object_name.value:
                measurements = []
                child_columns = self.get_child_columns(pipeline)
                measurements += [column[1] for column in child_columns]
                return measurements
            elif category == "Children":
                return ["%s_Count"%self.sub_object_name.value]
        elif object_name == self.sub_object_name.value and category == "Parent":
            return [ self.parent_name.value ]
        elif (object_name == self.sub_object_name.value and 
              category == C_DISTANCE):
            result = []
            if self.find_parent_child_distances in (D_BOTH, D_CENTROID):
                result += ['%s_%s' % (FEAT_CENTROID, parent_name)
                           for parent_name in self.get_parent_names()]
            if self.find_parent_child_distances in (D_BOTH, D_MINIMUM):
                result += ['%s_%s' % (FEAT_MINIMUM, parent_name)
                           for parent_name in self.get_parent_names()]
            return result
        return []

    def prepare_settings(self, setting_values):
        """Do any sort of adjustment to the settings required for the given values
        
        setting_values - the values for the settings just prior to mapping
                         as done by set_settings_from_values
        This method allows a module to specialize itself according to
        the number of settings and their value. For instance, a module that
        takes a variable number of images or objects can increase or decrease
        the number of relevant settings so they map correctly to the values.
        """
        setting_count = len(setting_values)
        step_parent_count = ((setting_count - FIXED_SETTING_COUNT) / 
                             VARIABLE_SETTING_COUNT)
        assert len(self.step_parent_names) > 0
        self.step_parent_names = self.step_parent_names[:1]
        for i in range(1,step_parent_count):
            self.add_step_parent()
            
    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        if from_matlab and variable_revision_number == 2:
            setting_values = [setting_values[0],
                              setting_values[1],
                              setting_values[2],
                              cps.DO_NOT_USE,
                              cps.YES]
            variable_revision_number = 3
            
        if from_matlab and variable_revision_number == 3:
            setting_values = list(setting_values)
            setting_values[2] = (D_MINIMUM if setting_values[2] == cps.YES 
                                 else D_NONE)
            variable_revision_number = 4
                
        if from_matlab and variable_revision_number == 4:
            if setting_values[2] == cps.DO_NOT_USE:
                setting_values = (setting_values[:2] + [D_NONE] +
                                  setting_values[3:])
            from_matlab = False
            variable_revision_number = 1
        if (not from_matlab) and variable_revision_number == 1:
            #
            # Added other distance parents
            #
            if setting_values[2] == cps.DO_NOT_USE:
                find_parent_distances = D_NONE
            else:
                find_parent_distances = setting_values[2]
            wants_step_parent_distances = (
                cps.NO if setting_values[3].upper() == cps.DO_NOT_USE.upper()
                else cps.YES)
            setting_values = (setting_values[:2] +
                              [find_parent_distances,
                               setting_values[4],
                               wants_step_parent_distances,
                               setting_values[3]])
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

Relate = RelateObjects
