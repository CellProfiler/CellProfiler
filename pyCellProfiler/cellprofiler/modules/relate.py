'''relate.py - Relate child objects to parents

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import sys
import numpy as np
import scipy.ndimage as scind

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix

D_NONE = cps.DO_NOT_USE
D_CENTROID = "Centroid"
D_MINIMUM = "Minimum"
D_BOTH = "Both"

D_ALL = [D_NONE, D_CENTROID, D_MINIMUM, D_BOTH]

FF_PARENT = "Parent_%s"

FF_CHILDREN_COUNT = "Children_%s_Count"

FF_MEAN = 'Mean_%s_%s'

FF_SUB_OBJECT_FLAG = 'SubObjectFlag'

class Relate(cpm.CPModule):
    ''' SHORT DESCRIPTION:
    Assigns relationships: All objects (e.g. speckles) within a parent object
    (e.g. nucleus) become its children.
    *************************************************************************
    
    Allows associating "children" objects with "parent" objects. This is
    useful for counting the number of children associated with each parent,
    and for calculating mean measurement values for all children that are
    associated with each parent. For every measurement that has been made of
    the children objects upstream in the pipeline, this module calculates the
    mean value of that measurement over all children and stores it as a
    measurement for the parent, as "Mean_<child>_<category>_<feature>". 
    For this reason, this module should be placed *after* all Measure modules
    that make measurements of the children objects.

    An object will be considered a child even if the edge is the only part
    touching a parent object. If an object is touching two parent objects,
    the object will be assigned to the parent that shares the largest
    number of pixels with the child.
    '''

    category = "Object Processing"
    variable_revision_number = 1
    
    def create_settings(self):
        self.module_name = 'Relate'
        self.sub_object_name = cps.ObjectNameSubscriber('What objects do you want as the children (i.e. sub-objects)?',
                                                        'None')
        self.parent_name = cps.ObjectNameSubscriber('What objects do you want as the parents?',
                                                    'None')
        self.find_parent_child_distances = cps.Choice("Do you want to find minimum distances of each child to its parent?",
                                                      [D_NONE, D_MINIMUM])
        self.step_parent_name = cps.ObjectNameSubscriber("What other object do you want to find distances to?", None)
        self.wants_per_parent_means = cps.Binary('Do you want to generate per-parent means for all child measurements?',
                                                 False)

    def settings(self):
        return [self.sub_object_name, self.parent_name, 
                self.find_parent_child_distances, self.step_parent_name,
                self.wants_per_parent_means]


    def backwards_compatibilize(self, setting_values, variable_revision_number, module_name, from_matlab):
        if from_matlab and variable_revision_number == 2:
            setting_values = [setting_values[0],
                              setting_values[1],
                              setting_values[2],
                              cps.YES,
                              cps.YES]
            variable_revision_number = 3
            
        if from_matlab and variable_revision_number == 3:
            setting_values = list(setting_values)
            setting_values[2] = (D_MINIMUM if setting_values[2] == cps.YES 
                                 else D_NONE)
            variable_revision_number = 4
                
        if from_matlab and variable_revision_number == 4:
            if setting_values[2] in (D_CENTROID, D_BOTH):
                sys.stderr.write("Warning: the Relate module doesn't currently support the centroid distance measurement\n")
            from_matlab = False
            variable_revision_number = 1
        return setting_values, variable_revision_number, from_matlab

    def visible_settings(self):
        # Currently, we don't support measuring distances, so those questions
        # are not shown.
        return [self.sub_object_name, self.parent_name,
                self.wants_per_parent_means]

    def run(self, workspace):
        parents = workspace.object_set.get_objects(self.parent_name.value)
        children = workspace.object_set.get_objects(self.sub_object_name.value)
        child_count, parents_of = parents.relate_children(children)
        m = workspace.measurements
        if self.wants_per_parent_means.value:
            parent_indexes = np.arange(np.max(parents.segmented))+1
            for feature_name in m.get_feature_names(self.sub_object_name.value):
                data = m.get_current_measurement(self.sub_object_name.value,
                                                 feature_name)
                means = fix(scind.mean(data, parents_of, parent_indexes))
                mean_feature_name = FF_MEAN%(self.sub_object_name.value,
                                             feature_name)
                m.add_measurement(self.parent_name.value, mean_feature_name,
                                  means)
        m.add_measurement(self.sub_object_name.value,
                          FF_PARENT%(self.parent_name.value),
                          parents_of)
        m.add_measurement(self.sub_object_name.value,
                          FF_SUB_OBJECT_FLAG,
                          np.ones(len(parents_of), int))
        m.add_measurement(self.parent_name.value,
                          FF_CHILDREN_COUNT%(self.sub_object_name.value),
                          child_count)
        if workspace.frame is not None:
            figure = workspace.create_or_find_figure(subplots=(2,2))
            figure.subplot_imshow_labels(0,0,parents.segmented,
                                         title = self.parent_name.value)
            figure.subplot_imshow_labels(1,0,children.segmented,
                                         title = self.sub_object_name.value)
            parent_labeled_children = np.zeros(children.segmented.shape, int)
            parent_labeled_children[children.segmented > 0] = \
                parents_of[children.segmented[children.segmented > 0]-1]
            figure.subplot_imshow_labels(0,1,parent_labeled_children,
                                         "%s labeled by %s"%
                                         (self.sub_object_name.value,
                                          self.parent_name.value))
    
    def get_measurement_columns(self, pipeline):
        '''Return the column definitions for this module's measurements'''
        columns = [(self.sub_object_name.value,
                    FF_PARENT%(self.parent_name.value),
                    cpmeas.COLTYPE_INTEGER),
                   (self.sub_object_name.value,
                    FF_SUB_OBJECT_FLAG,
                    cpmeas.COLTYPE_INTEGER),
                   (self.parent_name.value,
                    FF_CHILDREN_COUNT%self.sub_object_name.value,
                    cpmeas.COLTYPE_INTEGER)]
        if self.wants_per_parent_means.value:
            child_columns = pipeline.get_measurement_columns(self)
            columns += [(self.parent_name.value,
                         FF_MEAN%(self.sub_object_name.value, column[1]),
                         cpmeas.COLTYPE_FLOAT)
                        for column in child_columns]
        return columns

    def get_categories(self, pipeline, object_name):
        if object_name == self.parent_name.value:
            return ["Mean_%s"%self.sub_object_name.value,
                    "Children"]
        elif object_name == self.sub_object_name.value:
            return ["Parent"]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.parent_name.value:
            if category == "Mean_%s"%self.sub_object_name.value:
                measurements = []
                for module in pipeline.modules():
                    c = module.get_categories(self.sub_object_name.value)
                    for category in c:
                        m = module.get_measurements(self.sub_object_name.value,
                                                    category)
                        measurements += ["%s_%s"%(c,x) for x in m]
                return measurements
            elif category == "Children":
                return ["%s_Count"%self.sub_object_name.value]
        elif object_name == self.sub_object_name.value and category == "Parent":
            return [ self.parent_name.value ]
        return []
