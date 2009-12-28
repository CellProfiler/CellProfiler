'''<b>Relate</b> assigns relationships; all objects (e.g. speckles) within a
parent object (e.g. nucleus) become its children
<hr>
Allows associating "children" objects with "parent" objects. This is
useful for counting the number of children associated with each parent,
and for calculating mean measurement values for all children that are
associated with each parent.

<p>An object will be considered a child even if the edge is the only part
touching a parent object. If an object is touching two parent objects,
the object will be assigned to the parent that shares the largest
number of pixels with the child.
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

__version__ = "$Revision$"

import sys
import numpy as np
import scipy.ndimage as scind

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.cpmath.cpmorphology import centers_of_labels
from cellprofiler.cpmath.outline import outline

D_NONE = cps.DO_NOT_USE
D_CENTROID = "Centroid"
D_MINIMUM = "Minimum"
D_BOTH = "Both"

D_ALL = [D_NONE, D_CENTROID, D_MINIMUM, D_BOTH]

FF_PARENT = "Parent_%s"

FF_CHILDREN_COUNT = "Children_%s_Count"

FF_MEAN = 'Mean_%s_%s'

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

class Relate(cpm.CPModule):

    module_name = 'Relate'
    category = "Object Processing"
    variable_revision_number = 1

    def create_settings(self):
        self.sub_object_name = cps.ObjectNameSubscriber('Select the input child objects',
                                                        'None',doc="""
            The child objects are defined as those objects contained within the
            parent object.For example, to <b>Relate</b> a speckle to a containing
            nucleus, the child is the speckle object(s).""")
        self.parent_name = cps.ObjectNameSubscriber('Select the input parent objects',
                                                    'None',doc="""
            The parent objects are defined as those objects which encompass the 
            child object. For example, to <b>Relate</b> a speckle to a 
            containing nucleus, the parent is the nucleus object.""")
        self.find_parent_child_distances = cps.Choice(
            "Find distances?",
            D_ALL,doc="""
            Do you want to find the minimum distances of each child to its 
            parent?
            <br>
            <ul><li>The <i>minimum distance</i> is the distance from the 
            centroid of the child object to the closest perimeter point on
            the parent object.</li>
            <li>The <i>centroid distance</i> is the distance from the
            centroid of the child object to the centroid of the parent.
            </li></ul>""")
        self.step_parent_name = cps.ObjectNameSubscriber("Select additional objects to find distances to:", None,doc = """
                                                         You can find distances to additional objects,
                                                         or "step-parents""")
        self.wants_per_parent_means = cps.Binary('Calculate per-parent means for all child measurements?',
                                                 False,doc="""
            For every measurement that has been made of
            the children objects upstream in the pipeline, this module calculates the
            mean value of that measurement over all children and stores it as a
            measurement for the parent, as "Mean_&lt;child&gt;_&lt;category&gt;_&lt;feature&gt;". 
            For this reason, this module should be placed <i>after</i> all <b>Measure</b>
            modules that make measurements of the children objects.""")

    def settings(self):
        return [self.sub_object_name, self.parent_name, 
                self.find_parent_child_distances, self.step_parent_name,
                self.wants_per_parent_means]


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
                if data is not None:
                    if len(parents_of) > 0:
                        means = fix(scind.mean(data.astype(float), 
                                               parents_of, parent_indexes))
                    else:
                        means = np.zeros((0,))
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
        if self.find_parent_child_distances in (D_BOTH, D_CENTROID):
            self.calculate_centroid_distances(workspace)
        if self.find_parent_child_distances in (D_BOTH, D_MINIMUM):
            self.calculate_minimum_distances(workspace)
            
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
    
    def calculate_centroid_distances(self, workspace):
        '''Calculate the centroid-centroid distance between parent & child'''
        meas = workspace.measurements
        assert isinstance(meas,cpmeas.Measurements)
        parent_name = self.parent_name.value
        sub_object_name = self.sub_object_name.value
        parents = workspace.object_set.get_objects(parent_name)
        children = workspace.object_set.get_objects(sub_object_name)
        parents_of = meas.get_current_measurement(sub_object_name,
                                                  FF_PARENT%(parent_name))
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

    def calculate_minimum_distances(self, workspace):
        '''Calculate the distance from child center to parent perimeter'''
        meas = workspace.measurements
        assert isinstance(meas,cpmeas.Measurements)
        parent_name = self.parent_name.value
        sub_object_name = self.sub_object_name.value
        parents = workspace.object_set.get_objects(parent_name)
        children = workspace.object_set.get_objects(sub_object_name)
        parents_of = meas.get_current_measurement(sub_object_name,
                                                  FF_PARENT%(parent_name))
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
                                   np.arange(1,perim_idx[-1]+1))).astype(int)
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
            
    def validate_module(self, pipeline):
        '''Validate the module's settings
        
        Relate will complain if the children and parents are related
        by a prior module'''
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
        
    def get_measurement_columns(self, pipeline):
        '''Return the column definitions for this module's measurements'''
        columns = [(self.sub_object_name.value,
                    FF_PARENT%(self.parent_name.value),
                    cpmeas.COLTYPE_INTEGER),
                   (self.parent_name.value,
                    FF_CHILDREN_COUNT%self.sub_object_name.value,
                    cpmeas.COLTYPE_INTEGER)]
        if self.wants_per_parent_means.value:
            child_columns = pipeline.get_measurement_columns(self)
            columns += [(self.parent_name.value,
                         FF_MEAN%(self.sub_object_name.value, column[1]),
                         cpmeas.COLTYPE_FLOAT)
                        for column in child_columns
                        if column[0] == self.sub_object_name.value]
        if self.find_parent_child_distances in (D_BOTH, D_CENTROID):
            columns += [(self.sub_object_name.value,
                         FF_CENTROID % self.parent_name.value,
                         cpmeas.COLTYPE_INTEGER)]
        if self.find_parent_child_distances in (D_BOTH, D_MINIMUM):
            columns += [(self.sub_object_name.value,
                         FF_MINIMUM % self.parent_name.value,
                         cpmeas.COLTYPE_INTEGER)]
        return columns

    def get_categories(self, pipeline, object_name):
        if object_name == self.parent_name.value:
            return ["Mean_%s"%self.sub_object_name.value,
                    "Children"]
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
        elif (object_name == self.sub_object_name.value and 
              category == C_DISTANCE):
            result = []
            if self.find_parent_child_distances in (D_BOTH, D_CENTROID):
                result += [FF_CENTROID % self.parent_name.value]
            if self.find_parent_child_distances in (D_BOTH, D_MINIMUM):
                result += [FF_MINIMUM % self.parent_name.value]
            return result
        return []
    
    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
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

