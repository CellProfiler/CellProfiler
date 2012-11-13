'''<b>Reassign Object Numbers</b> renumbers objects
<hr>

Objects in CellProfiler are tracked, and measurements of objects are associated 
with each other, based on object numbers (also known as object labels). Typically,
each object is assigned a single unique number; exported measurements are ordered
by this numbering.  This module
allows the reassignment of object numbers, which may be useful in certain cases. 
The <i>Unify</i> option assigns adjacent or nearby
objects the same number based on certain criteria. It can be useful, for example, 
to merge together touching objects that were incorrectly split into two pieces 
by an <b>Identify</b> module.
The <i>Split</i> option assigns unique numbers to 
portions of separate objects that previously had been using the same number, which 
might occur if you applied certain operations in the <b>Morph</b> module to objects.
<p>
Technically, reassignment means that the numerical value of every pixel consisting 
of an object (in the label matrix version of the image) is changed, according to
the module's settings. In order to ensure that objects are numbered consecutively 
without gaps in the numbering (which other modules may depend on), 
<b>ReassignObjectNumbers</b> will typically result in most of the objects having 
their numbers reassigned. This reassignment information is stored and can be exported 
from CellProfiler like any other measurement: each original input object will have
its reassigned object number stored as a feature in case you need to track the 
reassignment.

See also <b>RelateObjects</b>.
'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2012 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org


__version__="$Revision$"

import numpy as np
import scipy.ndimage as scind
from scipy.sparse import coo_matrix

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi
import cellprofiler.preferences as cpprefs
from cellprofiler.modules.identify import get_object_measurement_columns
from cellprofiler.modules.identify import add_object_count_measurements
from cellprofiler.modules.identify import add_object_location_measurements
from cellprofiler.modules.identify import C_PARENT, C_CHILDREN
from cellprofiler.modules.identify import FF_CHILDREN_COUNT, FF_PARENT
import cellprofiler.cpmath.cpmorphology as morph
from cellprofiler.cpmath.filter import stretch
import cellprofiler.cpmath.outline

OPTION_UNIFY = "Unify"
OPTION_SPLIT = "Split"

UNIFY_DISTANCE = "Distance"
UNIFY_PARENT = "Per-parent"
UNIFY_SHAPE = "Shape"

CA_CENTROIDS = "Centroids"
CA_CLOSEST_POINT = "Closest point"

class ReassignObjectNumbers(cpm.CPModule):
    module_name = "ReassignObjectNumbers"
    category = "Object Processing"
    variable_revision_number = 4
    
    def create_settings(self):
        self.objects_name = cps.ObjectNameSubscriber(
            "Select the input objects",
            cps.NONE,
            doc="""Select the objects whose object numbers you want to reassign.
            You can use any objects that were created in previous modules, such as 
            <b>IdentifyPrimaryObjects</b> or <b>IdentifySecondaryObjects</b>.""")
        
        self.output_objects_name = cps.ObjectNameProvider(
            "Name the new objects","RelabeledNuclei",
            doc="""What do you want to call the objects whose numbers have been reassigned?
            You can use this name in subsequent
            modules that take objects as inputs.""")
        
        self.relabel_option = cps.Choice(
            "Operation to perform",[OPTION_UNIFY, OPTION_SPLIT],
            doc="""Choose <i>Unify</i> to assign adjacent or nearby objects the same
            object number. Choose <i>Split</i> to give a unique number to non-adjacent objects
            that currently share the same object number.""")
        
        self.unify_option = cps.Choice(
            "Unification to perform",[UNIFY_DISTANCE, UNIFY_PARENT, UNIFY_SHAPE],
            doc="""
            <i>(Used only with the Unify option)</i><br>
            You can unify objects in one of two ways:
            <ul>
            <li><i>%(UNIFY_DISTANCE)s: </i> All objects within a certain pixel radius 
            from each other will be unified</li>
            <li><i>%(UNIFY_PARENT)s: </i>All objects which share the same parent 
            relationship to another object will be unified. This is not be confused
            with using the <b>RelateObjects</b> module, in which the related objects
            remain as individual objects. See <b>RelateObjects</b> for more details.</li>
            <li><i>%(UNIFY_SHAPE)s:</i> This method finds the vertices of a pair of objects
            (the location where the pair of objects and the background meet) and
            calculates the fraction of contiguous non-object-pair-labeled pixels within a window around
            the vertex. If at least one of the vertices is above the threshold, the objects are joined.
            <ul>
            <li>&lt;0.5: The objects curve away from the background (convex).</li>
            <li>0.5: The object surface is flat with respect to the background.</li>
            <li>&gt;0.5: The objects curve into from the background (concave).</li>
            </ul>
            This method works well for round cell objects which do not satisfy the conditions for the Centroids
            method.</li>
            </ul>
            """%globals())
        
        self.parent_object = cps.Choice(
            "Select the parent object", [cps.NONE], choices_fn = self.get_parent_choices, doc = """
            Select the parent object that will be used to
            unify the child objects. Please note the following:
            <ul>
            <li>You must have established a parent-child relationship
            between the objects using a prior <b>RelateObjects</b> module.</li>
            <li>Primary objects and their associated secondary objects are
            already in a one-to-one parent-child relationship, so it makes no
            sense to unify them here.</li>
            </ul>""")
        
        self.distance_threshold = cps.Integer(
            "Maximum distance within which to unify objects",
            0,minval=0, doc="""
            <i>(Used only when Unifying by distance)</i><br>
            Objects that are less than or equal to the distance
            you enter here, in pixels, will be unified. If you choose zero 
            (the default), only objects that are touching will be unified. 
            Note that <i>Unify</i> will not actually connect or bridge
            the two objects by adding any new pixels; it simply assigns the same object number
            to the portions of the object. The new, unified object
            may therefore consist of two or more unconnected components.""")
        
        self.wants_image = cps.Binary(
            "Unify using a grayscale image?", False,
            doc="""
            <i>(Used only with the unify option)</i><br>
            <i>Unify</i> can use the objects' intensity features to determine whether two
            objects should be unified. If you choose to use a grayscale image,
            <i>Unify</i> will unify two objects only if they
            are within the distance you have specified <i>and</i> certain criteria about the objects
            within the grayscale image are met.""")
        
        self.image_name = cps.ImageNameSubscriber(
            "Select the grayscale image to guide unification", cps.NONE,
            doc="""
            <i>(Used only if a grayscale image is to be used as a guide for unification)</i><br>
            Select the name of an image loaded or created by a previous module.""")
        
        self.minimum_intensity_fraction = cps.Float(
            "Minimum intensity fraction", .9, minval=0, maxval=1,
            doc="""
            <i>(Used only if a grayscale image is to be used as a guide for unification)</i><br>
            Select the minimum acceptable intensity fraction. This will be used 
            as described for the method you choose in the next setting.""")
        
        self.where_algorithm = cps.Choice(
            "Method to find object intensity",
            [CA_CLOSEST_POINT, CA_CENTROIDS],
            doc = """
            <i>(Used only if a grayscale image is to be used as a guide for unification)</i><br>
            You can use one of two methods to determine whether two
            objects should unified, assuming they meet the distance criteria (as specified above):
            <ul>
            <li><i>Centroids:</i> When the module considers merging two objects, 
            this method identifies the centroid of each object, 
            records the intensity value of the dimmer of the two centroids, 
            multiplies this value by the <i>minimum intensity fraction</i> to generate a threshold,
            and draws a line between the centroids. The method will unify the 
            two objects only if the intensity of every point along the line is above 
            the threshold. For instance, if the intensity
            of one centroid is 0.75 and the other is 0.50 and the <i>minimum intensity fraction</i>
            has been chosen to be 0.9, all points along the line would need to have an intensity
            of min(0.75, 0.50) * 0.9 = 0.50 * 0.9 = 0.45.<br>
            This method works well for round cells whose maximum intensity
            is in the center of the cell: a single cell that was incorrectly segmented 
            into two objects will typically not have a dim line between the centroids 
            of the two halves and will be correctly unified.</li>
            
            <li><i>Closest point:</i> This method is useful for unifying irregularly shaped cells 
            which are connected. It starts by assigning background pixels in the vicinity of the objects to the nearest
            object. Objects are then unified if each object has background pixels that are:
            <ul>
            <li>Within a distance threshold from each object;</li>
            <li>Above the minimum intensity fraction of the nearest object pixel;</li>
            <li>Adjacent to background pixels assigned to a neighboring object.</li>
            </ul>
            An example of a feature that satisfies the above constraints is a line of
            pixels that connect two neighboring objects and is roughly the same intensity 
            as the boundary pixels of both (such as an axon connecting two neurons).</li>
            </ul>
            """)

        self.shape_window_size = cps.Integer("Window size", 10,
            doc = """<i>(Used only with the Shape method)</i><br>
            Enter how big the search window should be. The window will be
            a square of (2 * window size) + 1 on each side.""")

        self.shape_threshold = cps.Float("Threshold", 0.5,
            doc = """<i>Used only with the Shape method)</i><br>
            Enter the threshold to be used. Generally this should remain at or near 0.5.""")
        
        self.wants_outlines = cps.Binary(
            "Retain outlines of the relabeled objects?", False,
            doc = """<i>(Used only if objects are output)</i><br>
            Check this setting if you want to save an image of the outlines
            of the relabeled objects.""")
        
        self.outlines_name = cps.OutlineNameProvider(
            'Name the outlines',
            'RelabeledNucleiOutlines',
            doc = """<i>(Used only if outlined are to be retained)</i><br>
            Enter a name that will allow the outlines to be selected later in the pipeline.""")

    def get_parent_choices(self,pipeline):
        columns = pipeline.get_measurement_columns()
        choices = [cps.NONE]
        for column in columns:
            object_name, feature, coltype = column[:3]
            if (object_name == self.objects_name.value and
                feature.startswith(C_PARENT)):
                choices.append(feature[(len(C_PARENT)+1):])
        return choices
    
    def validate_module(self, pipeline):
        if self.relabel_option == OPTION_UNIFY and self.unify_option == UNIFY_PARENT and self.parent_object.value == cps.NONE:
            raise cps.ValidationError(
                    '%s is not a valid object name'%cps.NONE,
                    self.parent_object)

    def settings(self):
        return [self.objects_name, self.output_objects_name,
                self.relabel_option, self.distance_threshold, 
                self.wants_image, self.image_name, 
                self.minimum_intensity_fraction,
                self.where_algorithm,
                self.wants_outlines, self.outlines_name,
                self.unify_option, self.parent_object,
                self.shape_window_size, self.shape_threshold]
    
    def visible_settings(self):
        result = [self.objects_name, self.output_objects_name,
                  self.relabel_option]
        if self.relabel_option == OPTION_UNIFY:
            result += [self.unify_option]
            if self.unify_option == UNIFY_DISTANCE:
                result += [self.distance_threshold, self.wants_image]
                if self.wants_image:
                    result += [self.image_name, self.minimum_intensity_fraction,
                               self.where_algorithm]
            elif self.unify_option == UNIFY_PARENT:
                result += [self.parent_object]
            elif self.unify_option == UNIFY_SHAPE:
                result += [self.shape_window_size, self.shape_threshold]
        result += [self.wants_outlines]
        if self.wants_outlines:
            result += [self.outlines_name]
        return result
    
    def is_interactive(self):
        return False
    
    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.objects_name.value)
        assert isinstance(objects, cpo.Objects)
        labels = objects.segmented
        if self.relabel_option == OPTION_SPLIT:
            output_labels, count = scind.label(labels > 0, np.ones((3,3),bool))
        else:
            if self.unify_option == UNIFY_DISTANCE:
                mask = labels > 0
                if self.distance_threshold.value > 0:
                    #
                    # Take the distance transform of the reverse of the mask
                    # and figure out what points are less than 1/2 of the
                    # distance from an object.
                    #
                    d = scind.distance_transform_edt(~mask)
                    mask = d < self.distance_threshold.value/2+1
                output_labels, count = scind.label(mask, np.ones((3,3), bool))
                output_labels[labels == 0] = 0
                if self.wants_image:
                    output_labels = self.filter_using_image(workspace, mask)
            elif self.unify_option == UNIFY_PARENT:
                parent_objects = workspace.object_set.get_objects(self.parent_object.value)
                output_labels = parent_objects.segmented.copy()
                output_labels[labels == 0] = 0
            elif self.unify_option == UNIFY_SHAPE:
                half_window_size = self.shape_window_size
                threshold = self.shape_threshold
                output_labels = objects.segmented.copy()
                max_objects = np.max(output_labels)
                indices = np.arange(max_objects+1)

                # Get object slices
                object_slices = morph.fixup_scipy_ndimage_result(scind.measurements.find_objects(output_labels))

                # Calculate perimeters
                perimeters = morph.calculate_perimeters(output_labels, indices)

                # It would be easier to start if I knew which objects were touching which other objects
                neighbors = np.zeros((max_objects+1, max_objects+1), bool) # neighbors[i,j] = True when object j is a neighbor of object i.
                for k in range(1, max_objects+1):
                    k_bounds_array = object_slices[k-1]
                    k_dim1_bounds = k_bounds_array[0].indices(output_labels.shape[0])
                    k_dim2_bounds = k_bounds_array[1].indices(output_labels.shape[1])
                    
                    for i in range(-1,2):
                        ki_min = 0
                        ki_max = 0
                        if k_dim1_bounds[0] + i < 0:
                            ki_min = -i
                        else:
                            ki_min = k_dim1_bounds[0]

                        if k_dim1_bounds[1] + i > output_labels.shape[0]:
                            ki_max = output_labels.shape[0] - i
                        else:
                            ki_max = k_dim1_bounds[1]
                            
                        ki_slice = slice(ki_min, ki_max)
                        ki_test_slice = slice(ki_min+i, ki_max+i)
                        
                        for j in range(-1,2):
                            if i == j == 0:
                                continue

                            kj_min = 0
                            kj_max = 0
                            
                            if k_dim2_bounds[0] + j < 0:
                                kj_min = -j
                            else:
                                kj_min = k_dim2_bounds[0]

                            if k_dim2_bounds[1] + j > output_labels.shape[1]:
                                kj_max = output_labels.shape[1] - j
                            else:
                                kj_max = k_dim2_bounds[1]

                            if kj_min == kj_max or ki_min == ki_max:
                                continue

                            kj_slice = slice(kj_min, kj_max)
                            kj_test_slice = slice(kj_min+j, kj_max+j)
                        
                            k_bounds = (ki_slice, kj_slice)
                            k_test_bounds = (ki_test_slice, kj_test_slice)
                            
                            for l in range(1, max_objects+1):
                                if k == l:
                                    continue
                                neighbors[k,l] = neighbors[k,l] or np.max(np.logical_and(output_labels[k_bounds] == k,
                                                                                         output_labels[k_test_bounds] == l))

                # Generate window dimensions for each location
                dim1_window = np.ones(output_labels.shape, int) * half_window_size
                dim2_window = np.ones(output_labels.shape, int) * half_window_size
                for i in range(0, half_window_size):
                    dim1_window[i:output_labels.shape[0]-i,0:output_labels.shape[1]] += 1
                    dim2_window[0:output_labels.shape[0],i:output_labels.shape[1]-i] += 1
                
                # Loop over all objects
                for k in range(1, max_objects+1):
                    if (perimeters[k] == 0) or not np.max(neighbors[k]): # Has been removed by a merge or has no neighbors
                        continue

                    k_bounds_array = object_slices[k-1]
                    k_dim1_bounds = k_bounds_array[0].indices(output_labels.shape[0])
                    k_dim2_bounds = k_bounds_array[1].indices(output_labels.shape[1])
                    
                    # Loop over all neighbors of object k
                    l = 0
                    while l < max_objects:
                        l += 1
                        
                        if k == l or (perimeters[l] == 0) or not neighbors[k,l]: # Has been removed by a merge or is not a neighbor of object k
                            continue

                        l_bounds_array = object_slices[l-1]
                        l_dim1_bounds = l_bounds_array[0].indices(output_labels.shape[0])
                        l_dim2_bounds = l_bounds_array[1].indices(output_labels.shape[1])

                        kl_dim1_min_bound = min(k_dim1_bounds[0], l_dim1_bounds[0])
                        kl_dim1_max_bound = max(k_dim1_bounds[1], l_dim1_bounds[1])
                        kl_dim2_min_bound = min(k_dim2_bounds[0], l_dim2_bounds[0])
                        kl_dim2_max_bound = max(k_dim2_bounds[1], l_dim2_bounds[1])

                        kl_bounds = (slice(kl_dim1_min_bound, kl_dim1_max_bound), slice(kl_dim2_min_bound, kl_dim2_max_bound))
                        kl_shape = (kl_dim1_max_bound - kl_dim1_min_bound, kl_dim2_max_bound - kl_dim2_min_bound)
                        
                        #
                        # Find the vertices
                        #
                        vertices = np.zeros(kl_shape, bool)
                        isAdjacent = np.zeros(output_labels.shape, bool)
                        isBorder = np.zeros(output_labels.shape, bool)
                        
                        for i in range(-1,2):
                            ki_min = 0
                            ki_max = 0
                            li_min = 0
                            li_max = 0
                            if k_dim1_bounds[0] + i < 0:
                                ki_min = -i
                            else:
                                ki_min = k_dim1_bounds[0]

                            if l_dim1_bounds[0] + i < 0:
                                li_min = -i
                            else:
                                li_min = l_dim1_bounds[0]

                            if k_dim1_bounds[1] + i > output_labels.shape[0]:
                                ki_max = output_labels.shape[0] - i
                            else:
                                ki_max = k_dim1_bounds[1]

                            if l_dim1_bounds[1] + i > output_labels.shape[0]:
                                li_max = output_labels.shape[0] - i
                            else:
                                li_max = l_dim1_bounds[1]

                            ki_slice = slice(ki_min, ki_max)
                            li_slice = slice(li_min, li_max)
                            ki_test_slice = slice(ki_min+i, ki_max+i)
                            li_test_slice = slice(li_min+i, li_max+i)
                            
                            for j in range(-1,2):
                                if i == j == 0:
                                    continue

                                kj_min = 0
                                kj_max = 0
                                lj_min = 0
                                lj_max = 0
                                if k_dim2_bounds[0] + j < 0:
                                    kj_min = -j
                                else:
                                    kj_min = k_dim2_bounds[0]

                                if l_dim2_bounds[0] + j < 0:
                                    lj_min = -j
                                else:
                                    lj_min = l_dim2_bounds[0]

                                if k_dim2_bounds[1] + j > output_labels.shape[0]:
                                    kj_max = output_labels.shape[0] - j
                                else:
                                    kj_max = k_dim2_bounds[1]

                                if l_dim2_bounds[1] + j > output_labels.shape[0]:
                                    lj_max = output_labels.shape[0] - j
                                else:
                                    lj_max = l_dim2_bounds[1]

                                if (kj_min == kj_max or ki_min == ki_max) and (lj_min == lj_max or li_min == li_max):
                                    continue

                                kj_slice = slice(kj_min, kj_max)
                                lj_slice = slice(lj_min, lj_max)
                                kj_test_slice = slice(kj_min+j, kj_max+j)
                                lj_test_slice = slice(lj_min+j, lj_max+j)

                                k_bounds = (ki_slice, kj_slice)
                                l_bounds = (li_slice, lj_slice)
                                k_test_bounds = (ki_test_slice, kj_test_slice)
                                l_test_bounds = (li_test_slice, lj_test_slice)
                                kl_mod_bounds = (slice(min(ki_min, li_min), max(ki_max, li_max)), slice(min(kj_min, lj_min), max(kj_max, lj_max)))

                                isAdjacentSlice = np.zeros(output_labels.shape, bool)
                                isAdjacentSlice[k_bounds] = np.logical_and(output_labels[k_bounds] == k,
                                                                           output_labels[k_test_bounds] == l)
                                isAdjacentSlice[l_bounds] = np.logical_or(isAdjacentSlice[l_bounds],
                                                                          np.logical_and(output_labels[l_bounds] == l,
                                                                                         output_labels[l_test_bounds] == k))
                                isAdjacent[kl_mod_bounds] = np.logical_or(isAdjacent[kl_mod_bounds],
                                                                          isAdjacentSlice[kl_mod_bounds])

                                isBorderSlice = np.zeros(output_labels.shape, bool)
                                isBorderSlice[k_bounds] = np.logical_and(output_labels[k_bounds] == k,
                                                                         np.logical_and(output_labels[k_test_bounds] != k,
                                                                                        output_labels[k_test_bounds] != l))
                                isBorderSlice[l_bounds] = np.logical_or(isBorderSlice[l_bounds],
                                                                        np.logical_and(output_labels[l_bounds] == l,
                                                                                       np.logical_and(output_labels[l_test_bounds] != k,
                                                                                                      output_labels[l_test_bounds] != l)))
                                isBorder[kl_mod_bounds] = np.logical_or(isBorder[kl_mod_bounds],
                                                                        isBorderSlice[kl_mod_bounds])
                        
                        vertices = np.logical_and(isAdjacent[kl_bounds], isBorder[kl_bounds])

                        #
                        # Calculate the maximum vertex score for the pair
                        #
                        '''+1 for every non-I/J labeled pixel in the window
                        +1 more for every non-I/J labeled pixel adjacent to it

                        Divided by the score that would result if the objects were flat

                        e.g. (vertex is starred)

                        0  0  0  0  0  0  0
                        0  0  0  0  0  0  0
                        1  1  0  0  0  0  0
                        1  1  1 *1* 2  2  0
                        1  1  1  1  2  2  2
                        1  1  1  1  2  2  2
                        1  1  1  2  2  2  2

                        If it were flat, it would be 7 * 3 = 21, so the divisor 7 * 6 = 42.
                        In the case that the window is rectangular (such as at the image edge),
                        we split the difference.
                        Generalized:

                        Vertex Score = (Sum of NotIJ) / [A * B - 0.5 * (A + B)]
                        '''
                        
                        sum_not_IJ = np.zeros(kl_shape, int)
                        for i in range(0, kl_shape[0]):
                            for j in range(0,kl_shape[1]):
                                if not vertices[i,j]:
                                    continue

                                window_bounds = (slice(i - half_window_size + kl_dim1_min_bound, i + half_window_size + kl_dim1_min_bound),
                                                 slice(j - half_window_size + kl_dim2_min_bound, j + half_window_size + kl_dim2_min_bound))

                                window_slice = output_labels[window_bounds]

                                sum_not_IJ[i,j] = np.count_nonzero(np.logical_and(window_slice != k,
                                                                             window_slice != l))

                        vertex_scores = sum_not_IJ / (dim1_window[kl_bounds] * dim2_window[kl_bounds] - 0.5 * (dim1_window[kl_bounds] + dim2_window[kl_bounds]))

                        merged = np.zeros(kl_shape, int)
                        merged[output_labels[kl_bounds] == k] = 1
                        merged[output_labels[kl_bounds] == l] = 1

                        merged_perimeter = morph.calculate_perimeters(merged, [1])

                        seam_length = (perimeters[k] + perimeters[l] - merged_perimeter) / 2
                        min_external_perimeter = min(perimeters[k], perimeters[l]) - seam_length
                        adjusted_min_external_perimeter = min_external_perimeter / np.pi

                        seam_fraction = seam_length / (adjusted_min_external_perimeter + seam_length)
                        
                        max_vertex_score = np.max(vertex_scores)

                        score = (max_vertex_score + seam_fraction) / 2

                        #
                        # Join object pair with score above the threshold
                        #
                        if score >= threshold:
                            output_labels[output_labels == l] = k
                            
                            # Calculate new perimeters
                            perimeters[k] = merged_perimeter
                            perimeters[l] = 0

                            # Reconfigure neighbors
                            neighbors[k] = np.logical_or(neighbors[k], neighbors[l])
                            for x in range(1, max_objects+1): # Is there an array method to do this?
                                if neighbors[x,l]:
                                    neighbors[x,k] = True
                            neighbors[0:max_objects,l] = False

                            # Recalculate bounds
                            k_dim1_bounds = kl_bounds[0].indices(output_labels.shape[0])
                            k_dim2_bounds = kl_bounds[1].indices(output_labels.shape[1])
                            
                            # Reset l to 0 so that we check all the neighbors against this new object
                            l = 0
                        else:
                            pass
                        
        output_objects = cpo.Objects()
        output_objects.segmented = output_labels
        if objects.has_small_removed_segmented:
            output_objects.small_removed_segmented = \
                copy_labels(objects.small_removed_segmented, output_labels)
        if objects.has_unedited_segmented:
            output_objects.unedited_segmented = \
                copy_labels(objects.unedited_segmented, output_labels)
        output_objects.parent_image = objects.parent_image
        workspace.object_set.add_objects(output_objects, self.output_objects_name.value)
        
        measurements = workspace.measurements
        add_object_count_measurements(measurements,
                                      self.output_objects_name.value,
                                      np.max(output_objects.segmented))
        add_object_location_measurements(measurements,
                                         self.output_objects_name.value,
                                         output_objects.segmented)
        
        #
        # Relate the output objects to the input ones and record
        # the relationship.
        #
        children_per_parent, parents_of_children = \
            objects.relate_children(output_objects)
        measurements.add_measurement(self.objects_name.value,
                                     FF_CHILDREN_COUNT % 
                                     self.output_objects_name.value,
                                     children_per_parent)
        measurements.add_measurement(self.output_objects_name.value,
                                     FF_PARENT%self.objects_name.value,
                                     parents_of_children)
        if self.wants_outlines:
            outlines = cellprofiler.cpmath.outline.outline(output_labels)
            outline_image = cpi.Image(outlines.astype(bool))
            workspace.image_set.add(self.outlines_name.value,
                                    outline_image)
                    
        if workspace.frame is not None:
            workspace.display_data.orig_labels = objects.segmented
            workspace.display_data.output_labels = output_objects.segmented        
    
    def display(self, workspace):
        '''Display the results of relabeling
        
        workspace - workspace containing saved display data
        '''
        from cellprofiler.gui.cpfigure import renumber_labels_for_display
        import matplotlib.cm as cm
        
        figure = workspace.create_or_find_figure(title="ReassignObjectNumbers, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(1,2))
        figure.subplot_imshow_labels(0,0, workspace.display_data.orig_labels,
                                     title = self.objects_name.value)
        
        output_labels = renumber_labels_for_display(
                workspace.display_data.output_labels)
        if self.relabel_option == OPTION_UNIFY:
            if self.unify_option == UNIFY_DISTANCE and self.wants_image:
                #
                # Make a nice picture which superimposes the labels on the
                # guiding image
                #
                image = (stretch(workspace.display_data.image) * 255).astype(np.uint8)
            elif self.unify_option == UNIFY_PARENT:
                parent_objects = workspace.object_set.get_objects(self.parent_object.value)
                labels = parent_objects.segmented
                image = labels.astype(float) / (1.0 if np.max(labels) == 0 else np.max(labels))
                image = (stretch(image) * 255).astype(np.uint8)
            elif self.unify_option == UNIFY_SHAPE:
                objects = workspace.object_set.get_objects(self.objects_name.value)
                labels = objects.segmented
                image = labels.astype(float) / (1.0 if np.max(labels) == 0 else np.max(labels))
                image = (stretch(image) * 255).astype(np.uint8)
            
            image = np.dstack((image,image,image))
            my_cm = cm.get_cmap(cpprefs.get_default_colormap())
            my_cm.set_bad((0,0,0))
            sm = cm.ScalarMappable(cmap=my_cm)
            m_output_labels = np.ma.array(output_labels,
                                        mask = output_labels == 0)
            output_image = sm.to_rgba(m_output_labels, bytes=True)[:,:,:3]
            image[output_labels > 0 ] = (
                image[output_labels > 0] / 4 * 3 +
                output_image[output_labels > 0,:] / 4)
        
            figure.subplot_imshow(0,1, image,
                                  title = self.output_objects_name.value,
                                  sharex = figure.subplot(0,0),
                                  sharey = figure.subplot(0,0))
        else:
            figure.subplot_imshow_labels(0,1, 
                                         workspace.display_data.output_labels,
                                         title = self.output_objects_name.value,
                                         sharex = figure.subplot(0,0),
                                         sharey = figure.subplot(0,0))

    def filter_using_image(self, workspace, mask):
        '''Filter out connections using local intensity minima between objects
        
        workspace - the workspace for the image set
        mask - mask of background points within the minimum distance
        '''
        #
        # NOTE: This is an efficient implementation and an improvement
        #       in accuracy over the Matlab version. It would be faster and
        #       more accurate to eliminate the line-connecting and instead
        #       do the following:
        #     * Distance transform to get the coordinates of the closest
        #       point in an object for points in the background that are
        #       at most 1/2 of the max distance between objects.
        #     * Take the intensity at this closest point and similarly
        #       label the background point if the background intensity
        #       is at least the minimum intensity fraction
        #     * Assume there is a connection between objects if, after this
        #       labeling, there are adjacent points in each object.
        #
        # As it is, the algorithm duplicates the Matlab version but suffers
        # for cells whose intensity isn't high in the centroid and clearly
        # suffers when two cells touch at some point that's off of the line
        # between the two.
        #
        objects = workspace.object_set.get_objects(self.objects_name.value)
        labels = objects.segmented
        image = self.get_image(workspace)
        if workspace.frame is not None:
            # Save the image for display
            workspace.display_data.image = image
        #
        # Do a distance transform into the background to label points
        # in the background with their closest foreground object
        #
        i, j = scind.distance_transform_edt(labels==0, 
                                            return_indices=True,
                                            return_distances=False)
        confluent_labels = labels[i,j]
        confluent_labels[~mask] = 0
        if self.where_algorithm == CA_CLOSEST_POINT:
            #
            # For the closest point method, find the intensity at
            # the closest point in the object (which will be the point itself
            # for points in the object).
            # 
            object_intensity = image[i,j] * self.minimum_intensity_fraction.value
            confluent_labels[object_intensity > image] = 0
        count, index, c_j = morph.find_neighbors(confluent_labels)
        if len(c_j) == 0:
            # Nobody touches - return the labels matrix
            return labels
        #
        # Make a row of i matching the touching j
        #
        c_i = np.zeros(len(c_j))
        #
        # Eliminate labels without matches
        #
        label_numbers = np.arange(1,len(count)+1)[count > 0]
        index = index[count > 0]
        count = count[count > 0]
        #
        # Get the differences between labels so we can use a cumsum trick
        # to increment to the next label when they change
        #
        label_numbers[1:] = label_numbers[1:] - label_numbers[:-1]
        c_i[index] = label_numbers
        c_i = np.cumsum(c_i).astype(int)
        if self.where_algorithm == CA_CENTROIDS:
            #
            # Only connect points > minimum intensity fraction
            #
            center_i, center_j = morph.centers_of_labels(labels)
            indexes, counts, i, j = morph.get_line_pts(
                center_i[c_i-1], center_j[c_i-1],
                center_i[c_j-1], center_j[c_j-1])
            #
            # The indexes of the centroids at pt1
            #
            last_indexes = indexes+counts-1
            #
            # The minimum of the intensities at pt0 and pt1
            #
            centroid_intensities = np.minimum(
                image[i[indexes],j[indexes]],
                image[i[last_indexes], j[last_indexes]])
            #
            # Assign label numbers to each point so we can use
            # scipy.ndimage.minimum. The label numbers are indexes into
            # "connections" above.
            #
            pt_labels = np.zeros(len(i), int)
            pt_labels[indexes[1:]] = 1
            pt_labels = np.cumsum(pt_labels)
            minima = scind.minimum(image[i,j], pt_labels, np.arange(len(indexes)))
            minima = morph.fixup_scipy_ndimage_result(minima)
            #
            # Filter the connections using the image
            #
            mif = self.minimum_intensity_fraction.value
            i = c_i[centroid_intensities * mif <= minima]
            j = c_j[centroid_intensities * mif <= minima]
        else:
            i = c_i
            j = c_j
        #
        # Add in connections from self to self
        #
        unique_labels = np.unique(labels)
        i = np.hstack((i, unique_labels))
        j = np.hstack((j, unique_labels))
        #
        # Run "all_connected_components" to get a component # for
        # objects identified as same.
        #
        new_indexes = morph.all_connected_components(i, j)
        new_labels = np.zeros(labels.shape, int)
        new_labels[labels != 0] = new_indexes[labels[labels != 0]]
        return new_labels
    
    def upgrade_settings(self,setting_values,variable_revision_number,
                         module_name,from_matlab):
        '''Adjust setting values if they came from a previous revision
        
        setting_values - a sequence of strings representing the settings
                         for the module as stored in the pipeline
        variable_revision_number - the variable revision number of the
                         module at the time the pipeline was saved. Use this
                         to determine how the incoming setting values map
                         to those of the current module version.
        module_name - the name of the module that did the saving. This can be
                      used to import the settings from another module if
                      that module was merged into the current module
        from_matlab - True if the settings came from a Matlab pipeline, False
                      if the settings are from a CellProfiler 2.0 pipeline.
        
        Overriding modules should return a tuple of setting_values,
        variable_revision_number and True if upgraded to CP 2.0, otherwise
        they should leave things as-is so that the caller can report
        an error.
        '''
        if (from_matlab and variable_revision_number == 1 and
            module_name == 'SplitIntoContiguousObjects'):
            setting_values = setting_values + [OPTION_SPLIT,'0',cps.DO_NOT_USE]
            variable_revision_number = 1
            module_name = 'RelabelObjects'
        if (from_matlab and variable_revision_number == 1 and
            module_name == 'UnifyObjects'):
            setting_values = (setting_values[:2] + [OPTION_UNIFY] + 
                              setting_values[2:])
            variable_revision_number = 1
            module_name = 'RelabelObjects'
        if (from_matlab and variable_revision_number == 1 and
            module_name == 'RelabelObjects'):
            object_name, relabeled_object_name, relabel_option, \
                   distance_threshold, grayscale_image_name = setting_values
            wants_image = (cps.NO if grayscale_image_name == cps.DO_NOT_USE
                           else cps.YES)
            setting_values = [object_name, relabeled_object_name,
                              relabel_option, distance_threshold,
                              wants_image, grayscale_image_name,
                              "0.9", CA_CENTROIDS]
            from_matlab = False
            variable_revision_number = 1
            
        if (not from_matlab) and variable_revision_number == 1:
            # Added outline options
            setting_values += [cps.NO, "RelabeledNucleiOutlines"]
            variable_revision_number = 2
            
        if (not from_matlab) and variable_revision_number == 2:
            # Added per-parent unification
            setting_values += [UNIFY_DISTANCE, cps.NONE]
            variable_revision_number = 3

        if (not from_matlab) and variable_revision_number == 3:
            # Added shape-based unification
            setting_values += []
                       
        return setting_values, variable_revision_number, from_matlab
    
    def get_image(self, workspace):
        '''Get the image for image-directed merging'''
        objects = workspace.object_set.get_objects(self.objects_name.value)
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_grayscale=True)
        image = objects.crop_image_similarly(image.pixel_data)
        return image
        
    def get_measurement_columns(self, pipeline):
        columns =  get_object_measurement_columns(self.output_objects_name.value)
        columns += [(self.output_objects_name.value,
                     FF_PARENT % self.objects_name.value,
                     cpmeas.COLTYPE_INTEGER),
                    (self.objects_name.value,
                     FF_CHILDREN_COUNT % self.output_objects_name.value,
                     cpmeas.COLTYPE_INTEGER)]
        return columns
    
    def get_categories(self,pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        if object_name == 'Image':
            return ['Count']
        elif object_name == self.output_objects_name.value:
            return ['Location','Parent','Number']
        elif object_name == self.objects_name.value:
            return ['Children']
        return []
      
    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        if object_name == 'Image' and category == 'Count':
            return [ self.output_objects_name.value ]
        elif object_name == self.output_objects_name.value and category == 'Location':
            return ['Center_X','Center_Y']
        elif object_name == self.output_objects_name.value and category == 'Parent':
            return [ self.objects_name.value]
        elif object_name == self.output_objects_name.value and category == 'Number':
            return ['Object_Number']
        elif object_name == self.objects_name.value and category == 'Children':
            return [ "%s_Count" % self.output_objects_name.value]
        return []

def copy_labels(labels, segmented):
    '''Carry differences between orig_segmented and new_segmented into "labels"
    
    labels - labels matrix similarly segmented to "segmented"
    segmented - the newly numbered labels matrix (a subset of pixels are labeled)
    '''
    max_labels = np.max(segmented)
    seglabel = scind.minimum(labels, segmented, np.arange(1, max_labels+1))
    labels_new = labels.copy()
    labels_new[segmented != 0] = seglabel[segmented[segmented != 0] - 1]
    return labels_new

RelabelObjects = ReassignObjectNumbers
