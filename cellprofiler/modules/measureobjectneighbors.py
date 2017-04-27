'''<b>Measure Object Neighbors</b> calculates how many neighbors each
object has and records various properties about the neighbors' relationships,
including the percentage of an object's edge pixels that touch a neighbor.
<hr>
Given an image with objects identified (e.g., nuclei or cells), this
module determines how many neighbors each object has. You can specify
the distance within which objects should be considered neighbors, or
that objects are only considered neighbors if they are directly touching.

<h4>Available measurements</h4>
<b>Object measurements</b>
<ul>
<li><i>NumberOfNeighbors:</i> Number of neighbor objects.</li>
<li><i>PercentTouching:</i> Percent of the object's boundary pixels that touch
neighbors, after the objects have been expanded to the specified distance.
Note: This measurement is only available if you use the same set of objects
for both objects and neighbors.</li>
<li><i>FirstClosestObjectNumber:</i> The index of the closest object.</li>
<li><i>FirstClosestDistance:</i> The distance to the closest object.</li>
<li><i>SecondClosestObjectNumber:</i> The index of the second closest object.</li>
<li><i>SecondClosestDistance:</i> The distance to the second closest object.</li>
<li><i>AngleBetweenNeighbors:</i> The angle formed with the object center as the
vertex and the first and second closest object centers along the vectors.</li>
</ul>

<b>Object relationships:</b> The identity of the neighboring objects, for
each object. Since per-object output is one-to-one and neighbors relationships
are often many-to-one, they may be saved as a separate file in
<b>ExportToSpreadsheet</b> by selecting <i>Object
relationships</i> from the list of objects to export.

<h4>Technical notes</h4>
Objects discarded via modules such as <b>IdentifyPrimaryObjects</b> or
<b>IdentifySecondaryObjects</b> will still register as a neighbors for the purposes
of accurate measurement. For instance, if an object touches a single object and
that object had been discarded, <i>NumberOfNeighbors</i> will be positive, but
there will not be a corresponding <i>ClosestObjectNumber</i>.

See also the <b>Identify</b> modules.
'''

import matplotlib.cm
import numpy as np
import scipy.ndimage as scind
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.cpmorphology import strel_disk, centers_of_labels
from centrosome.outline import outline

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.setting as cps
import cellprofiler.workspace as cpw
from cellprofiler.setting import YES, NO
import skimage.morphology

D_ADJACENT = 'Adjacent'
D_EXPAND = 'Expand until adjacent'
D_WITHIN = 'Within a specified distance'
D_ALL = [D_ADJACENT, D_EXPAND, D_WITHIN]

M_NUMBER_OF_NEIGHBORS = 'NumberOfNeighbors'
M_PERCENT_TOUCHING = 'PercentTouching'
M_FIRST_CLOSEST_OBJECT_NUMBER = 'FirstClosestObjectNumber'
M_FIRST_CLOSEST_DISTANCE = 'FirstClosestDistance'
M_SECOND_CLOSEST_OBJECT_NUMBER = 'SecondClosestObjectNumber'
M_SECOND_CLOSEST_DISTANCE = 'SecondClosestDistance'
M_ANGLE_BETWEEN_NEIGHBORS = 'AngleBetweenNeighbors'
M_ALL = [M_NUMBER_OF_NEIGHBORS, M_PERCENT_TOUCHING,
         M_FIRST_CLOSEST_OBJECT_NUMBER, M_FIRST_CLOSEST_DISTANCE,
         M_SECOND_CLOSEST_OBJECT_NUMBER, M_SECOND_CLOSEST_DISTANCE,
         M_ANGLE_BETWEEN_NEIGHBORS]

C_NEIGHBORS = 'Neighbors'

S_EXPANDED = 'Expanded'
S_ADJACENT = 'Adjacent'


class MeasureObjectNeighbors(cpm.Module):
    module_name = 'MeasureObjectNeighbors'
    category = "Measurement"
    variable_revision_number = 2

    def create_settings(self):
        self.object_name = cps.ObjectNameSubscriber(
                'Select objects to measure', cps.NONE, doc="""
            Select the objects whose neighbors you want to measure.""")

        self.neighbors_name = cps.ObjectNameSubscriber(
                'Select neighboring objects to measure', cps.NONE, doc="""
            This is the name of the objects that are potential
            neighbors of the above objects. You can find the neighbors
            within the same set of objects by selecting the same objects
            as above.""")

        self.distance_method = cps.Choice(
                'Method to determine neighbors',
                D_ALL, D_EXPAND, doc="""
            There are several methods by which to determine whether objects are neighbors:
            <ul>
            <li><i>%(D_ADJACENT)s:</i> In this mode, two objects must have adjacent
            boundary pixels to be neighbors. </li>
            <li><i>%(D_EXPAND)s:</i> The objects are expanded until all
            pixels on the object boundaries are touching another. Two objects are
            neighbors if any of their boundary pixels are adjacent after
            expansion.</li>
            <li><i>%(D_WITHIN)s:</i> Each object is expanded by
            the number of pixels you specify. Two objects are
            neighbors if they have adjacent pixels after expansion. </li>
            </ul>

            <p>For <i>%(D_ADJACENT)s</i> and <i>%(D_EXPAND)s</i>, the
            <i>%(M_PERCENT_TOUCHING)s</i> measurement is the percentage of pixels on the boundary
            of an object that touch adjacent objects. For <i>%(D_WITHIN)s</i>,
            two objects are touching if any of their boundary
            pixels are adjacent after expansion and <i>%(M_PERCENT_TOUCHING)s</i> measures the
            percentage of boundary pixels of an <i>expanded</i> object that
            touch adjacent objects.</p>""" % globals())

        self.distance = cps.Integer(
                'Neighbor distance', 5, 1, doc="""
            <i>(Used only when "%(D_WITHIN)s" is selected)</i> <br>
            The Neighbor distance is the number of pixels that each object is
            expanded for the neighbor calculation. Expanded objects that touch
            are considered neighbors.""" % globals())

        self.wants_count_image = cps.Binary(
                'Retain the image of objects colored by numbers of neighbors?',
                False, doc="""
            An output image showing the input objects
            colored by numbers of neighbors may be retained. A colormap of your choice shows
            how many neighbors each object has. The background is set
            to -1. Objects are colored with an increasing color value
            corresponding to the number of neighbors, such that objects with no
            neighbors are given a color corresponding to 0. Use the <b>SaveImages</b>
            module to save this image to a file.""")

        self.count_image_name = cps.ImageNameProvider(
                'Name the output image',
                'ObjectNeighborCount', doc="""
            <i>(Used only if the image of objects colored by numbers of neighbors
            is to be retained for later use in the pipeline)</i> <br>
            Specify a name
            that will allow the image of objects colored by numbers of neighbors
            to be selected later in the pipeline.""")

        self.count_colormap = cps.Colormap(
                'Select colormap', doc="""
            <i>(Used only if the image of objects colored by numbers of neighbors
            is to be retained for later use in the pipeline)</i> <br>
            Select the colormap to use to color the neighbor number image. All available colormaps can be seen
            <a href="http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps">here</a>.""")

        self.wants_percent_touching_image = cps.Binary(
                'Retain the image of objects colored by percent of touching pixels?',
                False, doc="""
            Select <i>%(YES)s</i> to keep an image of the input objects
            colored by the percentage of the boundary touching their neighbors.
            A colormap of your choice is used to show the touching percentage of
            each object. Use the <b>SaveImages</b> module to save this image to a file.""" % globals())

        self.touching_image_name = cps.ImageNameProvider(
                'Name the output image',
                'PercentTouching', doc="""
            <i>(Used only if the image of objects colored by percent touching
            is to be retained for later use in the pipeline)</i> <br>
            Specify a name that will allow the image of objects colored by percent of touching
            pixels to be selected later in the pipeline.""")

        self.touching_colormap = cps.Colormap(
                'Select a colormap', doc="""
            <i>(Used only if the image of objects colored by percent touching
            is to be retained for later use in the pipeline)</i> <br>
            Select the colormap to use to color the percent touching image. All available colormaps can be seen
            <a href="http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps">here</a>.""")

    def settings(self):
        return [self.object_name, self.neighbors_name,
                self.distance_method, self.distance,
                self.wants_count_image, self.count_image_name,
                self.count_colormap, self.wants_percent_touching_image,
                self.touching_image_name, self.touching_colormap]

    def visible_settings(self):
        result = [self.object_name, self.neighbors_name, self.distance_method]
        if self.distance_method == D_WITHIN:
            result += [self.distance]
        result += [self.wants_count_image]
        if self.wants_count_image.value:
            result += [self.count_image_name, self.count_colormap]
        if self.neighbors_are_objects:
            result += [self.wants_percent_touching_image]
            if self.wants_percent_touching_image.value:
                result += [self.touching_image_name, self.touching_colormap]
        return result

    @property
    def neighbors_are_objects(self):
        '''True if the neighbors are taken from the same object set as objects'''
        return self.object_name.value == self.neighbors_name.value

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.object_name.value)
        dimensions = len(objects.shape)
        assert isinstance(objects, cpo.Objects)
        has_pixels = objects.areas > 0
        labels = objects.small_removed_segmented
        kept_labels = objects.segmented
        neighbor_objects = workspace.object_set.get_objects(
                self.neighbors_name.value)
        assert isinstance(neighbor_objects, cpo.Objects)
        neighbor_labels = neighbor_objects.small_removed_segmented
        #
        # Need to add in labels touching border.
        #
        unedited_segmented = neighbor_objects.unedited_segmented
        touching_border = np.zeros(np.max(unedited_segmented) + 1, bool)
        touching_border[unedited_segmented[0, :]] = True
        touching_border[unedited_segmented[-1, :]] = True
        touching_border[unedited_segmented[:, 0]] = True
        touching_border[unedited_segmented[:, -1]] = True
        touching_border[0] = False
        touching_border_mask = touching_border[unedited_segmented]
        nobjects = np.max(labels)
        nkept_objects = objects.count
        nneighbors = np.max(neighbor_labels)
        if np.any(touching_border) and \
                np.all(~ touching_border_mask[neighbor_labels != 0]):
            # Add the border labels if any were excluded
            touching_border_object_number = np.cumsum(touching_border) + \
                                            np.max(neighbor_labels)
            touching_border_mask = touching_border_mask & (neighbor_labels == 0)
            neighbor_labels = neighbor_labels.copy().astype(np.int32)
            neighbor_labels[touching_border_mask] = touching_border_object_number[
                unedited_segmented[touching_border_mask]]

        _, object_numbers = objects.relate_labels(labels, kept_labels)
        if self.neighbors_are_objects:
            neighbor_numbers = object_numbers
            neighbor_has_pixels = has_pixels
        else:
            _, neighbor_numbers = neighbor_objects.relate_labels(
                    neighbor_labels, neighbor_objects.segmented)
            neighbor_has_pixels = np.bincount(neighbor_labels.ravel())[1:] > 0
        neighbor_count = np.zeros((nobjects,))
        pixel_count = np.zeros((nobjects,))
        first_object_number = np.zeros((nobjects,), int)
        second_object_number = np.zeros((nobjects,), int)
        first_x_vector = np.zeros((nobjects,))
        second_x_vector = np.zeros((nobjects,))
        first_y_vector = np.zeros((nobjects,))
        second_y_vector = np.zeros((nobjects,))
        angle = np.zeros((nobjects,))
        percent_touching = np.zeros((nobjects,))
        expanded_labels = None
        if self.distance_method == D_EXPAND:
            # Find the i,j coordinates of the nearest foreground point
            # to every background point
            if dimensions == 2:
                i, j = scind.distance_transform_edt(labels == 0,
                                                    return_distances=False,
                                                    return_indices=True)
                # Assign each background pixel to the label of its nearest
                # foreground pixel. Assign label to label for foreground.
                labels = labels[i, j]
            else:
                k, i, j = scind.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
                labels = labels[k, i, j]
            expanded_labels = labels  # for display
            distance = 1  # dilate once to make touching edges overlap
            scale = S_EXPANDED
            if self.neighbors_are_objects:
                neighbor_labels = labels.copy()
        elif self.distance_method == D_WITHIN:
            distance = self.distance.value
            scale = str(distance)
        elif self.distance_method == D_ADJACENT:
            distance = 1
            scale = S_ADJACENT
        else:
            raise ValueError("Unknown distance method: %s" %
                             self.distance_method.value)
        if nneighbors > (1 if self.neighbors_are_objects else 0):
            first_objects = []
            second_objects = []
            object_indexes = np.arange(nobjects, dtype=np.int32) + 1
            #
            # First, compute the first and second nearest neighbors,
            # and the angles between self and the first and second
            # nearest neighbors
            #
            ocenters = centers_of_labels(
                    objects.small_removed_segmented).transpose()
            ncenters = centers_of_labels(
                    neighbor_objects.small_removed_segmented).transpose()
            areas = fix(scind.sum(np.ones(labels.shape), labels, object_indexes))
            perimeter_outlines = outline(labels)
            perimeters = fix(scind.sum(
                    np.ones(labels.shape), perimeter_outlines, object_indexes))

            i, j = np.mgrid[0:nobjects, 0:nneighbors]
            distance_matrix = np.sqrt((ocenters[i, 0] - ncenters[j, 0]) ** 2 +
                                      (ocenters[i, 1] - ncenters[j, 1]) ** 2)
            #
            # order[:,0] should be arange(nobjects)
            # order[:,1] should be the nearest neighbor
            # order[:,2] should be the next nearest neighbor
            #
            if distance_matrix.shape[1] == 1:
                # a little buggy, lexsort assumes that a 2-d array of
                # second dimension = 1 is a 1-d array
                order = np.zeros(distance_matrix.shape, int)
            else:
                order = np.lexsort([distance_matrix])
            first_neighbor = 1 if self.neighbors_are_objects else 0
            first_object_index = order[:, first_neighbor]
            first_x_vector = ncenters[first_object_index, 1] - ocenters[:, 1]
            first_y_vector = ncenters[first_object_index, 0] - ocenters[:, 0]
            if nneighbors > first_neighbor + 1:
                second_object_index = order[:, first_neighbor + 1]
                second_x_vector = ncenters[second_object_index, 1] - ocenters[:, 1]
                second_y_vector = ncenters[second_object_index, 0] - ocenters[:, 0]
                v1 = np.array((first_x_vector, first_y_vector))
                v2 = np.array((second_x_vector, second_y_vector))
                #
                # Project the unit vector v1 against the unit vector v2
                #
                dot = (np.sum(v1 * v2, 0) /
                       np.sqrt(np.sum(v1 ** 2, 0) * np.sum(v2 ** 2, 0)))
                angle = np.arccos(dot) * 180. / np.pi

            # Make the structuring element for dilation
            if dimensions == 2:
                strel = strel_disk(distance)
            else:
                strel = skimage.morphology.ball(distance)
            #
            # A little bigger one to enter into the border with a structure
            # that mimics the one used to create the outline
            #
            if dimensions == 2:
                strel_touching = strel_disk(distance + .5)
            else:
                strel_touching = skimage.morphology.ball(distance + 0.5)
            #
            # Get the extents for each object and calculate the patch
            # that excises the part of the image that is "distance"
            # away
            if dimensions == 2:
                i, j = np.mgrid[0:labels.shape[0], 0:labels.shape[1]]
                min_i, max_i, min_i_pos, max_i_pos = \
                    scind.extrema(i, labels, object_indexes)
                min_j, max_j, min_j_pos, max_j_pos = \
                    scind.extrema(j, labels, object_indexes)
                min_i = np.maximum(fix(min_i) - distance, 0).astype(int)
                max_i = np.minimum(fix(max_i) + distance + 1, labels.shape[0]).astype(int)
                min_j = np.maximum(fix(min_j) - distance, 0).astype(int)
                max_j = np.minimum(fix(max_j) + distance + 1, labels.shape[1]).astype(int)
            else:
                k, i, j = np.mgrid[0:labels.shape[0], 0:labels.shape[1], 0:labels.shape[2]]
                min_k, max_k, min_k_pos, max_k_pos = scind.extrema(k, labels, object_indexes)
                min_i, max_i, min_i_pos, max_i_pos = scind.extrema(i, labels, object_indexes)
                min_j, max_j, min_j_pos, max_j_pos = scind.extrema(j, labels, object_indexes)
                min_k = np.maximum(fix(min_k) - distance, 0).astype(int)
                max_k = np.minimum(fix(max_k) + distance + 1, labels.shape[0]).astype(int)
                min_i = np.maximum(fix(min_i) - distance, 0).astype(int)
                max_i = np.minimum(fix(max_i) + distance + 1, labels.shape[1]).astype(int)
                min_j = np.maximum(fix(min_j) - distance, 0).astype(int)
                max_j = np.minimum(fix(max_j) + distance + 1, labels.shape[2]).astype(int)
            #
            # Loop over all objects
            # Calculate which ones overlap "index"
            # Calculate how much overlap there is of others to "index"
            #
            for object_number in object_numbers:
                if object_number == 0:
                    #
                    # No corresponding object in small-removed. This means
                    # that the object has no pixels, e.g. not renumbered.
                    #
                    continue
                index = object_number - 1
                if dimensions == 2:
                    patch = labels[min_i[index]:max_i[index],
                            min_j[index]:max_j[index]]
                    npatch = neighbor_labels[min_i[index]:max_i[index],
                             min_j[index]:max_j[index]]
                else:
                    patch = labels[min_k[index]:max_k[index], min_i[index]:max_i[index], min_j[index]:max_j[index]]
                    npatch = neighbor_labels[min_k[index]:max_k[index], min_i[index]:max_i[index], min_j[index]:max_j[index]]

                #
                # Find the neighbors
                #
                patch_mask = patch == (index + 1)
                extended = scind.binary_dilation(patch_mask, strel)
                neighbors = np.unique(npatch[extended])
                neighbors = neighbors[neighbors != 0]
                if self.neighbors_are_objects:
                    neighbors = neighbors[neighbors != object_number]
                nc = len(neighbors)
                neighbor_count[index] = nc
                if nc > 0:
                    first_objects.append(np.ones(nc, int) * object_number)
                    second_objects.append(neighbors)
                if self.neighbors_are_objects:
                    #
                    # Find the # of overlapping pixels. Dilate the neighbors
                    # and see how many pixels overlap our image. Use a 3x3
                    # structuring element to expand the overlapping edge
                    # into the perimeter.
                    #
                    if dimensions == 2:
                        outline_patch = perimeter_outlines[
                                        min_i[index]:max_i[index],
                                        min_j[index]:max_j[index]] == object_number
                    else:
                        outline_patch = perimeter_outlines[min_k[index]:max_k[index], min_i[index]:max_i[index], min_j[index]:max_j[index]] == object_number

                    extended = scind.binary_dilation(
                            (patch != 0) & (patch != object_number), strel_touching)
                    overlap = np.sum(outline_patch & extended)
                    pixel_count[index] = overlap
            if sum([len(x) for x in first_objects]) > 0:
                first_objects = np.hstack(first_objects)
                reverse_object_numbers = np.zeros(
                        max(np.max(object_numbers), np.max(first_objects)) + 1, int)
                reverse_object_numbers[object_numbers] = np.arange(len(object_numbers)) + 1
                first_objects = reverse_object_numbers[first_objects]

                second_objects = np.hstack(second_objects)
                reverse_neighbor_numbers = np.zeros(
                        max(np.max(neighbor_numbers), np.max(second_objects)) + 1, int)
                reverse_neighbor_numbers[neighbor_numbers] = np.arange(len(neighbor_numbers)) + 1
                second_objects = reverse_neighbor_numbers[second_objects]
                to_keep = (first_objects > 0) & (second_objects > 0)
                first_objects = first_objects[to_keep]
                second_objects = second_objects[to_keep]
            else:
                first_objects = np.zeros(0, int)
                second_objects = np.zeros(0, int)
            if self.neighbors_are_objects:
                percent_touching = pixel_count * 100 / perimeters
            else:
                percent_touching = pixel_count * 100.0 / areas
            object_indexes = object_numbers - 1
            neighbor_indexes = neighbor_numbers - 1
            #
            # Have to recompute nearest
            #
            first_object_number = np.zeros(nkept_objects, int)
            second_object_number = np.zeros(nkept_objects, int)
            if nkept_objects > (1 if self.neighbors_are_objects else 0):
                di = (ocenters[object_indexes[:, np.newaxis], 0] -
                      ncenters[neighbor_indexes[np.newaxis, :], 0])
                dj = (ocenters[object_indexes[:, np.newaxis], 1] -
                      ncenters[neighbor_indexes[np.newaxis, :], 1])
                distance_matrix = np.sqrt(di * di + dj * dj)
                distance_matrix[~ has_pixels, :] = np.inf
                distance_matrix[:, ~neighbor_has_pixels] = np.inf
                #
                # order[:,0] should be arange(nobjects)
                # order[:,1] should be the nearest neighbor
                # order[:,2] should be the next nearest neighbor
                #
                order = np.lexsort([distance_matrix]).astype(
                        first_object_number.dtype)
                if self.neighbors_are_objects:
                    first_object_number[has_pixels] = order[has_pixels, 1] + 1
                    if nkept_objects > 2:
                        second_object_number[has_pixels] = order[has_pixels, 2] + 1
                else:
                    first_object_number[has_pixels] = order[has_pixels, 0] + 1
                    if order.shape[1] > 1:
                        second_object_number[has_pixels] = order[has_pixels, 1] + 1
        else:
            object_indexes = object_numbers - 1
            neighbor_indexes = neighbor_numbers - 1
            first_objects = np.zeros(0, int)
            second_objects = np.zeros(0, int)
        #
        # Now convert all measurements from the small-removed to
        # the final number set.
        #
        neighbor_count = neighbor_count[object_indexes]
        neighbor_count[~ has_pixels] = 0
        percent_touching = percent_touching[object_indexes]
        percent_touching[~ has_pixels] = 0
        first_x_vector = first_x_vector[object_indexes]
        second_x_vector = second_x_vector[object_indexes]
        first_y_vector = first_y_vector[object_indexes]
        second_y_vector = second_y_vector[object_indexes]
        angle = angle[object_indexes]
        #
        # Record the measurements
        #
        assert (isinstance(workspace, cpw.Workspace))
        m = workspace.measurements
        assert (isinstance(m, cpmeas.Measurements))
        image_set = workspace.image_set
        features_and_data = [
            (M_NUMBER_OF_NEIGHBORS, neighbor_count),
            (M_FIRST_CLOSEST_OBJECT_NUMBER, first_object_number),
            (M_FIRST_CLOSEST_DISTANCE, np.sqrt(first_x_vector ** 2 + first_y_vector ** 2)),
            (M_SECOND_CLOSEST_OBJECT_NUMBER, second_object_number),
            (M_SECOND_CLOSEST_DISTANCE, np.sqrt(second_x_vector ** 2 + second_y_vector ** 2)),
            (M_ANGLE_BETWEEN_NEIGHBORS, angle)]
        if self.neighbors_are_objects:
            features_and_data.append((M_PERCENT_TOUCHING, percent_touching))
        for feature_name, data in features_and_data:
            m.add_measurement(self.object_name.value,
                              self.get_measurement_name(feature_name),
                              data)
        if len(first_objects) > 0:
            m.add_relate_measurement(
                    self.module_num,
                    cpmeas.NEIGHBORS,
                    self.object_name.value,
                    self.object_name.value if self.neighbors_are_objects
                    else self.neighbors_name.value,
                    m.image_set_number * np.ones(first_objects.shape, int),
                    first_objects,
                    m.image_set_number * np.ones(second_objects.shape, int),
                    second_objects)

        labels = kept_labels

        neighbor_count_image = np.zeros(labels.shape, int)
        object_mask = objects.segmented != 0
        object_indexes = objects.segmented[object_mask] - 1
        neighbor_count_image[object_mask] = neighbor_count[object_indexes]
        workspace.display_data.neighbor_count_image = neighbor_count_image

        if self.neighbors_are_objects:
            percent_touching_image = np.zeros(labels.shape)
            percent_touching_image[object_mask] = percent_touching[object_indexes]
            workspace.display_data.percent_touching_image = percent_touching_image

        image_set = workspace.image_set
        if self.wants_count_image.value:
            neighbor_cm_name = self.count_colormap.value
            neighbor_cm = get_colormap(neighbor_cm_name)
            sm = matplotlib.cm.ScalarMappable(cmap=neighbor_cm)
            img = sm.to_rgba(neighbor_count_image)[:, :, :3]
            img[:, :, 0][~ object_mask] = 0
            img[:, :, 1][~ object_mask] = 0
            img[:, :, 2][~ object_mask] = 0
            count_image = cpi.Image(img, masking_objects=objects)
            image_set.add(self.count_image_name.value, count_image)
        else:
            neighbor_cm_name = cpprefs.get_default_colormap()
            neighbor_cm = matplotlib.cm.get_cmap(neighbor_cm_name)
        if self.neighbors_are_objects and self.wants_percent_touching_image:
            percent_touching_cm_name = self.touching_colormap.value
            percent_touching_cm = get_colormap(percent_touching_cm_name)
            sm = matplotlib.cm.ScalarMappable(cmap=percent_touching_cm)
            img = sm.to_rgba(percent_touching_image)[:, :, :3]
            img[:, :, 0][~ object_mask] = 0
            img[:, :, 1][~ object_mask] = 0
            img[:, :, 2][~ object_mask] = 0
            touching_image = cpi.Image(img, masking_objects=objects)
            image_set.add(self.touching_image_name.value,
                          touching_image)
        else:
            percent_touching_cm_name = cpprefs.get_default_colormap()
            percent_touching_cm = matplotlib.cm.get_cmap(percent_touching_cm_name)

        if self.show_window:
            workspace.display_data.neighbor_cm_name = neighbor_cm_name
            workspace.display_data.percent_touching_cm_name = percent_touching_cm_name
            workspace.display_data.orig_labels = objects.segmented
            workspace.display_data.expanded_labels = expanded_labels
            workspace.display_data.object_mask = object_mask
            workspace.display_data.dimensions = dimensions

    def display(self, workspace, figure):
        dimensions = workspace.display_data.dimensions
        figure.set_subplots((2, 2), dimensions=dimensions)
        figure.subplot_imshow_labels(0, 0, workspace.display_data.orig_labels,
                                     "Original: %s" % self.object_name.value,
                                     dimensions=dimensions)

        object_mask = workspace.display_data.object_mask
        expanded_labels = workspace.display_data.expanded_labels
        neighbor_count_image = workspace.display_data.neighbor_count_image
        neighbor_count_image[~ object_mask] = -1
        neighbor_cm = get_colormap(workspace.display_data.neighbor_cm_name)
        neighbor_cm.set_under((0, 0, 0))
        neighbor_cm = matplotlib.cm.ScalarMappable(cmap=neighbor_cm)
        if self.neighbors_are_objects:
            percent_touching_cm = \
                get_colormap(workspace.display_data.percent_touching_cm_name)
            percent_touching_cm.set_under((0, 0, 0))
            percent_touching_image = workspace.display_data.percent_touching_image
            percent_touching_image[~ object_mask] = -1
            percent_touching_cm = \
                matplotlib.cm.ScalarMappable(cmap=percent_touching_cm)
        if np.any(object_mask):
            figure.subplot_imshow(0, 1, neighbor_count_image,
                                  "%s colored by # of neighbors" %
                                  self.object_name.value,
                                  colormap=neighbor_cm,
                                  colorbar=True, vmin=0,
                                  vmax=max(neighbor_count_image.max(), 1),
                                  normalize=False,
                                  # sharexy=figure.subplot(0, 0),
                                  dimensions=dimensions)
            if self.neighbors_are_objects:
                figure.subplot_imshow(1, 1, percent_touching_image,
                                      "%s colored by pct touching" %
                                      self.object_name.value,
                                      colormap=percent_touching_cm,
                                      colorbar=True, vmin=0,
                                      vmax=max(percent_touching_image.max(), 1),
                                      normalize=False,
                                      # sharexy=figure.subplot(0, 0),
                                      dimensions=dimensions)
        else:
            # No objects - colorbar blows up.
            figure.subplot_imshow(0, 1, neighbor_count_image,
                                  "%s colored by # of neighbors" %
                                  self.object_name.value,
                                  colormap=neighbor_cm,
                                  vmin=0,
                                  vmax=max(neighbor_count_image.max(), 1),
                                  # sharexy=figure.subplot(0, 0),
                                  dimensions=dimensions)
            if self.neighbors_are_objects:
                figure.subplot_imshow(1, 1, percent_touching_image,
                                      "%s colored by pct touching" %
                                      self.object_name.value,
                                      colormap=percent_touching_cm,
                                      vmin=0,
                                      vmax=max(neighbor_count_image.max(), 1),
                                      # sharexy=figure.subplot(0, 0),
                                      dimensions=dimensions)

        if self.distance_method == D_EXPAND:
            figure.subplot_imshow_labels(1, 0, expanded_labels,
                                         "Expanded %s" %
                                         self.object_name.value,
                                         # sharexy=figure.subplot(0, 0),
                                         dimensions=dimensions)

    @property
    def all_features(self):
        if self.neighbors_are_objects:
            return M_ALL
        else:
            return filter(lambda x: x != M_PERCENT_TOUCHING, M_ALL)

    def get_measurement_name(self, feature):
        if self.distance_method == D_EXPAND:
            scale = S_EXPANDED
        elif self.distance_method == D_WITHIN:
            scale = str(self.distance.value)
        elif self.distance_method == D_ADJACENT:
            scale = S_ADJACENT
        if self.neighbors_are_objects:
            return "_".join((C_NEIGHBORS, feature, scale))
        else:
            return "_".join((C_NEIGHBORS, feature,
                             self.neighbors_name.value, scale))

    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        coltypes = dict([(feature,
                          cpmeas.COLTYPE_INTEGER
                          if feature in (M_NUMBER_OF_NEIGHBORS,
                                         M_FIRST_CLOSEST_OBJECT_NUMBER,
                                         M_SECOND_CLOSEST_OBJECT_NUMBER)
                          else cpmeas.COLTYPE_FLOAT)
                         for feature in self.all_features])
        return [(self.object_name.value,
                 self.get_measurement_name(feature_name),
                 coltypes[feature_name])
                for feature_name in self.all_features]

    def get_object_relationships(self, pipeline):
        '''Return column definitions for object relationships output by module'''
        objects_name = self.object_name.value
        if self.neighbors_are_objects:
            neighbors_name = objects_name
        else:
            neighbors_name = self.neighbors_name.value
        return [(cpmeas.NEIGHBORS, objects_name, neighbors_name,
                 cpmeas.MCA_AVAILABLE_EACH_CYCLE)]

    def get_categories(self, pipeline, object_name):
        if object_name == self.object_name:
            return [C_NEIGHBORS]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.object_name and category == C_NEIGHBORS:
            return filter(lambda x: (x is not M_PERCENT_TOUCHING
                                     or self.neighbors_are_objects), M_ALL)
        return []

    def get_measurement_objects(self, pipeline, object_name, category,
                                measurement):
        if (self.neighbors_are_objects or
                    measurement not in self.get_measurements(pipeline, object_name, category)):
            return []
        return [self.neighbors_name.value]

    def get_measurement_scales(self, pipeline, object_name, category, measurement, image_name):
        if measurement in self.get_measurements(pipeline, object_name, category):
            if self.distance_method == D_EXPAND:
                return [S_EXPANDED]
            elif self.distance_method == D_ADJACENT:
                return [S_ADJACENT]
            elif self.distance_method == D_WITHIN:
                return [str(self.distance.value)]
            else:
                raise ValueError("Unknown distance method: %s" %
                                 self.distance_method.value)
        return []

    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        if from_matlab and variable_revision_number == 5:
            wants_image = setting_values[2] != cps.DO_NOT_USE
            distance_method = D_EXPAND if setting_values[1] == "0" else D_WITHIN
            setting_values = [setting_values[0],
                              distance_method,
                              setting_values[1],
                              cps.YES if wants_image else cps.NO,
                              setting_values[2],
                              cps.DEFAULT,
                              cps.NO,
                              "PercentTouching",
                              cps.DEFAULT]
            from_matlab = False
            variable_revision_number = 1
        if variable_revision_number == 1:
            # Added neighbor objects
            # To upgrade, repeat object_name twice
            #
            setting_values = setting_values[:1] * 2 + setting_values[1:]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

    def volumetric(self):
        return True


def get_colormap(name):
    '''Get colormap, accounting for possible request for default'''
    if name == cps.DEFAULT:
        name = cpprefs.get_default_colormap()
    return matplotlib.cm.get_cmap(name)
