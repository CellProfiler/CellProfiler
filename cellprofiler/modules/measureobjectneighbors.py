"""<b>Measure Object Neighbors</b> calculates how many neighbors each
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
"""

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.setting
import cellprofiler.workspace
import centrosome.cpmorphology
import centrosome.outline
import matplotlib.cm
import numpy
import scipy.ndimage

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


class MeasureObjectNeighbors(cellprofiler.module.Module):
    module_name = 'MeasureObjectNeighbors'
    category = "Measurement"
    variable_revision_number = 2

    def create_settings(self):
        self.object_name = cellprofiler.setting.ObjectNameSubscriber(
                'Select objects to measure', cellprofiler.setting.NONE, doc="""
            Select the objects whose neighbors you want to measure.""")

        self.neighbors_name = cellprofiler.setting.ObjectNameSubscriber(
                'Select neighboring objects to measure', cellprofiler.setting.NONE, doc="""
            This is the name of the objects that are potential
            neighbors of the above objects. You can find the neighbors
            within the same set of objects by selecting the same objects
            as above.""")

        self.distance_method = cellprofiler.setting.Choice(
                'Method to determine neighbors',
                D_ALL, D_EXPAND, doc="""
            There are several methods by which to determine whether objects are neighbors:
            <ul>
            <li><i>{d_adjacent}:</i> In this mode, two objects must have adjacent
            boundary pixels to be neighbors. </li>
            <li><i>{d_expand}:</i> The objects are expanded until all
            pixels on the object boundaries are touching another. Two objects are
            neighbors if any of their boundary pixels are adjacent after
            expansion.</li>
            <li><i>{d_within}:</i> Each object is expanded by
            the number of pixels you specify. Two objects are
            neighbors if they have adjacent pixels after expansion. </li>
            </ul>

            <p>For <i>{d_adjacent}</i> and <i>{d_expand}</i>, the
            <i>{m_percent_touching}</i> measurement is the percentage of pixels on the boundary
            of an object that touch adjacent objects. For <i>{d_within}</i>,
            two objects are touching if any of their boundary
            pixels are adjacent after expansion and <i>{m_percent_touching}</i> measures the
            percentage of boundary pixels of an <i>expanded</i> object that
            touch adjacent objects.</p>""".format(**{
                'd_adjacent': D_ADJACENT,
                'd_expand': D_EXPAND,
                'd_within': D_WITHIN,
                'm_percent_touching': M_PERCENT_TOUCHING
            }))

        self.distance = cellprofiler.setting.Integer(
                'Neighbor distance', 5, 1, doc="""
            <i>(Used only when "{}" is selected)</i> <br>
            The Neighbor distance is the number of pixels that each object is
            expanded for the neighbor calculation. Expanded objects that touch
            are considered neighbors.""".format(D_WITHIN))

        self.wants_count_image = cellprofiler.setting.Binary(
                'Retain the image of objects colored by numbers of neighbors?',
                False, doc="""
            An output image showing the input objects
            colored by numbers of neighbors may be retained. A colormap of your choice shows
            how many neighbors each object has. The background is set
            to -1. Objects are colored with an increasing color value
            corresponding to the number of neighbors, such that objects with no
            neighbors are given a color corresponding to 0. Use the <b>SaveImages</b>
            module to save this image to a file.""")

        self.count_image_name = cellprofiler.setting.ImageNameProvider(
                'Name the output image',
                'ObjectNeighborCount', doc="""
            <i>(Used only if the image of objects colored by numbers of neighbors
            is to be retained for later use in the pipeline)</i> <br>
            Specify a name
            that will allow the image of objects colored by numbers of neighbors
            to be selected later in the pipeline.""")

        self.count_colormap = cellprofiler.setting.Colormap(
                'Select colormap', doc="""
            <i>(Used only if the image of objects colored by numbers of neighbors
            is to be retained for later use in the pipeline)</i> <br>
            Select the colormap to use to color the neighbor number image. All available colormaps can be seen
            <a href="http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps">here</a>.""")

        self.wants_percent_touching_image = cellprofiler.setting.Binary(
                'Retain the image of objects colored by percent of touching pixels?',
                False, doc="""
            Select <i>{}</i> to keep an image of the input objects
            colored by the percentage of the boundary touching their neighbors.
            A colormap of your choice is used to show the touching percentage of
            each object. Use the <b>SaveImages</b> module to save this image to a file.""".format(cellprofiler.setting.YES))

        self.touching_image_name = cellprofiler.setting.ImageNameProvider(
                'Name the output image',
                'PercentTouching', doc="""
            <i>(Used only if the image of objects colored by percent touching
            is to be retained for later use in the pipeline)</i> <br>
            Specify a name that will allow the image of objects colored by percent of touching
            pixels to be selected later in the pipeline.""")

        self.touching_colormap = cellprofiler.setting.Colormap(
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
        """True if the neighbors are taken from the same object set as objects"""
        return self.object_name.value == self.neighbors_name.value

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.object_name.value)
        assert isinstance(objects, cellprofiler.region.Region)
        has_pixels = objects.areas > 0
        labels = objects.small_removed_segmented
        kept_labels = objects.segmented
        neighbor_objects = workspace.object_set.get_objects(
                self.neighbors_name.value)
        assert isinstance(neighbor_objects, cellprofiler.region.Region)
        neighbor_labels = neighbor_objects.small_removed_segmented
        #
        # Need to add in labels touching border.
        #
        unedited_segmented = neighbor_objects.unedited_segmented
        touching_border = numpy.zeros(numpy.max(unedited_segmented) + 1, bool)
        touching_border[unedited_segmented[0, :]] = True
        touching_border[unedited_segmented[-1, :]] = True
        touching_border[unedited_segmented[:, 0]] = True
        touching_border[unedited_segmented[:, -1]] = True
        touching_border[0] = False
        touching_border_mask = touching_border[unedited_segmented]
        nobjects = numpy.max(labels)
        nkept_objects = objects.count
        nneighbors = numpy.max(neighbor_labels)
        if numpy.any(touching_border) and \
                numpy.all(~ touching_border_mask[neighbor_labels != 0]):
            # Add the border labels if any were excluded
            touching_border_object_number = numpy.cumsum(touching_border) + \
                                            numpy.max(neighbor_labels)
            touching_border_mask &= neighbor_labels == 0
            neighbor_labels = neighbor_labels.copy().astype(numpy.int32)
            neighbor_labels[touching_border_mask] = touching_border_object_number[
                unedited_segmented[touching_border_mask]]

        _, object_numbers = objects.relate_labels(labels, kept_labels)
        if self.neighbors_are_objects:
            neighbor_numbers = object_numbers
            neighbor_has_pixels = has_pixels
        else:
            _, neighbor_numbers = neighbor_objects.relate_labels(
                    neighbor_labels, neighbor_objects.segmented)
            neighbor_has_pixels = numpy.bincount(neighbor_labels.ravel())[1:] > 0
        neighbor_count = numpy.zeros((nobjects,))
        pixel_count = numpy.zeros((nobjects,))
        first_object_number = numpy.zeros((nobjects,), int)
        second_object_number = numpy.zeros((nobjects,), int)
        first_x_vector = numpy.zeros((nobjects,))
        second_x_vector = numpy.zeros((nobjects,))
        first_y_vector = numpy.zeros((nobjects,))
        second_y_vector = numpy.zeros((nobjects,))
        angle = numpy.zeros((nobjects,))
        percent_touching = numpy.zeros((nobjects,))
        expanded_labels = None
        if self.distance_method == D_EXPAND:
            # Find the i,j coordinates of the nearest foreground point
            # to every background point
            i, j = scipy.ndimage.distance_transform_edt(labels == 0,
                                                        return_distances=False,
                                                        return_indices=True)
            # Assign each background pixel to the label of its nearest
            # foreground pixel. Assign label to label for foreground.
            labels = labels[i, j]
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
            object_indexes = numpy.arange(nobjects, dtype=numpy.int32) + 1
            #
            # First, compute the first and second nearest neighbors,
            # and the angles between self and the first and second
            # nearest neighbors
            #
            ocenters = centrosome.cpmorphology.centers_of_labels(
                    objects.small_removed_segmented).transpose()
            ncenters = centrosome.cpmorphology.centers_of_labels(
                    neighbor_objects.small_removed_segmented).transpose()
            areas = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.sum(numpy.ones(labels.shape), labels, object_indexes))
            perimeter_outlines = centrosome.outline.outline(labels)
            perimeters = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.sum(
                    numpy.ones(labels.shape), perimeter_outlines, object_indexes))

            i, j = numpy.mgrid[0:nobjects, 0:nneighbors]
            distance_matrix = numpy.sqrt((ocenters[i, 0] - ncenters[j, 0]) ** 2 +
                                         (ocenters[i, 1] - ncenters[j, 1]) ** 2)
            #
            # order[:,0] should be arange(nobjects)
            # order[:,1] should be the nearest neighbor
            # order[:,2] should be the next nearest neighbor
            #
            if distance_matrix.shape[1] == 1:
                # a little buggy, lexsort assumes that a 2-d array of
                # second dimension = 1 is a 1-d array
                order = numpy.zeros(distance_matrix.shape, int)
            else:
                order = numpy.lexsort([distance_matrix])
            first_neighbor = 1 if self.neighbors_are_objects else 0
            first_object_index = order[:, first_neighbor]
            first_x_vector = ncenters[first_object_index, 1] - ocenters[:, 1]
            first_y_vector = ncenters[first_object_index, 0] - ocenters[:, 0]
            if nneighbors > first_neighbor + 1:
                second_object_index = order[:, first_neighbor + 1]
                second_x_vector = ncenters[second_object_index, 1] - ocenters[:, 1]
                second_y_vector = ncenters[second_object_index, 0] - ocenters[:, 0]
                v1 = numpy.array((first_x_vector, first_y_vector))
                v2 = numpy.array((second_x_vector, second_y_vector))
                #
                # Project the unit vector v1 against the unit vector v2
                #
                dot = (numpy.sum(v1 * v2, 0) /
                       numpy.sqrt(numpy.sum(v1 ** 2, 0) * numpy.sum(v2 ** 2, 0)))
                angle = numpy.arccos(dot) * 180. / numpy.pi

            # Make the structuring element for dilation
            strel = centrosome.cpmorphology.strel_disk(distance)
            #
            # A little bigger one to enter into the border with a structure
            # that mimics the one used to create the outline
            #
            strel_touching = centrosome.cpmorphology.strel_disk(distance + .5)
            #
            # Get the extents for each object and calculate the patch
            # that excises the part of the image that is "distance"
            # away
            i, j = numpy.mgrid[0:labels.shape[0], 0:labels.shape[1]]
            min_i, max_i, min_i_pos, max_i_pos = \
                scipy.ndimage.extrema(i, labels, object_indexes)
            min_j, max_j, min_j_pos, max_j_pos = \
                scipy.ndimage.extrema(j, labels, object_indexes)
            min_i = numpy.maximum(centrosome.cpmorphology.fixup_scipy_ndimage_result(min_i) - distance, 0).astype(int)
            max_i = numpy.minimum(centrosome.cpmorphology.fixup_scipy_ndimage_result(max_i) + distance + 1, labels.shape[0]).astype(int)
            min_j = numpy.maximum(centrosome.cpmorphology.fixup_scipy_ndimage_result(min_j) - distance, 0).astype(int)
            max_j = numpy.minimum(centrosome.cpmorphology.fixup_scipy_ndimage_result(max_j) + distance + 1, labels.shape[1]).astype(int)
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
                patch = labels[min_i[index]:max_i[index],
                        min_j[index]:max_j[index]]
                npatch = neighbor_labels[min_i[index]:max_i[index],
                         min_j[index]:max_j[index]]
                #
                # Find the neighbors
                #
                patch_mask = patch == (index + 1)
                extended = scipy.ndimage.binary_dilation(patch_mask, strel)
                neighbors = numpy.unique(npatch[extended])
                neighbors = neighbors[neighbors != 0]
                if self.neighbors_are_objects:
                    neighbors = neighbors[neighbors != object_number]
                nc = len(neighbors)
                neighbor_count[index] = nc
                if nc > 0:
                    first_objects.append(numpy.ones(nc, int) * object_number)
                    second_objects.append(neighbors)
                if self.neighbors_are_objects:
                    #
                    # Find the # of overlapping pixels. Dilate the neighbors
                    # and see how many pixels overlap our image. Use a 3x3
                    # structuring element to expand the overlapping edge
                    # into the perimeter.
                    #
                    outline_patch = perimeter_outlines[
                                    min_i[index]:max_i[index],
                                    min_j[index]:max_j[index]] == object_number
                    extended = scipy.ndimage.binary_dilation(
                            (patch != 0) & (patch != object_number), strel_touching)
                    overlap = numpy.sum(outline_patch & extended)
                    pixel_count[index] = overlap
            if sum([len(x) for x in first_objects]) > 0:
                first_objects = numpy.hstack(first_objects)
                reverse_object_numbers = numpy.zeros(
                    max(numpy.max(object_numbers), numpy.max(first_objects)) + 1, int)
                reverse_object_numbers[object_numbers] = numpy.arange(len(object_numbers)) + 1
                first_objects = reverse_object_numbers[first_objects]

                second_objects = numpy.hstack(second_objects)
                reverse_neighbor_numbers = numpy.zeros(
                    max(numpy.max(neighbor_numbers), numpy.max(second_objects)) + 1, int)
                reverse_neighbor_numbers[neighbor_numbers] = numpy.arange(len(neighbor_numbers)) + 1
                second_objects = reverse_neighbor_numbers[second_objects]
                to_keep = (first_objects > 0) & (second_objects > 0)
                first_objects = first_objects[to_keep]
                second_objects = second_objects[to_keep]
            else:
                first_objects = numpy.zeros(0, int)
                second_objects = numpy.zeros(0, int)
            if self.neighbors_are_objects:
                percent_touching = pixel_count * 100 / perimeters
            else:
                percent_touching = pixel_count * 100.0 / areas
            object_indexes = object_numbers - 1
            neighbor_indexes = neighbor_numbers - 1
            #
            # Have to recompute nearest
            #
            first_object_number = numpy.zeros(nkept_objects, int)
            second_object_number = numpy.zeros(nkept_objects, int)
            if nkept_objects > (1 if self.neighbors_are_objects else 0):
                di = (ocenters[object_indexes[:, numpy.newaxis], 0] -
                      ncenters[neighbor_indexes[numpy.newaxis, :], 0])
                dj = (ocenters[object_indexes[:, numpy.newaxis], 1] -
                      ncenters[neighbor_indexes[numpy.newaxis, :], 1])
                distance_matrix = numpy.sqrt(di * di + dj * dj)
                distance_matrix[~ has_pixels, :] = numpy.inf
                distance_matrix[:, ~neighbor_has_pixels] = numpy.inf
                #
                # order[:,0] should be arange(nobjects)
                # order[:,1] should be the nearest neighbor
                # order[:,2] should be the next nearest neighbor
                #
                order = numpy.lexsort([distance_matrix]).astype(
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
            first_objects = numpy.zeros(0, int)
            second_objects = numpy.zeros(0, int)
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
        assert (isinstance(workspace, cellprofiler.workspace.Workspace))
        m = workspace.measurements
        assert (isinstance(m, cellprofiler.measurement.Measurements))
        image_set = workspace.image_set
        features_and_data = [
            (M_NUMBER_OF_NEIGHBORS, neighbor_count),
            (M_FIRST_CLOSEST_OBJECT_NUMBER, first_object_number),
            (M_FIRST_CLOSEST_DISTANCE, numpy.sqrt(first_x_vector ** 2 + first_y_vector ** 2)),
            (M_SECOND_CLOSEST_OBJECT_NUMBER, second_object_number),
            (M_SECOND_CLOSEST_DISTANCE, numpy.sqrt(second_x_vector ** 2 + second_y_vector ** 2)),
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
                    cellprofiler.measurement.NEIGHBORS,
                    self.object_name.value,
                    self.object_name.value if self.neighbors_are_objects
                    else self.neighbors_name.value,
                m.image_set_number * numpy.ones(first_objects.shape, int),
                    first_objects,
                m.image_set_number * numpy.ones(second_objects.shape, int),
                    second_objects)

        labels = kept_labels

        neighbor_count_image = numpy.zeros(labels.shape, int)
        object_mask = objects.segmented != 0
        object_indexes = objects.segmented[object_mask] - 1
        neighbor_count_image[object_mask] = neighbor_count[object_indexes]
        workspace.display_data.neighbor_count_image = neighbor_count_image

        if self.neighbors_are_objects:
            percent_touching_image = numpy.zeros(labels.shape)
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
            count_image = cellprofiler.image.Image(img, masking_objects=objects)
            image_set.add(self.count_image_name.value, count_image)
        else:
            neighbor_cm_name = cellprofiler.preferences.get_default_colormap()
            neighbor_cm = matplotlib.cm.get_cmap(neighbor_cm_name)
        if self.neighbors_are_objects and self.wants_percent_touching_image:
            percent_touching_cm_name = self.touching_colormap.value
            percent_touching_cm = get_colormap(percent_touching_cm_name)
            sm = matplotlib.cm.ScalarMappable(cmap=percent_touching_cm)
            img = sm.to_rgba(percent_touching_image)[:, :, :3]
            img[:, :, 0][~ object_mask] = 0
            img[:, :, 1][~ object_mask] = 0
            img[:, :, 2][~ object_mask] = 0
            touching_image = cellprofiler.image.Image(img, masking_objects=objects)
            image_set.add(self.touching_image_name.value,
                          touching_image)
        else:
            percent_touching_cm_name = cellprofiler.preferences.get_default_colormap()
            percent_touching_cm = matplotlib.cm.get_cmap(percent_touching_cm_name)

        if self.show_window:
            workspace.display_data.neighbor_cm_name = neighbor_cm_name
            workspace.display_data.percent_touching_cm_name = percent_touching_cm_name
            workspace.display_data.orig_labels = objects.segmented
            workspace.display_data.expanded_labels = expanded_labels
            workspace.display_data.object_mask = object_mask

    def display(self, workspace, figure):
        figure.set_subplots((2, 2))
        figure.subplot_imshow_labels(0, 0, workspace.display_data.orig_labels,
                                     "Original: %s" % self.object_name.value)

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
        if numpy.any(object_mask):
            figure.subplot_imshow(0, 1, neighbor_count_image,
                                  "%s colored by # of neighbors" %
                                  self.object_name.value,
                                  colormap=neighbor_cm,
                                  colorbar=True, vmin=0,
                                  vmax=max(neighbor_count_image.max(), 1),
                                  normalize=False,
                                  sharexy=figure.subplot(0, 0))
            if self.neighbors_are_objects:
                figure.subplot_imshow(1, 1, percent_touching_image,
                                      "%s colored by pct touching" %
                                      self.object_name.value,
                                      colormap=percent_touching_cm,
                                      colorbar=True, vmin=0,
                                      vmax=max(percent_touching_image.max(), 1),
                                      normalize=False,
                                      sharexy=figure.subplot(0, 0))
        else:
            # No objects - colorbar blows up.
            figure.subplot_imshow(0, 1, neighbor_count_image,
                                  "%s colored by # of neighbors" %
                                  self.object_name.value,
                                  colormap=neighbor_cm,
                                  vmin=0,
                                  vmax=max(neighbor_count_image.max(), 1),
                                  sharexy=figure.subplot(0, 0))
            if self.neighbors_are_objects:
                figure.subplot_imshow(1, 1, percent_touching_image,
                                      "%s colored by pct touching" %
                                      self.object_name.value,
                                      colormap=percent_touching_cm,
                                      vmin=0,
                                      vmax=max(neighbor_count_image.max(), 1),
                                      sharexy=figure.subplot(0, 0))

        if self.distance_method == D_EXPAND:
            figure.subplot_imshow_labels(1, 0, expanded_labels,
                                         "Expanded %s" %
                                         self.object_name.value,
                                         sharexy=figure.subplot(0, 0))

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
        """Return column definitions for measurements made by this module"""
        coltypes = dict([(feature,
                          cellprofiler.measurement.COLTYPE_INTEGER
                          if feature in (M_NUMBER_OF_NEIGHBORS,
                                         M_FIRST_CLOSEST_OBJECT_NUMBER,
                                         M_SECOND_CLOSEST_OBJECT_NUMBER)
                          else cellprofiler.measurement.COLTYPE_FLOAT)
                         for feature in self.all_features])
        return [(self.object_name.value,
                 self.get_measurement_name(feature_name),
                 coltypes[feature_name])
                for feature_name in self.all_features]

    def get_object_relationships(self, pipeline):
        """Return column definitions for object relationships output by module"""
        objects_name = self.object_name.value
        if self.neighbors_are_objects:
            neighbors_name = objects_name
        else:
            neighbors_name = self.neighbors_name.value
        return [(cellprofiler.measurement.NEIGHBORS, objects_name, neighbors_name,
                 cellprofiler.measurement.MCA_AVAILABLE_EACH_CYCLE)]

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
            wants_image = setting_values[2] != cellprofiler.setting.DO_NOT_USE
            distance_method = D_EXPAND if setting_values[1] == "0" else D_WITHIN
            setting_values = [setting_values[0],
                              distance_method,
                              setting_values[1],
                              cellprofiler.setting.YES if wants_image else cellprofiler.setting.NO,
                              setting_values[2],
                              cellprofiler.setting.DEFAULT,
                              cellprofiler.setting.NO,
                              "PercentTouching",
                              cellprofiler.setting.DEFAULT]
            from_matlab = False
            variable_revision_number = 1
        if variable_revision_number == 1:
            # Added neighbor objects
            # To upgrade, repeat object_name twice
            #
            setting_values = setting_values[:1] * 2 + setting_values[1:]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab


def get_colormap(name):
    """Get colormap, accounting for possible request for default"""
    if name == cellprofiler.setting.DEFAULT:
        name = cellprofiler.preferences.get_default_colormap()
    return matplotlib.cm.get_cmap(name)
