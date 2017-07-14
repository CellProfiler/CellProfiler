# coding=utf-8

'''
<b>Reassign Object Numbers</b> renumbers previously identified objects.
<hr>
Objects and their measurements are associated
with each other based on their object numbers (also known as <i>labels</i>). Typically,
each object is assigned a single unique number, such that the exported measurements are ordered
by this numbering.  This module
allows the reassignment of object numbers by either unifying separate objects to share
the same label, or splitting portions of separate objects that previously had the same label.

<h4>Available measurements</h4>
<b>Parent object measurements:</b>
<ul>
<li><i>Children Count:</i> The number of relabeled objects created from each parent object.</li>
</ul>

<b>Reassigned object measurements:</b>
<ul>
<li><i>Parent:</i>The label number of the parent object.</li>
<li><i>Location_X, Location_Y:</i> The pixel (X,Y) coordinates of the center of
mass of the reassigned objects.</li>
</ul>

<h4>Technical notes</h4>
Reassignment means that the numerical value of every pixel within
an object (in the label matrix version of the image) gets changed, as specified by
the module settings. In order to ensure that objects are labeled consecutively
without gaps in the numbering (which other modules may depend on),
<b>ReassignObjectNumbers</b> will typically result in most of the objects having
their numbers reordered. This reassignment information is stored as a per-object measurement
with both the original input and reasigned output objects, in case you need to track the
reassignment.

<p>See also <b>RelateObjects</b>.</p>
'''

import centrosome.cpmorphology as morph
import centrosome.outline
import numpy as np
import scipy.ndimage as scind
from centrosome.filter import stretch
from scipy.sparse import coo_matrix

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.setting as cps
from cellprofiler.gui.help import RETAINING_OUTLINES_HELP, NAMING_OUTLINES_HELP
from cellprofiler.measurement import C_PARENT, C_CHILDREN, FF_CHILDREN_COUNT, FF_PARENT
from cellprofiler.modules.identify import add_object_count_measurements
from cellprofiler.modules.identify import add_object_location_measurements
from cellprofiler.modules.identify import get_object_measurement_columns
from cellprofiler.setting import YES, NO

OPTION_UNIFY = "Unify"
OPTION_SPLIT = "Split"

UNIFY_DISTANCE = "Distance"
UNIFY_PARENT = "Per-parent"

CA_CENTROIDS = "Centroids"
CA_CLOSEST_POINT = "Closest point"

UM_DISCONNECTED = "Disconnected"
UM_CONVEX_HULL = "Convex hull"


class ReassignObjectNumbers(cpm.Module):
    module_name = "ReassignObjectNumbers"
    category = "Object Processing"
    variable_revision_number = 4

    def create_settings(self):
        self.objects_name = cps.ObjectNameSubscriber(
                "Select the input objects",
                cps.NONE, doc="""
            Select the objects whose object numbers you want to reassign.
            You can use any objects that were created in previous modules, such as
            <b>IdentifyPrimaryObjects</b> or <b>IdentifySecondaryObjects</b>.""")

        self.output_objects_name = cps.ObjectNameProvider(
                "Name the new objects", "RelabeledNuclei", doc="""
            Enter a name for the objects whose numbers have been reassigned.
            You can use this name in subsequent modules that take objects as inputs.""")

        self.relabel_option = cps.Choice(
                "Operation", [OPTION_UNIFY, OPTION_SPLIT], doc="""
            You can choose one of the following options:
            <ul>
            <li><i>%(OPTION_UNIFY)s:</i> Assign adjacent or nearby objects the same
            label based on certain criteria. It can be useful, for example,
            to merge together touching objects that were incorrectly split into two pieces
            by an <b>Identify</b> module.</li>
            <li><i>%(OPTION_SPLIT)s:</i> Assign a unique number to separate objects
            that currently share the same label. This can occur if you applied certain
            operations with the <b>Morph</b> module to objects.</li>
            </ul>""" % globals())

        self.unify_option = cps.Choice(
                "Unification method", [UNIFY_DISTANCE, UNIFY_PARENT], doc="""
            <i>(Used only with the %(OPTION_UNIFY)s option)</i><br>
            You can unify objects in one of two ways:
            <ul>
            <li><i>%(UNIFY_DISTANCE)s: </i> All objects within a certain pixel radius
            from each other will be unified</li>
            <li><i>%(UNIFY_PARENT)s: </i>All objects which share the same parent
            relationship to another object will be unified. This is not be confused
            with using the <b>RelateObjects</b> module, in which the related objects
            remain as individual objects. See <b>RelateObjects</b> for more details.</li>
            </ul>
            """ % globals())

        self.unification_method = cps.Choice(
                "Output object type", [UM_DISCONNECTED, UM_CONVEX_HULL],
                doc="""
            <i>(Used only with the %(UNIFY_PARENT)s unification method)</i>
            <br>
            <b>ReassignObjectNumbers</b> can either unify the child objects
            and keep them disconnected or it can find the smallest convex
            polygon (the convex hull) that encloses all of a parent's child
            objects. The convex hull will be truncated to include only those
            pixels in the parent - in that case it may not truly be convex.
            Choose %(UM_DISCONNECTED)s to leave the children as
            disconnected pieces. Choose %(UM_CONVEX_HULL)s to create an
            output object that is the convex hull around them all.
            """ % globals()
        )

        self.parent_object = cps.Choice(
                "Select the parent object", [cps.NONE],
                choices_fn=self.get_parent_choices, doc="""
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
                0, minval=0, doc="""
            <i>(Used only with the %(OPTION_UNIFY)s option and the %(UNIFY_DISTANCE)s method)</i><br>
            Objects that are less than or equal to the distance
            you enter here, in pixels, will be unified. If you choose zero
            (the default), only objects that are touching will be unified.
            Note that <i>%(OPTION_UNIFY)s </i> will not actually connect or bridge
            the two objects by adding any new pixels; it simply assigns the same object number
            to the portions of the object. The new, unified object
            may therefore consist of two or more unconnected components.""" % globals())

        self.wants_image = cps.Binary(
                "Unify using a grayscale image?", False, doc="""
            <i>(Used only with the %(OPTION_UNIFY)s option)</i><br>
            Select <i>%(YES)s</i> to use the objects' intensity features to determine whether two
            objects should be unified. If you choose to use a grayscale image,
            <i>%(OPTION_UNIFY)s</i> will unify two objects only if they
            are within the distance you have specified <i>and</i> certain criteria about the objects
            within the grayscale image are met.""" % globals())

        self.image_name = cps.ImageNameSubscriber(
                "Select the grayscale image to guide unification", cps.NONE, doc="""
            <i>(Used only if a grayscale image is to be used as a guide for unification)</i><br>
            Select the name of an image loaded or created by a previous module.""")

        self.minimum_intensity_fraction = cps.Float(
                "Minimum intensity fraction", .9, minval=0, maxval=1, doc="""
            <i>(Used only if a grayscale image is to be used as a guide for unification)</i><br>
            Select the minimum acceptable intensity fraction. This will be used
            as described for the method you choose in the next setting.""")

        self.where_algorithm = cps.Choice(
                "Method to find object intensity",
                [CA_CLOSEST_POINT, CA_CENTROIDS], doc="""
            <i>(Used only if a grayscale image is to be used as a guide for unification)</i><br>
            You can use one of two methods to determine whether two
            objects should unified, assuming they meet the distance criteria (as specified above):
            <ul>
            <li><i>%(CA_CENTROIDS)s:</i> When the module considers merging two objects,
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

            <li><i>%(CA_CLOSEST_POINT)s:</i> This method is useful for unifying irregularly shaped cells
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
            </ul>""" % globals())

        self.wants_outlines = cps.Binary(
                "Retain outlines of the relabeled objects?", False, doc="""
            %(RETAINING_OUTLINES_HELP)s""" % globals())

        self.outlines_name = cps.OutlineNameProvider(
                'Name the outlines',
                'RelabeledNucleiOutlines', doc="""
            %(NAMING_OUTLINES_HELP)s""" % globals())

    def get_parent_choices(self, pipeline):
        columns = pipeline.get_measurement_columns()
        choices = [cps.NONE]
        for column in columns:
            object_name, feature, coltype = column[:3]
            if (object_name == self.objects_name.value and
                    feature.startswith(C_PARENT)):
                choices.append(feature[(len(C_PARENT) + 1):])
        return choices

    def validate_module(self, pipeline):
        if self.relabel_option == OPTION_UNIFY and self.unify_option == UNIFY_PARENT and self.parent_object.value == cps.NONE:
            raise cps.ValidationError(
                    '%s is not a valid object name' % cps.NONE,
                    self.parent_object)

    def settings(self):
        return [self.objects_name, self.output_objects_name,
                self.relabel_option, self.distance_threshold,
                self.wants_image, self.image_name,
                self.minimum_intensity_fraction,
                self.where_algorithm,
                self.wants_outlines, self.outlines_name,
                self.unify_option, self.parent_object,
                self.unification_method]

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
                result += [self.unification_method, self.parent_object]
        result += [self.wants_outlines]
        if self.wants_outlines:
            result += [self.outlines_name]
        return result

    def run(self, workspace):
        objects_name = self.objects_name.value
        objects = workspace.object_set.get_objects(objects_name)
        assert isinstance(objects, cpo.Objects)
        labels = objects.segmented
        if self.relabel_option == OPTION_SPLIT:
            output_labels, count = scind.label(labels > 0, np.ones((3, 3), bool))
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
                    mask = d < self.distance_threshold.value / 2 + 1
                output_labels, count = scind.label(mask, np.ones((3, 3), bool))
                output_labels[labels == 0] = 0
                if self.wants_image:
                    output_labels = self.filter_using_image(workspace, mask)
            elif self.unify_option == UNIFY_PARENT:
                parents_name = self.parent_object.value
                parents_of = workspace.measurements[
                    objects_name, "_".join((C_PARENT, parents_name))]
                output_labels = labels.copy().astype(np.uint32)
                output_labels[labels > 0] = parents_of[labels[labels > 0] - 1]
                if self.unification_method == UM_CONVEX_HULL:
                    ch_pts, n_pts = morph.convex_hull(output_labels)
                    ijv = morph.fill_convex_hulls(ch_pts, n_pts)
                    output_labels[ijv[:, 0], ijv[:, 1]] = ijv[:, 2]

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
                                     FF_PARENT % self.objects_name.value,
                                     parents_of_children)
        if self.wants_outlines:
            outlines = centrosome.outline.outline(output_labels)
            outline_image = cpi.Image(outlines.astype(bool))
            workspace.image_set.add(self.outlines_name.value,
                                    outline_image)

        if self.show_window:
            workspace.display_data.orig_labels = objects.segmented
            workspace.display_data.output_labels = output_objects.segmented
            if self.unify_option == UNIFY_PARENT:
                workspace.display_data.parent_labels = \
                    workspace.object_set.get_objects(self.parent_object.value).segmented

    def display(self, workspace, figure):
        '''Display the results of relabeling

        workspace - workspace containing saved display data
        '''
        from cellprofiler.gui.tools import renumber_labels_for_display
        import matplotlib.cm as cm

        figure.set_subplots((2, 1))
        ax = figure.subplot_imshow_labels(
                0, 0, workspace.display_data.orig_labels,
                title=self.objects_name.value)

        if self.relabel_option == OPTION_UNIFY and (
                    (self.unify_option == UNIFY_DISTANCE and self.wants_image) or (self.unify_option == UNIFY_PARENT)):
            if self.unify_option == UNIFY_DISTANCE and self.wants_image:
                image = workspace.display_data.image
                cplabels = [
                    dict(name=self.output_objects_name.value,
                         labels=[workspace.display_data.output_labels]),
                    dict(name=self.objects_name.value,
                         labels=[workspace.display_data.orig_labels])]

            elif self.unify_option == UNIFY_PARENT:
                image = np.zeros(workspace.display_data.output_labels.shape)
                cplabels = [
                    dict(name=self.output_objects_name.value,
                         labels=[workspace.display_data.output_labels]),
                    dict(name=self.parent_object.value,
                         labels=[workspace.display_data.parent_labels]),
                    dict(name=self.objects_name.value,
                         labels=[workspace.display_data.orig_labels],
                         mode="none")
                ]
            if image.ndim == 2:
                figure.subplot_imshow_grayscale(
                        1, 0, image, title=self.output_objects_name.value,
                        cplabels=cplabels, sharexy=ax)
            else:
                figure.subplot_imshow_color(
                        1, 0, image, title=self.output_objects_name.value,
                        cplabels=cplabels, sharexy=ax)
        else:
            figure.subplot_imshow_labels(1, 0,
                                         workspace.display_data.output_labels,
                                         title=self.output_objects_name.value,
                                         sharexy=ax)

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
        if self.show_window:
            # Save the image for display
            workspace.display_data.image = image
        #
        # Do a distance transform into the background to label points
        # in the background with their closest foreground object
        #
        i, j = scind.distance_transform_edt(labels == 0,
                                            return_indices=True,
                                            return_distances=False)
        confluent_labels = labels[i, j]
        confluent_labels[~mask] = 0
        if self.where_algorithm == CA_CLOSEST_POINT:
            #
            # For the closest point method, find the intensity at
            # the closest point in the object (which will be the point itself
            # for points in the object).
            #
            object_intensity = image[i, j] * self.minimum_intensity_fraction.value
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
        label_numbers = np.arange(1, len(count) + 1)[count > 0]
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
                    center_i[c_i - 1], center_j[c_i - 1],
                    center_i[c_j - 1], center_j[c_j - 1])
            #
            # The indexes of the centroids at pt1
            #
            last_indexes = indexes + counts - 1
            #
            # The minimum of the intensities at pt0 and pt1
            #
            centroid_intensities = np.minimum(
                    image[i[indexes], j[indexes]],
                    image[i[last_indexes], j[last_indexes]])
            #
            # Assign label numbers to each point so we can use
            # scipy.ndimage.minimum. The label numbers are indexes into
            # "connections" above.
            #
            pt_labels = np.zeros(len(i), int)
            pt_labels[indexes[1:]] = 1
            pt_labels = np.cumsum(pt_labels)
            minima = scind.minimum(image[i, j], pt_labels, np.arange(len(indexes)))
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

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
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
            setting_values = setting_values + [OPTION_SPLIT, '0', cps.DO_NOT_USE]
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

        if (not from_matlab) and variable_revision_number == 1:
            # Added per-parent unification
            setting_values += [UNIFY_DISTANCE, cps.NONE]
            variable_revision_number = 3

        if (not from_matlab) and variable_revision_number == 3:
            setting_values = setting_values + [UM_DISCONNECTED]
            variable_revision_number = 4

        return setting_values, variable_revision_number, from_matlab

    def get_image(self, workspace):
        '''Get the image for image-directed merging'''
        objects = workspace.object_set.get_objects(self.objects_name.value)
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_grayscale=True)
        image = objects.crop_image_similarly(image.pixel_data)
        return image

    def get_measurement_columns(self, pipeline):
        columns = get_object_measurement_columns(self.output_objects_name.value)
        columns += [(self.output_objects_name.value,
                     FF_PARENT % self.objects_name.value,
                     cpmeas.COLTYPE_INTEGER),
                    (self.objects_name.value,
                     FF_CHILDREN_COUNT % self.output_objects_name.value,
                     cpmeas.COLTYPE_INTEGER)]
        return columns

    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        if object_name == 'Image':
            return ['Count']
        elif object_name == self.output_objects_name.value:
            return ['Location', 'Parent', 'Number']
        elif object_name == self.objects_name.value:
            return ['Children']
        return []

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        if object_name == 'Image' and category == 'Count':
            return [self.output_objects_name.value]
        elif object_name == self.output_objects_name.value and category == 'Location':
            return ['Center_X', 'Center_Y']
        elif object_name == self.output_objects_name.value and category == 'Parent':
            return [self.objects_name.value]
        elif object_name == self.output_objects_name.value and category == 'Number':
            return ['Object_Number']
        elif object_name == self.objects_name.value and category == 'Children':
            return ["%s_Count" % self.output_objects_name.value]
        return []


def copy_labels(labels, segmented):
    '''Carry differences between orig_segmented and new_segmented into "labels"

    labels - labels matrix similarly segmented to "segmented"
    segmented - the newly numbered labels matrix (a subset of pixels are labeled)
    '''
    max_labels = np.max(segmented)
    seglabel = scind.minimum(labels, segmented, np.arange(1, max_labels + 1))
    labels_new = labels.copy()
    labels_new[segmented != 0] = seglabel[segmented[segmented != 0] - 1]
    return labels_new


RelabelObjects = ReassignObjectNumbers
