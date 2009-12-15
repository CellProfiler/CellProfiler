'''<b>RelabelObjects</b>
Relabels objects so that objects within a specified distance of each
other, or objects with a straight line connecting
their centroids that has a relatively uniform intensity, 
get the same label and thereby become the same object.
Optionally, if an object consists of two or more unconnected components, this
module can relabel them so that the components become separate objects.
<hr>
Relabeling objects changes the labels of the pixels in an object such
that it either becomes equal to the label of another (unify) or changes
the labels to distinguish two different components of an object such that
they are two different objects (Split).

If the distance threshold is zero (the default), only
objects that are touching will be unified. Note that selecting "unify" will not connect or bridge
the two objects by adding any new pixels. The new, unified object
may consist of two or more unconnected components.
 
As an experimental feature, it is possible to specify a grayscale
image to help guide the decision of which objects to unify.  When
the module considers merging two objects, it looks at the pixels
along the line connecting their centroids in this image.  If the
intensity of any of these pixels is below 90 percent of either
centroid, the objects are not unified.

In order to ensure that objects are labeled consecutively (which
other modules depend on), RelabelObjects may change the label (i.e.,
the object number) of any object.  A new "measurement" will be added
for each input object.  This "measurement" is a number that
indicates the relabeled object number.
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
import scipy.ndimage as scind
from scipy.sparse import coo_matrix

import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs
from cellprofiler.modules.identify import get_object_measurement_columns
from cellprofiler.modules.identify import add_object_count_measurements
from cellprofiler.modules.identify import add_object_location_measurements
import cellprofiler.cpmath.cpmorphology as morph
from cellprofiler.cpmath.filter import stretch

OPTION_UNIFY = "Unify"
OPTION_SPLIT = "Split"

CA_CENTROIDS = "Centroids"
CA_CLOSEST_POINT = "Closest point"

class RelabelObjects(cpm.CPModule):
    module_name = "RelabelObjects"
    category = "Object Processing"
    variable_revision_number = 1
    
    def create_settings(self):
        self.objects_name = cps.ObjectNameSubscriber(
            "Enter original objects name:",
            "None",
            doc="""This setting names the objects that will be relabeled.
            You can use any objects that were created in previous modules
            (for instance, in <b>IdentifyPrimAutomatic</b> or
            <b>IdentifySecondary</b>""")
        self.output_objects_name = cps.ObjectNameProvider(
            "Enter new objects name:","RelabeledNuclei",
            doc="""This setting names the objects that are the result of
            the "relate" operation. You can use this name in subsequent
            modules that take objects as inputs.""")
        self.relabel_option = cps.Choice(
            "Unify or split objects?",[OPTION_UNIFY, OPTION_SPLIT],
            doc="""Choose "Unify" to turn nearby or touching objects into
            a single object. Choose "Split" to break non-adjacent pieces of
            objects into separate objects.""")
        self.distance_threshold = cps.Integer(
            "What is the maximum distance between two objects that should be unified?",
            0,minval=0)
        self.wants_image = cps.Binary(
            "Do you want to use an image during unification?", False,
            doc="""Unify can use image information to determine whether two
            objects should be unified. Unify will unify two objects if they
            are within the minimum distance and all points along the line
            connecting the centroids are at least 90% of the intensity
            at the centroids""")
        self.image_name = cps.ImageNameSubscriber(
            "Grayscale image name:", "None",
            doc="""This is the name of an image from a previous module. The
            image is used during unification to determine if the intensities
            between objects are within 90% of that at the centroids""")
        
        self.minimum_intensity_fraction = cps.Float(
            "Minimum intensity fraction", .9, minval=0, maxval=1,
            doc="""The grayscale algorithm finds the points along the line
            connecting two objects' centroids. Two objects can only be connected
            if the points along this line are all greater than a fraction
            of the intensity of the dimmest centroid. This setting determines
            the minimum acceptable fraction. For instance, if the intensity
            of one centroid was .75 and the other was .50 and this setting
            was .9, all points along the line would need to have an intensity
            of min(.75, .50) * .9 = .50 * .9 = .45""")
        
        self.where_algorithm = cps.Choice(
            "How do you want to find the object intensity?",
            [CA_CLOSEST_POINT, CA_CENTROIDS],
            doc = """
            You can use one of two algorithms to determine whether two
            objects are touching. The "Centroids" method draws a line
            between the centroids of two cells and each point on that line.
            The algorithm records the lower of the two centroid intensities.
            Each point along the line must be at least a fraction of this
            intensity. This is good for round cells whose maximum intensity
            is in the center of the cell.
            
            The "Closest point" method finds the closest point to each point
            in the background. Two objects touch<br>
            <ul><li>if each has a nearby background pixel that is
            at most 1/2 of the maximum distance from the object</li>
            <li>if the background pixel's intensity is at least the
            minimum intensity fraction of its nearest pixel in the cell</li>
            <li>if one of these background pixels in one object is adjacent
            to one of these background pixels in the other object</li></ul><br>
            The "Closest point" method is best to use for irregularly-shaped
            objects that are connected by pixels of roughly the same intensity
            as that of the edge of the objects to be connected.""")

        
    def settings(self):
        return [self.objects_name, self.output_objects_name,
                self.relabel_option, self.distance_threshold, 
                self.wants_image, self.image_name, 
                self.minimum_intensity_fraction,
                self.where_algorithm]
    
    def visible_settings(self):
        result = [self.objects_name, self.output_objects_name,
                  self.relabel_option]
        if self.relabel_option == OPTION_UNIFY:
            result += [self.distance_threshold, self.wants_image]
            if self.wants_image:
                result += [self.image_name, self.minimum_intensity_fraction,
                           self.where_algorithm]
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
        add_object_count_measurements(workspace.measurements,
                                      self.output_objects_name.value,
                                      np.max(output_objects.segmented))
        add_object_location_measurements(workspace.measurements,
                                         self.output_objects_name.value,
                                         output_objects.segmented)
        if workspace.frame is not None:
            workspace.display_data.orig_labels = objects.segmented
            workspace.display_data.output_labels = output_objects.segmented
    
    def display(self, workspace):
        '''Display the results of relabeling
        
        workspace - workspace containing saved display data
        '''
        from cellprofiler.gui.cpfigure import renumber_labels_for_display
        import matplotlib.cm as cm
        
        figure = workspace.create_or_find_figure(subplots=(1,2))
        figure.subplot_imshow_labels(0,0, workspace.display_data.orig_labels,
                                     title = self.objects_name.value)
        if self.wants_image:
            #
            # Make a nice picture which superimposes the labels on the
            # guiding image
            #
            output_labels = renumber_labels_for_display(
                workspace.display_data.output_labels)
            image = (stretch(workspace.display_data.image) * 255).astype(np.uint8)
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
                                  title = self.output_objects_name.value)
        else:
            figure.subplot_imshow_labels(0,1, 
                                         workspace.display_data.output_labels,
                                         title = self.output_objects_name.value)

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
        if from_matlab and variable_revision_number == 1:
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
                       
        return setting_values, variable_revision_number, from_matlab
    
    def get_image(self, workspace):
        '''Get the image for image-directed merging'''
        objects = workspace.object_set.get_objects(self.objects_name.value)
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_grayscale=True)
        image = objects.crop_image_similarly(image.pixel_data)
        return image
        
    def get_measurement_columns(self, pipeline):
        return get_object_measurement_columns(self.output_objects_name.value)
    
    def get_categories(self,pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        if object_name == 'Image':
            return ['Count']
        elif object_name == self.output_objects_name.value:
            return ['Location']
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
        return []

def copy_labels(labels, segmented):
    '''Carry differences between orig_segmented and new_segmented into "labels"
    
    labels - labels matrix similarly segmented to "segmented"
    segmented - the newly numbered labels matrix (a subset of pixels are labeled)
    '''
    max_labels = np.max(labels)
    labels_new = segmented+max_labels
    labels_new[segmented==0] = labels[segmented==0]
    unique_labels = np.unique(labels_new)
    new_indexes = np.zeros(np.max(unique_labels)+1,int)
    new_indexes[unique_labels] = np.arange(len(unique_labels))+1
    labels_new = new_indexes[labels_new]
    return labels_new
