'''measureobjectneighbors.py - MeasureObjectNeighbors module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version = "$Revision$"

import numpy as np
import scipy.ndimage as scind
import matplotlib.cm

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.cpmath.cpmorphology import strel_disk

D_ADJACENT = 'Adjacent'
D_EXPAND   = 'Expand until adjacent'
D_WITHIN   = 'Within a specified distance'
D_ALL = [D_ADJACENT, D_EXPAND, D_WITHIN]

M_NUMBER_OF_NEIGHBORS = 'NumberOfNeighbors'
M_PERCENT_TOUCHING = 'PercentTouching'
M_FIRST_CLOSEST_OBJECT_NUMBER = 'FirstClosestObjectNumber'
M_FIRST_CLOSEST_X_VECTOR = 'FirstClosestXVector'
M_FIRST_CLOSEST_Y_VECTOR = 'FirstClosestYVector'
M_SECOND_CLOSEST_OBJECT_NUMBER = 'SecondClosestObjectNumber'
M_SECOND_CLOSEST_X_VECTOR ='SecondClosestXVector'
M_SECOND_CLOSEST_Y_VECTOR ='SecondClosestYVector'
M_ANGLE_BETWEEN_NEIGHBORS = 'AngleBetweenNeighbors'
M_ALL = [M_NUMBER_OF_NEIGHBORS, M_PERCENT_TOUCHING, 
         M_FIRST_CLOSEST_OBJECT_NUMBER, M_FIRST_CLOSEST_X_VECTOR,
         M_FIRST_CLOSEST_Y_VECTOR, M_SECOND_CLOSEST_OBJECT_NUMBER,
         M_SECOND_CLOSEST_X_VECTOR, M_SECOND_CLOSEST_Y_VECTOR,
         M_ANGLE_BETWEEN_NEIGHBORS]

C_NEIGHBORS = 'Neighbors'

S_EXPANDED = 'Expanded'
S_ADJACENT = 'Adjacent'

class MeasureObjectNeighbors(cpm.CPModule):
    '''SHORT DESCRIPTION:
    Calculates how many neighbors each object has.
    *************************************************************************
    
    Given an image with objects identified (e.g. nuclei or cells), this
    module determines how many neighbors each object has. The user selects
    the distance within which objects should be considered neighbors. The
    module can measure the number of neighbors each object has if every
    object were expanded up until the point where it hits another object; to
    use this option, enter 0 (the number zero) for the pixel distance. If you
    want your objects to be touching before you count neighbors (for 
    instance, in an image of tissue), use the ExpandOrShrink module to expand
    your objects beforehand.
    
    Features measured (per object):
    NumberOfNeighbors         # of neighbor objects
    PercentTouching           percent of pixels within the measured distance
    FirstClosestObjectNumber  The index of the closest object
    FirstClosestXVector       The difference between these objects' X locations
    FirstClosestYVector       The difference between these objects' Y locations
    SecondClosestObjectNumber The index of the second closest object
    SecondClosestXVector      The difference between these objects' X locations
    SecondClosestYVector      The difference between these objects' Y locations
    AngleBetweenNeighbors     The angle formed with the object center as the
                              vertex and the first and second closest object
                              centers along the vectors. 
    
    There are three modes of object expansion:
    * Objects are not expanded (Adjacent) - in this mode, two objects must
      have adjacent pixels to be touching. PercentTouching measures the percent
      of adjacent pixels.
    * Objects are expanded until adjacent - in this mode, pixels in the
      background are assigned to the nearest object, expanding each object
      until all border pixels are touching some other object. Two objects
      are touching if they have adjacent pixels. PercentTouching measures
      the percent of adjacent pixels.
    * Objects are expanded a manually-entered number of pixels - in this mode,
      each object is expanded in-turn by the number of pixels entered by the
      user. Two objects are touching if pixels of one object are within
      the specified distance of pixels in the other object. PercentTouching
      measures the percent of pixels in an object that overlap nearby objects
      when those objects are expanded.
    How it works: Retrieves objects in label matrix format. The objects
    are expanded by the number of pixels the user specifies, and then
    the module counts up how many other objects the object is
    overlapping.  PercentTouching, if computed, is defined as the number
    of boundary pixels on an object not obscured when other objects are
    dilated by the Neighbor distance limit (or 2 pixels if this distance
    is set to 0 for the maximum expansion option detailed above).
    
    Interpreting the module output:
    In the color image output of the module, there is a color spectrum used
    to determine which objects have neighbors, and how many. According to the
    indices on the spectrum, the background is -1, objects with no neighbors
    are 0, and objects with neighbors are greater than 0, with the increasing
    index corresponding to more neighbors.
    
    You can save the image of objects colored by numbers of neighbors.
    CellProfiler creates a color image using the color map you choose. Use
    the SaveImages module to save the image to a file.
    
    You can also save the image of objects colored by the percent of pixels
    that are touching other objects.
    '''

    category = "Measurement"
    variable_revision_number = 1

    def create_settings(self):
        self.module_name = 'MeasureObjectNeighbors'
        self.object_name = cps.ObjectNameSubscriber('What did you call the objects whose neighbors you want to measure?','None')
        self.distance_method = cps.Choice('How do you want to determine whether objects are touching?',
                                          D_ALL, D_EXPAND)
        self.distance = cps.Integer('Within what distance (in pixels) are objects to be considered neighbors?',
                                    5,1)
        self.wants_count_image = cps.Binary('Do you want to save the image of objects colored by numbers of neighbors?',
                                            False)
        self.count_image_name = cps.ImageNameProvider('What do you want to call the image of objects colored by numbers of neighbors?',
                                                      'ObjectNeighborCount')
        self.count_colormap = cps.Colormap('What colormap do you want to use to color the above image?')
        self.wants_percent_touching_image = cps.Binary('Do you want to save the image of objects colored by percent of touching pixels?',
                                                       False)
        self.touching_image_name = cps.ImageNameProvider('What do you want to call the image of objects colored by percent of touching pixels?',
                                                         'PercentTouching')
        self.touching_colormap = cps.Colormap('What colormap do you want to use to color the above image?')

    def settings(self):
        return [self.object_name, self.distance_method, self.distance,
                self.wants_count_image, self.count_image_name,
                self.count_colormap, self.wants_percent_touching_image,
                self.touching_image_name, self.touching_colormap]

    def backwards_compatibilize(self, setting_values, variable_revision_number, module_name, from_matlab):
        if from_matlab and variable_revision_number == 5:
            wants_image = setting_values[2] != cps.DO_NOT_USE
            distance_method =  D_EXPAND if setting_values[1] == "0" else D_WITHIN
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
        return setting_values, variable_revision_number, from_matlab
    
    def visible_settings(self):
        result = [self.object_name, self.distance_method]
        if self.distance_method == D_WITHIN:
            result += [self.distance]
        result += [self.wants_count_image]
        if self.wants_count_image.value:
            result += [self.count_image_name, self.count_colormap]
        result += [self.wants_percent_touching_image]
        if self.wants_percent_touching_image.value:
            result += [self.touching_image_name, self.touching_colormap]
        return result

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.object_name.value)
        labels = objects.segmented
        nobjects = np.max(labels)
        neighbor_count = np.zeros((nobjects,))
        pixel_count = np.zeros((nobjects,))
        first_object_number = np.zeros((nobjects,),int)
        second_object_number = np.zeros((nobjects,),int)
        first_x_vector = np.zeros((nobjects,))
        second_x_vector = np.zeros((nobjects,))
        first_y_vector = np.zeros((nobjects,))
        second_y_vector = np.zeros((nobjects,))
        angle = np.zeros((nobjects,))
        if self.distance_method == D_EXPAND:
            # Find the i,j coordinates of the nearest foreground point
            # to every background point
            i,j = scind.distance_transform_edt(labels==0,
                                               return_distances=False,
                                               return_indices=True)
            # Assign each background pixel to the label of its nearest
            # foreground pixel. Assign label to label for foreground.
            labels = labels[i,j]
            distance = 1 # dilate once to make touching edges overlap
            scale = S_EXPANDED
        elif self.distance_method == D_WITHIN:
            distance = self.distance.value
            scale = str(distance)
        elif self.distance_method == D_ADJACENT:
            distance = 1
            scale = S_ADJACENT
        else:
            raise ValueError("Unknown distance method: %s" %
                             self.distance_method.value)
        if nobjects > 1:
            object_indexes = np.arange(nobjects)+1
            #
            # First, compute the first and second nearest neighbors,
            # and the angles between self and the first and second
            # nearest neighbors
            #
            centers = scind.center_of_mass(np.ones(labels.shape), 
                                           objects.segmented, 
                                           object_indexes)
            if nobjects == 1:
                centers = np.array([centers])
            else:
                centers = np.array(centers)
            areas = fix(scind.sum(np.ones(labels.shape),labels, object_indexes))
            i,j = np.mgrid[0:nobjects,0:nobjects]
            distance_matrix = np.sqrt((centers[i,0]-centers[j,0])**2 +
                                      (centers[i,1]-centers[j,1])**2)
            #
            # order[:,0] should be arange(nobjects)
            # order[:,1] should be the nearest neighbor
            # order[:,2] should be the next nearest neighbor
            #
            order = np.lexsort([distance_matrix])
            first_object_index = order[:,1]
            first_object_number = first_object_index+1
            first_x_vector = centers[first_object_index,1] - centers[:,1]
            first_y_vector = centers[first_object_index,0] - centers[:,0]
            if nobjects > 2:
                second_object_index = order[:,2]
                second_object_number = second_object_index+1
                second_x_vector = centers[second_object_index,1] - centers[:,1]
                second_y_vector = centers[second_object_index,0] - centers[:,0]
                v1 = np.array((first_x_vector,first_y_vector))
                v2 = np.array((second_x_vector,second_y_vector))
                #
                # Project the unit vector v1 against the unit vector v2
                #
                dot = (np.sum(v1*v2,0) / 
                       np.sqrt(np.sum(v1**2,0)*np.sum(v2**2,0)))
                angle = np.arccos(dot) * 180. / np.pi
            
            # Make the structuring element for dilation
            strel = strel_disk(distance)
            #
            # Get the extents for each object and calculate the patch
            # that excises the part of the image that is "distance"
            # away
            i,j = np.mgrid[0:labels.shape[0],0:labels.shape[1]]
            min_i, max_i, min_i_pos, max_i_pos =\
                scind.extrema(i,labels,object_indexes)
            min_j, max_j, min_j_pos, max_j_pos =\
                scind.extrema(j,labels,object_indexes)
            min_i = np.maximum(fix(min_i)-distance,0).astype(int)
            max_i = np.minimum(fix(max_i)+distance+1,labels.shape[0]).astype(int)
            min_j = np.maximum(fix(min_j)-distance,0).astype(int)
            max_j = np.minimum(fix(max_j)+distance+1,labels.shape[1]).astype(int)
            #
            # Loop over all objects
            # Calculate which ones overlap "index"
            # Calculate how much overlap there is of others to "index"
            #
            for index in range(nobjects):
                patch = labels[min_i[index]:max_i[index],
                               min_j[index]:max_j[index]]
                #
                # Find the neighbors
                #
                patch_mask = patch==(index+1)
                extended = scind.binary_dilation(patch_mask,strel)
                neighbors = np.setdiff1d(np.unique(patch[extended]),
                                         [0,index+1])
                neighbor_count[index] = len(neighbors)
                #
                # Find the # of overlapping pixels. Dilate the neighbors
                # and see how many pixels overlap our image
                #
                extended = scind.binary_dilation((~ patch_mask) & (patch != 0),
                                                 strel)
                overlap = np.sum(patch_mask & extended)
                pixel_count[index] = overlap
            percent_touching = pixel_count * 100.0 / areas
        else:
            percent_touching = np.zeros((nobjects,))
        #
        # Record the measurements
        #
        m = workspace.measurements
        for feature_name, data in \
            ((M_NUMBER_OF_NEIGHBORS, neighbor_count),
             (M_PERCENT_TOUCHING, percent_touching),
             (M_FIRST_CLOSEST_OBJECT_NUMBER, first_object_number),
             (M_FIRST_CLOSEST_X_VECTOR, first_x_vector),
             (M_FIRST_CLOSEST_Y_VECTOR, first_y_vector),
             (M_SECOND_CLOSEST_OBJECT_NUMBER, second_object_number),
             (M_SECOND_CLOSEST_X_VECTOR, second_x_vector),
             (M_SECOND_CLOSEST_Y_VECTOR, second_y_vector),
             (M_ANGLE_BETWEEN_NEIGHBORS, angle)):
            m.add_measurement(self.object_name.value,
                              '%s_%s_%s'%(C_NEIGHBORS, feature_name, scale),
                              data)
        #
        # Calculate the two heatmap images
        #
        neighbor_count_image = np.zeros(labels.shape,int)
        object_mask = objects.segmented != 0
        object_indexes = objects.segmented[object_mask]-1
        neighbor_count_image[object_mask] = neighbor_count[object_indexes]
        
        percent_touching_image = np.zeros(labels.shape)
        percent_touching_image[object_mask] = percent_touching[object_indexes]
        image_set = workspace.image_set
        if self.wants_count_image.value:
            neighbor_cm = get_colormap(self.count_colormap.value)
            sm = matplotlib.cm.ScalarMappable(cmap = neighbor_cm)
            img = sm.to_rgba(neighbor_count_image)[:,:,:3]
            img[:,:,0][~ object_mask] = 0
            img[:,:,1][~ object_mask] = 0
            img[:,:,2][~ object_mask] = 0
            count_image = cpi.Image(img, masking_objects = objects)
            image_set.add(self.count_image_name.value, count_image)
        else:
            neighbor_cm = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())
        if self.wants_percent_touching_image.value:
            percent_touching_cm = get_colormap(self.touching_colormap.value)
            sm = matplotlib.cm.ScalarMappable(cmap = percent_touching_cm)
            img = sm.to_rgba(percent_touching_image)[:,:,:3]
            img[:,:,0][~ object_mask] = 0
            img[:,:,1][~ object_mask] = 0
            img[:,:,2][~ object_mask] = 0
            touching_image = cpi.Image(img, masking_objects = objects)
            image_set.add(self.touching_image_name.value,
                          touching_image)
        else:
            percent_touching_cm = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())
        if not workspace.frame is None:
            figure = workspace.create_or_find_figure(subplots=(2,2))
            figure.subplot_imshow_labels(0,0,objects.segmented,
                                         "Original: %s"%self.object_name.value)
            neighbor_count_image[~ object_mask] = -1
            neighbor_cm.set_under((0,0,0))
            percent_touching_cm.set_under((0,0,0))
            percent_touching_image[~ object_mask] = -1
            if np.any(object_mask):
                figure.subplot_imshow(0,1, neighbor_count_image,
                                      "%s colored by # of neighbors" %
                                      self.object_name.value,
                                      colormap = neighbor_cm,
                                      colorbar=True, vmin=0)
                figure.subplot_imshow(1,1, percent_touching_image,
                                      "%s colored by pct touching"%
                                      self.object_name.value,
                                      colormap = percent_touching_cm,
                                      colorbar=True, vmin=0)
            else:
                # No objects - colorbar blows up.
                figure.subplot_imshow(0,1, neighbor_count_image,
                                      "%s colored by # of neighbors" %
                                      self.object_name.value,
                                      colormap = neighbor_cm)
                figure.subplot_imshow(1,1, percent_touching_image,
                                      "%s colored by pct touching"%
                                      self.object_name.value,
                                      colormap = percent_touching_cm)
                
            if self.distance_method == D_EXPAND:
                figure.subplot_imshow_labels(1,0, labels,
                                             "Expanded %s"%
                                             self.object_name.value)
    
    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        if self.distance_method == D_EXPAND:
            scale = S_EXPANDED
        elif self.distance_method == D_WITHIN:
            scale = str(self.distance.value)
        elif self.distance_method == D_ADJACENT:
            scale = S_ADJACENT
        coltypes = [cpmeas.COLTYPE_INTEGER 
                    if feature in (M_NUMBER_OF_NEIGHBORS, 
                                   M_FIRST_CLOSEST_OBJECT_NUMBER,
                                   M_SECOND_CLOSEST_OBJECT_NUMBER)
                    else cpmeas.COLTYPE_FLOAT
                    for feature in M_ALL]
        return [(self.object_name.value,
                 '%s_%s_%s'%(C_NEIGHBORS, feature_name, scale),
                 coltype)
                 for feature_name,coltype in zip(M_ALL, coltypes)]
        
    def get_categories(self, pipeline, object_name):
        if object_name == self.object_name:
            return [C_NEIGHBORS]
        return []


    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.object_name and category == C_NEIGHBORS:
            return M_ALL
        return []

    def get_measurement_scales(self, pipeline, object_name, category, measurement, image_name):
        if (object_name == self.object_name and category == C_NEIGHBORS and
            measurement in M_ALL):
            if self.distance_method == D_EXPAND:
                return [S_EXPANDED]
            elif self.distance_method == D_ADJACENT:
                return [S_ADJACENT]
            elif self.distance_method == D_WITHIN:
                return [str(self.distance.value)]
            else:
                raise ValueError("Unknown distance method: %s"%
                                 self.distance_method.value)
        return []
    
def get_colormap(name):
    '''Get colormap, accounting for possible request for default'''
    if name == cps.DEFAULT:
        name = cpprefs.get_default_colormap()
    return matplotlib.cm.get_cmap(name)
