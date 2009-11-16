"""trackobjects.py - Track objects in successive images

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import numpy as np
import matplotlib.figure
import matplotlib.axes
import matplotlib.backends.backend_agg
from scipy.ndimage import distance_transform_edt
import scipy.ndimage
import scipy.sparse

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
import cellprofiler.preferences as cpprefs
import contrib.LAP as LAP
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.cpmath.cpmorphology import centers_of_labels
from cellprofiler.cpmath.cpmorphology import associate_by_distance

TM_OVERLAP = 'Overlap'
TM_DISTANCE = 'Distance'
TM_MEASUREMENTS = 'Measurements'
TM_LAP = "LAP"
TM_ALL = [TM_OVERLAP, TM_DISTANCE, TM_MEASUREMENTS,TM_LAP]

DT_COLOR_AND_NUMBER = 'Color and Number'
DT_COLOR_ONLY = 'Color Only'
DT_ALL = [DT_COLOR_AND_NUMBER, DT_COLOR_ONLY]

F_PREFIX = "TrackObjects"
F_LABEL = "Label"
F_PARENT = "Parent"
F_TRAJECTORY_X = "TrajectoryX"
F_TRAJECTORY_Y = "TrajectoryY"
F_DISTANCE_TRAVELED = "DistanceTraveled"
F_INTEGRATED_DISTANCE = "IntegratedDistance"
F_LINEARITY = "Linearity"
F_LIFETIME = "Lifetime"

F_ALL_COLTYPE_ALL = [(F_LABEL, cpmeas.COLTYPE_INTEGER),
                     (F_PARENT, cpmeas.COLTYPE_INTEGER),
                     (F_TRAJECTORY_X, cpmeas.COLTYPE_INTEGER),
                     (F_TRAJECTORY_Y, cpmeas.COLTYPE_INTEGER),
                     (F_DISTANCE_TRAVELED, cpmeas.COLTYPE_FLOAT),
                     (F_INTEGRATED_DISTANCE, cpmeas.COLTYPE_FLOAT),
                     (F_LINEARITY, cpmeas.COLTYPE_FLOAT),
                     (F_LIFETIME, cpmeas.COLTYPE_INTEGER)]

F_ALL = [feature for feature, coltype in F_ALL_COLTYPE_ALL]

class TrackObjects(cpm.CPModule):
    '''SHORT DESCRIPTION:
Allows tracking objects throughout sequential frames of a movie, so that
each object maintains a unique identity in the output measurements.
*************************************************************************
This module must be run after the object to be tracked has been 
identified using an Identification module (e.g., IdentifyPrimAutomatic).

Settings:

Tracking method:
Choose between the methods based on which is most consistent from frame
to frame of your movie. For each, the maximum search distance that a 
tracked object will looked for is specified with the Neighborhood setting
below:

Overlap - Compare the amount of overlaps between identified objects in 
the previous frame with those in the current frame. The object with the
greatest amount of overlap will be assigned the same label. Recommended
for movies with high frame rates as compared to object motion.

Distance - Compare the distance between the centroid of each identified
object in the previous frame with that of the current frame. The 
closest objects to each other will be assigned the same label.
Distances are measured from the perimeter of each object. Recommended
for movies with lower frame rates as compared to object motion, but
the objects are clearly separable.

Measurement - Compare the specified measurement of each object in the 
current frame with that of objects in the previous frame. The object 
with the closest measurement will be selected as a match and will be 
assigned the same label. This selection requires that you run the 
specified Measurement module previous to this module in the pipeline so
that the measurement values can be used to track the objects. 

Catagory/Feature Name or Number/Image/Scale:
Specifies which type of measurement (catagory) and which feature from the
Measure module will be used for tracking. Select the feature name from 
the popup box or see each Measure module's help for the numbered list of
the features measured by that module. Additional details such as the 
image that the measurements originated from and the scale used as
specified below if neccesary.

Neighborhood:
This indicates the region (in pixels) within which objects in the
next frame are to be compared. To determine pixel distances, you can look
at the markings on the side of each image (shown in pixel units) or
using the ShowOrHidePixelData Image tool (under the Image Tools menu of 
any CellProfiler figure window)

How do you want to display the tracked objects?
The objects can be displayed as a color image, in which an object with a 
unique label is assigned a unique color. This same color is maintained
throughout the object's lifetime. If desired, a number identifiying the 
object is superimposed on the object.

What number do you want displayed?
The displayed number is the unique label assigned to the object or the
progeny identifier.

Do you want to calculate statistics:
Select whether you want statistics on the tracked objects to be added to
the measurements for that object. The current statistics are collected:

Features measured:
TrajectoryX
TrajectoryY
DistanceTraveled
IntegratedDistance
Linearity
Parent
Label
Lifetime

Desscription of each feature:
Label: Each tracked object is assigned a unique identifier (label). 
Results of splits or merges are seen as new objects and assigned a new
label.

Parent: The label of the object in the last frame. For a split, each
child object will have the label of the object it split from. For a merge,
the child will have the label of the closest parent.

Trajectory: The direction of motion (in x and y coordinates) of the 
object from the previous frame to the curent frame.

Distance traveled: The distance traveled by the object from the 
previous frame to the curent frame (calculated as the magnititude of 
the distance traveled vector).

Lifetime: The duration (in frames) of the object. The lifetime begins 
at the frame when an object appears and is ouput as a measurement when
the object disappears. At the final frame of the image set/movie, the 
lifetimes of all remaining objects are ouput.

Integrated distance: The total distance traveled by the object during
the lifetime of the object

Linearity: A measure of how linear the object trajectity is during the
object lifetime. Calculated as (distance from initial to final 
location)/(integrated object distance). Value is in range of [0,1].

What do you want to call the image with the tracked objects?
Specify a name to give the image showing the tracked objects. This image
can be saved with a SaveImages module placed after this module.

Additional notes:

In the figure window, a popupmenu allows you to display the objects as a
solid color or as an outline with the current objects in color and the
previous objects in white.

Since the movie is processed sequentially by frame, it cannot be broken
up into batches for execution on a distributed cluster.

If running on a cluster and saving the colored image with text labels,
the labels will not show up in the final result. This is a limitation of
using MATLAB's hardcopy command.

See also: Any of the Measure* modules, IdentifyPrimAutomatic
'''
    category = "Object Processing"
    variable_revision_number = 2
    
    def create_settings(self):
        self.module_name = 'TrackObjects'
        self.tracking_method = cps.Choice('Choose a tracking method',
                                          TM_ALL)
        self.object_name = cps.ObjectNameSubscriber(
            'What did you call the objects you want to track?','None')
        self.measurement = cps.Measurement(
            'What measurement do you want to use?',
            lambda : self.object_name.value)
        self.pixel_radius = cps.Integer(
            'Within what pixel distance will objects be considered to find '
            'a potential match?',50,minval=1)
	self.born_cost = cps.Integer(
	    'What is the cost of an object being born?', 100, minval=1)
	self.die_cost = cps.Integer(
	    'What is the cost of an object dying?', 100, minval=1)
        self.display_type = cps.Choice(
            'How do you want to display the tracked objects?',
            DT_ALL)
        self.wants_image = cps.Binary(
            "Do you want to save the image with tracked, color-coded objects?",
            False)
        self.image_name = cps.ImageNameProvider(
            "What do you want to call the images?", "TrackedCells")

    def settings(self):
        return [self.tracking_method, self.object_name, self.measurement,
                self.pixel_radius, self.display_type, self.wants_image,
                self.image_name, self.born_cost, self.die_cost]

    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        if from_matlab and variable_revision_number == 3:
            wants_image = setting_values[10] != cps.DO_NOT_USE
            measurement =  '_'.join(setting_values[2:6])
            setting_values = [ setting_values[0], # tracking method
                               setting_values[1], # object name
                               measurement,
                               setting_values[6], # pixel_radius
                               setting_values[7], # display_type
                               wants_image,
                               setting_values[10]]
            variable_revision_number = 1
            from_matlab = False
	if (not from_matlab) and variable_revision_number == 1:
	    setting_values += [100,100]
	    variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

    def visible_settings(self):
        result = [self.tracking_method, self.object_name]
        if self.tracking_method == TM_MEASUREMENTS:
            result += [ self.measurement]
        result += [self.pixel_radius]
	if self.tracking_method == TM_LAP:
	    result += [self.born_cost, self.die_cost]
	result +=[ self.display_type, self.wants_image]
        if self.wants_image.value:
            result += [self.image_name]
        return result

    @property
    def module_key(self):
        return "TrackObjects_%d" % self.module_num
    
    def get_dictionary(self, workspace):
        return workspace.image_set_list.legacy_fields[self.module_key]

    def __get(self, field, workspace, default):
        if self.get_dictionary(workspace).has_key(field):
            return self.get_dictionary(workspace)[field]
        return default
    
    def __set(self, field, workspace, value):
        self.get_dictionary(workspace)[field] = value
        
    def get_saved_measurements(self, workspace):
        return self.__get("measurements", workspace, np.array([], float))
    
    def set_saved_measurements(self, workspace, value):
        self.__set("measurements", workspace, value)
    
    def get_saved_coordinates(self, workspace):
        return self.__get("coordinates", workspace, np.zeros((2,0), int))
    
    def set_saved_coordinates(self, workspace, value):
        self.__set("coordinates", workspace, value)
    
    def get_orig_coordinates(self, workspace):
        '''The coordinates of the first occurrence of an object's ancestor'''
        return self.__get("orig coordinates", workspace, np.zeros((2,0), int))
    
    def set_orig_coordinates(self, workspace, value):
        self.__set("orig coordinates", workspace, value)
    
    def get_saved_labels(self, workspace):
        return self.__get("labels", workspace, None)
    
    def set_saved_labels(self, workspace, value):
        self.__set("labels", workspace, value)
    
    def get_saved_object_numbers(self, workspace):
        return self.__get("object_numbers", workspace, np.array([], int))
    
    def set_saved_object_numbers(self, workspace, value):
        return self.__set("object_numbers", workspace, value)
    
    def get_saved_ages(self, workspace):
        return self.__get("ages", workspace, np.array([], int))
    
    def set_saved_ages(self, workspace, values):
        self.__set("ages", workspace, values)
    
    def get_saved_distances(self, workspace):
        return self.__get("distances", workspace, np.zeros((0,)))
    
    def set_saved_distances(self, workspace, values):
        self.__set("distances", workspace, values)
    
    def get_max_object_number(self, workspace):
        return self.__get("max_object_number", workspace, 0)
    
    def set_max_object_number(self, workspace, value):
        self.__set("max_object_number", workspace, value)
    
    def prepare_run(self, pipeline, image_set_list, frame):
        '''Erase any tracking information at the start of a run'''
        image_set_list.legacy_fields[self.module_key] = {}
        return True
    
    def measurement_name(self, feature):
        '''Return a measurement name for the given feature'''
        return "%s_%s_%s" % (F_PREFIX, feature, str(self.pixel_radius.value))
    
    def add_measurement(self, workspace, feature, values):
        '''Add a measurement to the workspace's measurements
        
        workspace - current image set's workspace
        feature - name of feature being measured
        values - one value per object
        '''
        workspace.measurements.add_measurement(
            self.object_name.value,
            self.measurement_name(feature),
            values)
        
    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.object_name.value)
        if self.tracking_method == TM_DISTANCE:
            self.run_distance(workspace, objects)
        elif self.tracking_method == TM_OVERLAP:
            self.run_overlap(workspace, objects)
        elif self.tracking_method == TM_MEASUREMENTS:
            self.run_measurements(workspace, objects)
	elif self.tracking_method == TM_LAP:
	    self.run_lapdistance(workspace, objects)
        else:
            raise NotImplementedError("Unimplemented tracking method: %s" %
                                      self.tracking_method.value)
        draw = False
        if workspace.frame is not None:
            frame = workspace.create_or_find_figure(subplots=(1,1))
            figure = frame.figure
            figure.clf()
            canvas = figure.canvas
            draw = True
        elif self.wants_image.value:
            figure = matplotlib.figure.Figure()
            canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(figure)
            draw=True
        if draw:
            ax = figure.add_subplot(1,1,1)
            objects = workspace.object_set.get_objects(self.object_name.value)
            object_numbers = self.get_saved_object_numbers(workspace)
            indexer = np.zeros(len(object_numbers)+1,int)
            indexer[1:] = object_numbers
            #
            # We want to keep the colors stable, but we also want the
            # largest possible separation between adjacent colors. So, here
            # we reverse the significance of the bits in the indices so
            # that adjacent number (e.g. 0 and 1) differ by 128, roughly
            #
            pow_of_2 = 2**np.mgrid[0:8,0:len(indexer)][0]
            bits = (indexer & pow_of_2).astype(bool)
            indexer = np.sum(bits.transpose() * (2 ** np.arange(7,-1,-1)), 1)
            labels = indexer[objects.segmented]
            cm = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())
            norm = matplotlib.colors.BoundaryNorm(range(256), 256)
            img = ax.imshow(labels, cmap=cm, norm=norm)
            i,j = centers_of_labels(objects.segmented)
            for n, x, y in zip(object_numbers, j, i):
                ax.annotate(str(n), xy=(x,y),
                            arrowprops=dict(visible=False))
            if self.wants_image.value:
                # This is the recipe that gets a canvas to render itself
                # and then convert the resulting raster into the typical
                # Numpy color image format.
                canvas.draw()
                width, height = canvas.get_width_height()
                data = canvas.tostring_rgb()
                image_pixels = np.fromstring(data,np.uint8)
                image_pixels.shape = (height, width, 3)
                image = cpi.Image(image_pixels)
                workspace.image_set.add(self.image_name.value, image)
            
    def run_distance(self, workspace, objects):
        '''Track objects based on distance'''
        old_i, old_j = self.get_saved_coordinates(workspace)
        if len(old_i):
            distances, (i,j) = distance_transform_edt(objects.segmented == 0,
                                                      return_indices=True)
            #
            # Look up the coordinates of the nearest new object (given by
            # the transform i,j), then look up the label at that coordinate
            # (objects.segmented[#,#])
            #
            new_object_numbers = objects.segmented[i[old_i, old_j],
                                                   j[old_i, old_j]]
            #
            # Mask out any objects at too great of a distance
            #
            new_object_numbers[distances[old_i, old_j] >
                               self.pixel_radius.value] = 0
            #
            # Do the same with the new centers and old objects
            #
            i,j = (centers_of_labels(objects.segmented)+.5).astype(int)
            old_labels = self.get_saved_labels(workspace)
            distances, (old_i,old_j) = distance_transform_edt(
                old_labels == 0,
                return_indices=True)
            old_object_numbers = old_labels[old_i[i, j],
                                            old_j[i, j]]
            old_object_numbers[distances[i, j] > self.pixel_radius.value] = 0
            self.map_objects(workspace, 
                             new_object_numbers,
                             old_object_numbers, 
                             i,j)
        else:
            i,j = (centers_of_labels(objects.segmented)+.5).astype(int)
            count = len(i)
            self.map_objects(workspace, np.zeros((0,),int), 
                             np.zeros(count,int), i,j)
        self.set_saved_labels(workspace, objects.segmented)
    
    def run_lapdistance(self, workspace, objects):
        '''Track objects based on distance'''
	costBorn = self.born_cost.value
	costDie = self.die_cost.value
	minDist = self.pixel_radius.value
        old_i, old_j = self.get_saved_coordinates(workspace)
        if len(old_i):
            new_i, new_j = centers_of_labels(objects.segmented)
            i,j = np.mgrid[0:len(old_i), 0:len(new_i)]
            d = np.sqrt((old_i[i]-new_i[j])**2 + (old_j[i]-new_j[j])**2)
            n = len(old_i)+len(new_i)
            kk = np.zeros((n+10)*(n+10), np.int32)
            first = np.zeros(n+10, np.int32)
            cc = np.zeros((n+10)*(n+10), np.float)
            t = np.argwhere((d < minDist))
            x = np.sqrt((old_i[t[0:t.size, 0]]-new_i[t[0:t.size, 1]])**2 + (old_j[t[0:t.size, 0]]-new_j[t[0:t.size, 1]])**2)
            t = t+1
            t = np.column_stack((t, x))
            a = np.arange(len(old_i))+2
            x = np.searchsorted(t[0:(t.size/2),0], a)
            a = np.arange(len(old_i))+1
            b = np.arange(len(old_i))+len(new_i)+1
            c = np.zeros(len(old_i))+costDie
            b = np.column_stack((a, b, c))
            t = np.insert(t, x, b, 0)
            
            i,j = np.mgrid[0:len(new_i),0:len(old_i)+1]
            i = i+len(old_i)+1
            j = j+len(new_i)
            j[0:len(new_i)+1,0] = i[0:len(new_i)+1,0]-len(old_i)
            x = np.zeros((len(new_i),len(old_i)+1))
            x[0:len(new_i)+1,0] = costBorn
            i = i.flatten()
            j = j.flatten()
            x = x.flatten()
            x = np.column_stack((i, j, x))
            t = np.vstack((t, x))
        
            kk = np.ndarray.astype(t[0:(t.size/3),1], 'int32')
            cc = t[0:(t.size/3),2]
        
            a = np.arange(len(old_i)+len(new_i)+2)
            first = np.bincount(np.ndarray.astype(t[0:(t.size/3),0], 'int32')+1)
            first = np.cumsum(first)+1
        
            first[0] = 0
            kk = np.hstack((np.array((0)), kk))
            cc = np.hstack((np.array((0.0)), cc))

            x, y =  LAP.LAP(kk, first, cc, n)
            a = np.argwhere(x > len(new_i))
	    b = np.argwhere(y >len(old_i))
            x[a[0:len(a)]] = 0
	    y[b[0:len(b)]] = 0
	    a = np.arange(len(old_i))+1
	    b = np.arange(len(new_i))+1
            new_object_numbers = x[a[0:len(a)]]
	    old_object_numbers = y[b[0:len(b)]]
            
            
            i,j = (centers_of_labels(objects.segmented)+.5).astype(int)
            self.map_objects(workspace, 
                             new_object_numbers,
                             old_object_numbers, 
                             i,j)
        else:
            i,j = (centers_of_labels(objects.segmented)+.5).astype(int)
            count = len(i)
            self.map_objects(workspace, np.zeros((0,),int), 
                             np.zeros(count,int), i,j)
        self.set_saved_labels(workspace, objects.segmented)
    
    def run_overlap(self, workspace, objects):
        '''Track objects by maximum # of overlapping pixels'''
        current_labels = objects.segmented
        old_labels = self.get_saved_labels(workspace)
        i,j = (centers_of_labels(objects.segmented)+.5).astype(int)
        if old_labels is None:
            count = len(i)
            self.map_objects(workspace, np.zeros((0,),int), 
                             np.zeros(count,int), i,j)
        else:
            mask = ((current_labels > 0) & (old_labels > 0))
            cur_count = np.max(current_labels)
            old_count = np.max(old_labels)
            count = np.sum(mask)
            if count == 0:
                # There's no overlap.
                self.map_objects(workspace,
                                 np.zeros(old_count, int),
                                 np.zeros(cur_count,int),
                                 i,j)
            else:
                cur = current_labels[mask]
                old = old_labels[mask]
                histogram = scipy.sparse.coo_matrix(
                    (np.ones(count),(cur, old)),
                    shape=(cur_count+1,old_count+1)).toarray()
                old_of_new = np.argmax(histogram, 1)[1:]
                new_of_old = np.argmax(histogram, 0)[1:]
                #
                # The cast here seems to be needed to make scipy.ndimage.sum
                # work. See http://projects.scipy.org/numpy/ticket/1012
                #
                old_of_new = np.array(old_of_new, np.int16)
                old_of_new = np.array(old_of_new, np.int32)
                new_of_old = np.array(new_of_old, np.int16)
                new_of_old = np.array(new_of_old, np.int32)
                self.map_objects(workspace,
                                 new_of_old,
                                 old_of_new,
                                 i,j)
        self.set_saved_labels(workspace, current_labels)
    
    def run_measurements(self, workspace, objects):
        current_labels = objects.segmented
        new_measurements = workspace.measurements.get_current_measurement(
            self.object_name.value,
            self.measurement.value)
        old_measurements = self.get_saved_measurements(workspace)
        old_labels = self.get_saved_labels(workspace)
        i,j = (centers_of_labels(objects.segmented)+.5).astype(int)
        if old_labels is None:
            count = len(i)
            self.map_objects(workspace, np.zeros((0,),int), 
                             np.zeros(count,int), i,j)
        else:
            associations = associate_by_distance(old_labels, current_labels,
                                                 self.pixel_radius.value)
            best_child = np.zeros(len(old_measurements), int)
            best_parent = np.zeros(len(new_measurements), int)
            best_child_measurement = (np.ones(len(old_measurements), int) *
                                      np.finfo(float).max)
            best_parent_measurement = (np.ones(len(new_measurements), int) *
                                      np.finfo(float).max)
            for old, new in associations:
                diff = abs(old_measurements[old-1] - new_measurements[new-1])
                if diff < best_child_measurement[old-1]:
                    best_child[old-1] = new
                    best_child_measurement[old-1] = diff
                if diff < best_parent_measurement[new-1]:
                    best_parent[new-1] = old
                    best_parent_measurement[new-1] = diff
            self.map_objects(workspace, best_child, best_parent, i,j)
        self.set_saved_labels(workspace,current_labels)
        self.set_saved_measurements(workspace, new_measurements)
            
    def map_objects(self, workspace, new_of_old, old_of_new, i,j):
        '''Record the mapping of old to new objects and vice-versa
        
        workspace - workspace for current image set
        new_to_old - an array of the new labels for every old label
        old_to_new - an array of the old labels for every new label
        score - a score that is higher for better mappings: we use this
                score to assign new label numbers so that the new objects
                that are "better" inherit the old objects' label number
        '''
        old_object_numbers = self.get_saved_object_numbers(workspace)
        max_object_number = self.get_max_object_number(workspace)
        old_count = len(new_of_old)
        new_count = len(old_of_new)
        #
        # Record the new objects' parents
        #
        parents = old_of_new.copy()
        parents[parents != 0] = old_object_numbers[old_of_new[parents!=0]-1]
        self.add_measurement(workspace, F_PARENT, parents)
        #
        # Assign object IDs to the new objects if unambiguous
        #
        mapping = np.zeros(new_count, int)
        if old_count:
            new_per_old = fix(scipy.ndimage.sum(np.ones(new_count),
                                                old_of_new,
                                                np.arange(old_count)+1))
            one_to_one = new_per_old == 1
            mapping[new_of_old[one_to_one]-1] = old_object_numbers[one_to_one]
            miss_count = np.sum(mapping == 0)
        else:
            miss_count = new_count
        mapping[mapping == 0] = np.arange(miss_count)+max_object_number+1
        self.set_max_object_number(workspace, miss_count + max_object_number)
        self.add_measurement(workspace, F_LABEL, mapping)
        self.set_saved_object_numbers(workspace, mapping)
        #
        # Compute distances and trajectories
        #
        diff_i = np.zeros(new_count, int)
        diff_j = np.zeros(new_count, int)
        distance = np.zeros(new_count)
        idistance = np.zeros(new_count)
        odistance = np.zeros(new_count)
        distance = np.zeros(new_count)
        linearity = np.ones(new_count)
        orig_i = i.copy()
        orig_j = j.copy()
        old_i, old_j = self.get_saved_coordinates(workspace)
        old_distance = self.get_saved_distances(workspace)
        old_orig_i, old_orig_j = self.get_orig_coordinates(workspace)
        has_old = (old_of_new != 0)
        if np.any(has_old):
            old_indexes = old_of_new[has_old]-1
            orig_i[has_old] = old_orig_i[old_indexes]
            orig_j[has_old] = old_orig_j[old_indexes]
            diff_i[has_old] = i[has_old] - old_i[old_indexes]
            diff_j[has_old] = j[has_old] - old_j[old_indexes]
            distance[has_old] = np.sqrt(diff_i[has_old]**2 + diff_j[has_old]**2)
    
            idistance[has_old] = (old_distance[old_indexes] + 
                                  distance[has_old])
            odistance = np.sqrt((i-orig_i)**2 + (j-orig_j)**2)
            linearity[has_old] = odistance[has_old] / idistance[has_old]
        self.add_measurement(workspace, F_TRAJECTORY_X, diff_j)
        self.add_measurement(workspace, F_TRAJECTORY_Y, diff_i)
        self.add_measurement(workspace, F_DISTANCE_TRAVELED, distance)
        self.add_measurement(workspace, F_INTEGRATED_DISTANCE, idistance)
        self.add_measurement(workspace, F_LINEARITY, linearity)
        self.set_saved_distances(workspace, idistance)
        self.set_orig_coordinates(workspace, (orig_i, orig_j))
        self.set_saved_coordinates(workspace, (i,j))
        #
        # Update the ages
        #
        age = np.zeros(new_count, int)
        if np.any(has_old):
            old_age = self.get_saved_ages(workspace)
            age[has_old] = old_age[old_of_new[has_old]-1]+1
        self.add_measurement(workspace, F_LIFETIME, age)
        self.set_saved_ages(workspace, age)
        self.set_saved_object_numbers(workspace, mapping)

    def get_measurement_columns(self, pipeline):
        return [(self.object_name.value,
                 self.measurement_name(feature),
                 coltype)
                for feature, coltype in F_ALL_COLTYPE_ALL]

    def get_categories(self, pipeline, object_name):
        if object_name == self.object_name.value:
            return [F_PREFIX]
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.object_name.value and category == F_PREFIX:
            return F_ALL
        return []
    
    def get_measurement_scales(self, pipeline, object_name, category, feature):
        if (object_name == self.object_name.value and
            category == F_PREFIX and
            feature in F_ALL):
            return [str(self.pixel_radius.value)]
        return []
        
    