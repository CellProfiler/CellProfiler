'''<b>CellStarTracking</b> tracks yeast cells in the series of images
<hr>

<h3>Technical notes</h3>

'''

#################################
#
# Imports from useful Python libraries
#
#################################

import sys
import random
from os.path import expanduser
from os.path import join as pj
import numpy as np
import scipy as sp
from scipy.ndimage.filters import *
from scipy.ndimage import distance_transform_edt
import scipy.ndimage
import scipy.sparse
import matplotlib.figure
import matplotlib.axes
import matplotlib.backends.backend_agg

#################################
#
# Imports from CellProfiler
#
##################################
try:
    import cellprofiler.cpmodule as cpm
    import cellprofiler.cpimage as cpi
    import cellprofiler.pipeline as cpp
    import cellprofiler.measurements as cpmeas
    import cellprofiler.settings as cps
    import cellprofiler.preferences as cpprefs
    import cellprofiler.cpmath as cpmath
    import cellprofiler.cpmath.outline
    import cellprofiler.objects
    from cellprofiler.cpmath.cpmorphology import centers_of_labels
    from cellprofiler.gui.help import HELP_ON_MEASURING_DISTANCES

#################################
#
# Specific imports
#
##################################

    from cellprofiler.cpmath.neighmovetrack import NeighbourMovementTracking

except ImportError as e: 
    # in new version 2.12 all the errors are properly shown in console (Windows)
    home = expanduser("~") # in principle it is system independent
    with open(pj(home,"cs_log.txt"), "a+") as log:
        log.write("Import exception")
        log.write(e.message)
    raise


###################################
#
# Constants
#
###################################

DEBUG = 1

###################################
#
# The module class
#
###################################

TM_DISTANCE = 'Distance'

DT_COLOR_AND_NUMBER = 'Color and Number'
DT_COLOR_ONLY = 'Color Only'
DT_ALL = [DT_COLOR_AND_NUMBER, DT_COLOR_ONLY]

F_PREFIX = "TrackObjects"
F_LABEL = "Label"
F_PARENT_OBJECT_NUMBER = "ParentObjectNumber"
F_PARENT_IMAGE_NUMBER = "ParentImageNumber"
F_TRAJECTORY_X = "TrajectoryX"
F_TRAJECTORY_Y = "TrajectoryY"
F_DISTANCE_TRAVELED = "DistanceTraveled"
F_INTEGRATED_DISTANCE = "IntegratedDistance"
F_LINEARITY = "Linearity"
F_LIFETIME = "Lifetime"
F_FINAL_AGE = "FinalAge"
F_EXPT_ORIG_NUMTRACKS = "%s_OriginalNumberOfTracks"%F_PREFIX
F_EXPT_FILT_NUMTRACKS = "%s_FilteredNumberOfTracks"%F_PREFIX
                                     
'''# of objects in the current frame without parents in the previous frame'''
F_NEW_OBJECT_COUNT = "NewObjectCount"
'''# of objects in the previous frame without parents in the new frame'''
F_LOST_OBJECT_COUNT = "LostObjectCount"
'''# of parents that split into more than one child'''
F_SPLIT_COUNT = "SplitCount"
'''# of children that are merged from more than one parent'''
F_MERGE_COUNT = "MergeCount"
'''Object area measurement for LAP method'''

F_ALL_COLTYPE_ALL = [(F_LABEL, cpmeas.COLTYPE_INTEGER),
                     (F_PARENT_OBJECT_NUMBER, cpmeas.COLTYPE_INTEGER),
                     (F_PARENT_IMAGE_NUMBER, cpmeas.COLTYPE_INTEGER),
                     (F_TRAJECTORY_X, cpmeas.COLTYPE_INTEGER),
                     (F_TRAJECTORY_Y, cpmeas.COLTYPE_INTEGER),
                     (F_DISTANCE_TRAVELED, cpmeas.COLTYPE_FLOAT),
                     (F_INTEGRATED_DISTANCE, cpmeas.COLTYPE_FLOAT),
                     (F_LINEARITY, cpmeas.COLTYPE_FLOAT),
                     (F_LIFETIME, cpmeas.COLTYPE_INTEGER),
                     (F_FINAL_AGE, cpmeas.COLTYPE_INTEGER)]

F_IMAGE_COLTYPE_ALL = [(F_NEW_OBJECT_COUNT, cpmeas.COLTYPE_INTEGER),
                       (F_LOST_OBJECT_COUNT, cpmeas.COLTYPE_INTEGER),
                       (F_SPLIT_COUNT, cpmeas.COLTYPE_INTEGER),
                       (F_MERGE_COUNT, cpmeas.COLTYPE_INTEGER)]

F_ALL = [feature for feature, coltype in F_ALL_COLTYPE_ALL]

F_IMAGE_ALL = [feature for feature, coltype in F_IMAGE_COLTYPE_ALL]

class YeastCellTracking(cpm.CPModule):

    module_name = 'YeastCellTracking'
    category = "Object Processing"
    variable_revision_number = 3

    ###########################
    #
    #   Module implementation 
    #
    ###########################
    
    def create_settings(self):
        
        self.object_name = cps.ObjectNameSubscriber(
            'Select the objects to track','None', doc="""How do you call the objects you want to track?""")
        
        #self.pixel_radius = cps.Integer(
        #    'Maximum pixel distance to consider matches',50,minval=1,doc="""
        #    Objects in the subsequent frame will be considered potential matches if 
        #    they are within this distance. To determine a suitable pixel distance, you can look
        #    at the axis increments on each image (shown in pixel units) or
        #    use the distance measurement tool. %(HELP_ON_MEASURING_DISTANCES)s"""%globals())

        self.average_cell_diameter = cps.Float(
            "Average cell diameter in pixels",
            35.0, minval=5, doc ='''\
            The average cell diameter is used to scale many algorithm parameters. 
            Please use e.g. ImageJ to measure the average cell size in pixels.
            '''
            )

        self.advanced_parameters = cps.Binary(
            'Use advanced configuration parameters', False, doc="""
            Do you want to use advanced parameters to configure plugin? They allow for more flexibility,
            however you need to know what you are doing.
            """
            )

        self.iterations = cps.Integer(
            "Number of matching steps",
            5, minval=0, maxval=15, doc ='''\
            Number of the matching improving iterations. Higher values allow to 
            correct coordinated colony drifts, but they increase the evaluation 
            time of the algorithm. Please see original paper of Delgado-Gonzalo et al. 2010 
            "Multi-target tracking of packed yeast cells" for more details (the parameter
            corresponds to number of repeats in Algorithm 1).
            '''
            )
        
        self.drop_cost = cps.Float(
            "Cost of cell to empty matching",
            15, minval=1, maxval=50, doc='''\
            The cost of assigning cell as "lost" in transition from t to t+1. Increasing this value leads to more 
            cells (from t) being matched with cells (from t+1) rather then classified as "lost".
            Too high value might cause incorrect cells to match between the frames. 
            Too lower might make the algorithm not to match cells between the frames.
            '''
            )
            
        self.areaWeight = cps.Float(
            "Weight of area difference in matching cost",
            25, minval=1, doc='''\
            Increasing this value will cause the algorithm to care more about area consistence between frames and less about distance between them.
            '''
            )
            
        self.display_type = cps.Choice(
            'Select display option', DT_ALL, doc="""
            How do you want to display the tracked objects?
            The output image can be saved as either a color-labeled image, with each tracked
            object assigned a unique color, or as a color-labeled image with the tracked object 
            number superimposed.""")

        self.wants_image = cps.Binary(
            "Save color-coded image?", False, doc="""
            Do you want to save the image with tracked, color-coded objects?
            Specify a name to give the image showing the tracked objects. This image
            can be saved with a <b>SaveImages</b> module placed after this module.""")

        self.image_name = cps.ImageNameProvider(
            "Name the output image", "TrackedCells", doc = '''
            <i>(Used only if saving the color-coded image)</i><br>
            What do you want to call the color-coded image, which will be available for downstream modules, such as <b>SaveImages</b>?''')

    def settings(self):
        return [self.object_name,self.average_cell_diameter, self.iterations,self.drop_cost,self.areaWeight, self.display_type, self.wants_image, #self.pixel_radius,
                self.image_name,self.advanced_parameters]
                
    def visible_settings(self):
        result = [self.object_name, self.average_cell_diameter, self.advanced_parameters]
        if self.advanced_parameters:
            result += [self.iterations,self.drop_cost,self.areaWeight] 
        result += [self.display_type, self.wants_image]
        if self.wants_image.value:
            result += [self.image_name]
        return result
    
    def is_interactive(self):
        return False
    
    def display(self, workspace, figure=None):
        # Shows both segmented labels and numbers.
        figure = workspace.create_or_find_figure(title="TrackObjects, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(1,1))
        ax = figure.subplot(0, 0)
        self.draw(workspace.display_data.labels, ax,
                  workspace.display_data.object_numbers)
        lead_subplot = figure.subplot(0,0)

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
        if variable_revision_number == 1:
            setting_values = setting_values[:1] + [35.0] + setting_values[1:]
            variable_revision_number = 2
        if variable_revision_number == 2:
            setting_values = setting_values + [False]
            variable_revision_number = 3
        return setting_values, variable_revision_number, from_matlab

    def run(self, workspace):
        import time
        start = time.clock()
        objects = workspace.object_set.get_objects(self.object_name.value)
        self.run_localised_matching(workspace, objects)

        # Prepare output images
        if self.wants_image.value:
            import matplotlib.transforms
            # [TODO OLD]
            #from cellprofiler.gui.cpfigure_tools import figure_to_image, only_display_image
            from cellprofiler.gui.cpfigure import figure_to_image, only_display_image
            
            figure = matplotlib.figure.Figure()
            canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(figure)
            ax = figure.add_subplot(1,1,1)
            self.draw(objects.segmented, ax, 
                      self.get_saved_object_numbers(workspace))
            #
            # This is the recipe for just showing the axis
            #
            only_display_image(figure, objects.segmented.shape)
            
            # [OPT] Filip!!: It is veeery slow (10 sec).
        
            image_pixels = figure_to_image(figure, dpi=figure.dpi)
            image = cpi.Image(image_pixels)
            workspace.image_set.add(self.image_name.value, image)
            
        #if workspace.frame is not None:
            workspace.display_data.labels = objects.segmented
            workspace.display_data.object_numbers = \
                     self.get_saved_object_numbers(workspace)
        end = time.clock()
        print "tracking_plugin", end - start
    
    ###########################
    #
    #   Measurements 
    #
    ###########################
        
    def measurement_name(self, feature):
        '''Return a measurement name for the given feature'''
        return "%s_%s_%s" % (F_PREFIX, feature, "1") #str(self.pixel_radius.value))
    
    def image_measurement_name(self, feature):
        '''Return a measurement name for an image measurement'''
        return "%s_%s_%s_%s" % (F_PREFIX, feature, self.object_name.value,
                               "1") #str(self.pixel_radius.value))

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

    def add_image_measurement(self, workspace, feature, value):
        measurement_name = self.image_measurement_name(feature)
        workspace.measurements.add_image_measurement(measurement_name, value)
        
    def get_measurement_columns(self, pipeline):
        result =  [(self.object_name.value,
                    self.measurement_name(feature),
                    coltype)
                   for feature, coltype in F_ALL_COLTYPE_ALL]
        result += [(cpmeas.IMAGE, self.image_measurement_name(feature), coltype)
                   for feature, coltype in F_IMAGE_COLTYPE_ALL]
        return result

    def get_categories(self, pipeline, object_name):
        if object_name in (self.object_name.value, cpmeas.IMAGE):
            return [F_PREFIX]
        elif object_name == cpmeas.EXPERIMENT:
            return [F_PREFIX]
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.object_name.value and category == F_PREFIX:
            result = list(F_ALL)
            return result
        if object_name == cpmeas.IMAGE:
            result = F_IMAGE_ALL
            return result
        if object_name == cpmeas.EXPERIMENT and category == F_PREFIX:
            return [F_EXPT_ORIG_NUMTRACKS, F_EXPT_FILT_NUMTRACKS]
        return []

    def get_measurement_objects(self, pipeline, object_name, category, measurement):
        if (object_name == cpmeas.IMAGE and category == F_PREFIX and
            measurement in F_IMAGE_ALL):
            return [ self.object_name.value]
        return []
    
    def get_measurement_scales(self, pipeline, object_name, category, feature,image_name):
        if feature in self.get_measurements(pipeline, object_name, category):
            return ["1"]#str(self.pixel_radius.value)]
        return []
    
    #def prepare_group(self, pipeline, image_set_list, grouping, image_numbers):
    #    '''Erase any tracking information at the start of a run'''
    #    d = self.get_dictionary(image_set_list)
    #    d.clear()

    #    return True
        
    def prepare_group(self, workspace, grouping, image_numbers):
        '''Erase any tracking information at the start of a run'''
        d = self.get_dictionary(workspace.image_set_list)
        d.clear()
       
        return True
        
    ###########################
    #
    #   Utilities 
    #
    ###########################
        
    def draw(self, labels, ax, object_numbers):
        # Draw image numbers on labels image.
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
        recolored_labels = indexer[labels]
        cm = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())
        cm.set_bad((0,0,0))
        norm = matplotlib.colors.BoundaryNorm(range(256), 256)
        img = ax.imshow(np.ma.array(recolored_labels, mask=(labels==0)),
                        cmap=cm, norm=norm)
        if DEBUG>0: print "geted image"
        if self.display_type == DT_COLOR_AND_NUMBER:
            if DEBUG>0:print "finals"
            i,j = centers_of_labels(labels)
            for n, x, y in zip(object_numbers, j, i):
                if np.isnan(x) or np.isnan(y):
                    # This happens if there are missing labels
                    continue
                ax.annotate(str(n), xy=(x,y),color='white',
                            arrowprops=dict(visible=False))

        if DEBUG>0: print "end"
        
    ###########################
    #
    #   Get set data methods
    #
    ###########################
    
    def get_ws_dictionary(self, workspace):
        return self.get_dictionary(workspace.image_set_list)
        
    def __get(self, field, workspace, default):
        if self.get_ws_dictionary(workspace).has_key(field):
            return self.get_ws_dictionary(workspace)[field]
        return default

    def __set(self, field, workspace, value):
        self.get_ws_dictionary(workspace)[field] = value
  
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

    ###########################
    #
    #   Algorithm logic 
    #
    ###########################
    
    def run_localised_matching(self, workspace, objects):
        '''Track based on localised matching costs'''
        cellstar = NeighbourMovementTracking()
        cellstar.parameters_tracking["avgCellDiameter"] = self.average_cell_diameter.value
        multiplier = float(NeighbourMovementTracking.parameters_cost_iteration["default_empty_cost"]) / NeighbourMovementTracking.parameters_cost_initial["default_empty_cost"]
        cellstar.parameters_cost_iteration["default_empty_cost"] = multiplier * self.drop_cost.value
        cellstar.parameters_cost_initial["default_empty_cost"] = self.drop_cost.value
        cellstar.parameters_tracking["iterations"] = self.iterations.value
        multiplier = float(NeighbourMovementTracking.parameters_cost_iteration["area_weight"]) / NeighbourMovementTracking.parameters_cost_initial["area_weight"]
        cellstar.parameters_cost_iteration["area_weight"] = multiplier * self.areaWeight.value
        cellstar.parameters_cost_initial["area_weight"] = self.areaWeight.value
        
        old_labels = self.get_saved_labels(workspace)
        if not old_labels is None:
            old_i,old_j = (centers_of_labels(old_labels)+.5).astype(int)
            old_count = len(old_i)
            
            i,j = (centers_of_labels(objects.segmented)+.5).astype(int)
            count = len(i)
            
            new_labels = objects.segmented
            # Matching is (expected to be) a injective function of old labels to new labels so we can inverse it.
            matching = cellstar.run_tracking(old_labels,new_labels)
            
            new_object_numbers = np.zeros(count,int)
            old_object_numbers = np.zeros(old_count,int)
            for old, new in matching:
                new_object_numbers[new-1] = old
                old_object_numbers[old-1] = new
            
            self.map_objects(workspace, 
                             old_object_numbers, 
                             new_object_numbers,
                             i,j)
        else:
            i,j = (centers_of_labels(objects.segmented)+.5).astype(int)
            count = len(i)
            self.map_objects(workspace, np.zeros((0,),int), 
                             np.zeros(count,int), i,j)
                             
                             
        self.set_saved_labels(workspace, objects.segmented)
        
    def map_objects(self, workspace, new_of_old, old_of_new, i,j):
        '''Record the mapping of old to new objects and vice-versa

        workspace - workspace for current image set
        new_to_old - an array of the new labels for every old label
        old_to_new - an array of the old labels for every new label
        score - a score that is higher for better mappings: we use this
                score to assign new label numbers so that the new objects
                that are "better" inherit the old objects' label number
        '''
        m = workspace.measurements
        #assert isinstance(m, cpmeas.Measurements)
        group_index = m.get_current_image_measurement(cpp.GROUP_INDEX)
        # [TODO OLD]
        #image_number = m.get_current_image_measurement(cpp.IMAGE_NUMBER) new version
        new_of_old = new_of_old.astype(int)
        old_of_new = old_of_new.astype(int)
        old_object_numbers = self.get_saved_object_numbers(workspace).astype(int)
        max_object_number = self.get_max_object_number(workspace)
        old_count = len(new_of_old)
        new_count = len(old_of_new)
        #
        # Record the new objects' parents
        #
        parents = old_of_new.copy()
        parents[parents != 0] =\
               old_object_numbers[(old_of_new[parents!=0]-1)].astype(parents.dtype)
        self.add_measurement(workspace, F_PARENT_OBJECT_NUMBER, old_of_new)
        parent_image_numbers = np.zeros(len(old_of_new))
        parent_image_numbers[parents != 0] = group_index - 1 #image_number - 1
        self.add_measurement(workspace, F_PARENT_IMAGE_NUMBER, 
                             parent_image_numbers)
        #
        # Assign object IDs to the new objects
        #
        mapping = np.zeros(new_count, int)
        if old_count > 0 and new_count > 0:
            mapping[old_of_new != 0] = \
                   old_object_numbers[old_of_new[old_of_new != 0] - 1]
            miss_count = np.sum(old_of_new == 0)
            lost_object_count = np.sum(new_of_old == 0)
        else:
            miss_count = new_count
            lost_object_count = old_count
        nunmapped = np.sum(mapping==0)
        new_max_object_number = max_object_number + nunmapped
        mapping[mapping == 0] = np.arange(max_object_number+1,
                                          new_max_object_number + 1)
        self.set_max_object_number(workspace, new_max_object_number)
        self.add_measurement(workspace, F_LABEL, mapping)
        self.set_saved_object_numbers(workspace, mapping)
        #
        # Compute distances and trajectories
        #
        diff_i = np.zeros(new_count)
        diff_j = np.zeros(new_count)
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
        age = np.ones(new_count, int)
        if np.any(has_old):
            old_age = self.get_saved_ages(workspace)
            age[has_old] = old_age[old_of_new[has_old]-1]+1
        self.add_measurement(workspace, F_LIFETIME, age)
        final_age = np.NaN*np.ones(new_count, float) # Initialize to NaN; will re-calc later
        self.add_measurement(workspace, F_FINAL_AGE, final_age)
        self.set_saved_ages(workspace, age)
        self.set_saved_object_numbers(workspace, mapping)
        #
        # Add image measurements
        #
        self.add_image_measurement(workspace, F_NEW_OBJECT_COUNT, 
                                   np.sum(parents==0))
        self.add_image_measurement(workspace, F_LOST_OBJECT_COUNT, 
                                   lost_object_count)
        #
        # Find parents with more than one child. These are the progenetors
        # for daughter cells.
        #
        if np.any(parents != 0):
            h = np.bincount(parents[parents != 0])
            split_count = np.sum(h > 1)
        else:
            split_count = 0
        self.add_image_measurement(workspace, F_SPLIT_COUNT, split_count)
        #
        # Find children with more than one parent. These are the merges
        #
        if np.any(new_of_old != 0):
            h = np.bincount(new_of_old[new_of_old != 0])
            merge_count = np.sum(h > 1)
        else:
            merge_count = 0
        self.add_image_measurement(workspace, F_MERGE_COUNT, merge_count)

