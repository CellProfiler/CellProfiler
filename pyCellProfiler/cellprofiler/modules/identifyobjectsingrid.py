'''<b>Identify Objects In Grid</b> identifies objects within each section of a grid that has been defined by
the <b>DefineGrid</b> module
<hr>
This module identifies objects that are in a grid pattern, allowing 
you to measure the objects using <b>Measure</b> modules. It requires you to have 
created a grid earlier in the pipeline, using the <b>DefineGrid</b> module.

For several of the automatic options, you will need to enter the names of previously identified objects. Typically,
this module is used to refine locations and/or shapes of objects of interest that 
you roughly identified in a previous <b>Identify</b> module. Within this module, objects are re-numbered according to the grid
definitions rather than their original numbering from the earlier  
<b>Identify</b> module. For the <i>Natural Shape</i> option, if an object does not 
exist within a grid compartment, an object consisting of one single pixel 
in the middle of the grid square will be created. Also, for the <i>Natural 
Shape</i> option, if a grid compartment contains two partial objects, they 
will be combined together into a single object.

If placing the objects within the grid is impossible for some reason (the
grid compartments are too close together to fit the proper sized circles,
for example) the grid will fail and processing will be canceled unless
you choose to re-use any previous grid or the first grid in the  
image cycle.

<i>Special note on saving images:</i> You can use the settings in this module to
pass object outlines along to the <b>OverlayOutlines</b>module and then
save them with the </b>SaveImages</b> module. Objects themselves can be passed along
to the object processing module <b>ConvertToImage</b> and then saved with the
<b>SaveImages</b> module.

<p>See also <b>DefineGrid</b>.
'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__="$Revision$"

import numpy as np

import cellprofiler.cpgridinfo as cpg
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
import cellprofiler.objects as cpo
from cellprofiler.modules.identify import add_object_count_measurements
from cellprofiler.modules.identify import add_object_location_measurements
from cellprofiler.modules.identify import get_object_measurement_columns
import cellprofiler.gui.cpfigure
from cellprofiler.cpmath.outline import outline
from cellprofiler.cpmath.cpmorphology import centers_of_labels

SHAPE_RECTANGLE = "Rectangle"
SHAPE_CIRCLE_FORCED = "Circle Forced Location"
SHAPE_CIRCLE_NATURAL = "Circle Natural Location"
SHAPE_NATURAL = "Natural Shape"

AM_AUTOMATIC = "Automatic"
AM_MANUAL = "Manual"

FAIL_NO = "No"
FAIL_ANY_PREVIOUS = "Any Previous"
FAIL_FIRST = "The First"

class IdentifyObjectsInGrid(cpm.CPModule):
    
    module_name = "IdentifyObjectsInGrid"
    variable_revision_number = 1
    category = "Object Processing"
    
    def create_settings(self):
        """Create your settings by subclassing this function
        
        create_settings is called at the end of initialization.
        """
        self.grid_name = cps.GridNameSubscriber("Select the defined grid","None",doc="""
            The name of a grid created by a previous <b>DefineGrid</b>
            module.""")
        
        self.output_objects_name = cps.ObjectNameProvider(
            "Name the identified objects","Wells",
            doc="""Enter the name you want to use for the grid objects created by this module. These objects will be available in
            subsequent modules.""")
        
        self.shape_choice = cps.Choice(
            "Select object shape",[SHAPE_RECTANGLE, SHAPE_CIRCLE_FORCED,
                             SHAPE_CIRCLE_NATURAL, SHAPE_NATURAL],
            doc="""Use this setting to choose the grid object shape and the
            algorithm used to create that shape:
            <ul>
            <li><i>Rectangle</i>: Each object occupies the entire grid
            rectangle.</li>
            <li><i>Circle Forced Location</i>: The object is a circle, centered
            in the middle of each grid. You will have an opportunity to
            specify the circle radius.</li>
            <li><i>Circle Natural Location</i>: The object is a circle. The
            algorithm takes all of the guiding objects that are within the grid
            except for ones whose centers are close to the grid edge, combines
            the parts that fall within the grid and finds the centroid of
            this aggregation. The circle's center is set to that centroid.</li>
            <li><i>Natural Location</i>: The object is an aggregation of
            all of the parts of guiding objects that fall within the grid.
            The algorithm filters out guiding objects that are close to the
            edge of the grid.</li>
            </ul>""")
        
        self.diameter_choice = cps.Choice(
            "Specify the circle diameter automatically?",
            [AM_AUTOMATIC, AM_MANUAL],
            doc="""<i>(Used if Circle is selected as object shape)</i><br>
            The automatic method uses the average diameter of guiding
            objects as the diameter. The manual method lets you specify the
            diameter directly.""")
        
        self.diameter = cps.Integer(
            "Circle diameter", 20, minval=2,
            doc="""<i>(Used if Circle is selected as object shape and diameter is 
            specified manually)</i><br>
            Enter the diameter to be used for each grid circle""")
        
        self.guiding_object_name = cps.ObjectNameSubscriber(
            "Select the guiding objects", "None",
            doc="""<i>(Used if Circle is selected as object shape and diameter is 
            specified automatically, or if Natural Location is selected as object 
            shape)</i><br>
            The names of previously identified objects that
            will be used to guide placement of the objects created by this
            module. These objects may used to automatically calculate the
            diameter of the circles, placement of the circles
            within the grid or the shape of the objects created by this
            module, depending on the module's settings.""")
        
        self.wants_outlines = cps.Binary(
            "Save outlines of the identified objects?", False,
            doc="""The module can create a binary image of the outlines
            of the objects it creates. You can then use <b>OverlayOutlines</b>
            to overlay the outlines on an image or use <b>SaveImages</b>.
            to save them""")
        
        self.outlines_name = cps.OutlineNameProvider(
            "Name the outline image","GridOutlines",
            doc="""<i>(Used if outlines are to be saved)</i><br>
            This setting names the outlines of the output objects.
            You can use this name to refer to the outlines in the
            <b>OverlayOutlines</b> and <b>SaveImages</b> modules.""")

    def settings(self):
        """Return the settings to be loaded or saved to/from the pipeline
        
        These are the settings (from cellprofiler.settings) that are
        either read from the strings in the pipeline or written out
        to the pipeline. The settings should appear in a consistent
        order so they can be matched to the strings in the pipeline.
        """
        return [self.grid_name, self.output_objects_name, self.shape_choice,
                self.diameter_choice, self.diameter, 
                self.guiding_object_name,
                self.wants_outlines, self.outlines_name]

    def visible_settings(self):
        """Return the settings that the user sees"""
        result = [self.grid_name, self.output_objects_name, self.shape_choice]
        if self.shape_choice in [SHAPE_CIRCLE_FORCED, SHAPE_CIRCLE_NATURAL]:
            result += [self.diameter_choice]
            if self.diameter_choice == AM_MANUAL:
                result += [self.diameter]
        if self.wants_guiding_objects():
            result+= [self.guiding_object_name]
        result+=[self.wants_outlines]
        if self.wants_outlines:
            result+=[self.outlines_name]
        return result
    
    def wants_guiding_objects(self):
        '''Return TRUE if the settings require valid guiding objects'''
        return ((self.shape_choice == SHAPE_CIRCLE_FORCED and
                 self.diameter_choice == AM_AUTOMATIC) or
                (self.shape_choice in (SHAPE_CIRCLE_NATURAL, SHAPE_NATURAL)))
    
    def run(self, workspace):
        '''Find the outlines on the current image set
        
        workspace    - The workspace contains
            pipeline     - instance of cpp for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - the parent frame to whatever frame is created. None means don't draw.
        '''
        gridding = workspace.get_grid(self.grid_name.value)
        if self.shape_choice == SHAPE_RECTANGLE:
            labels = self.run_rectangle(workspace, gridding)
        elif self.shape_choice == SHAPE_CIRCLE_FORCED:
            labels = self.run_forced_circle(workspace, gridding)
        elif self.shape_choice == SHAPE_CIRCLE_NATURAL:
            labels = self.run_natural_circle(workspace, gridding)
        elif self.shape_choice == SHAPE_NATURAL:
            labels = self.run_natural(workspace, gridding)
        objects = cpo.Objects()
        objects.segmented = labels
        workspace.object_set.add_objects(objects, 
                                         self.output_objects_name.value)
        add_object_location_measurements(workspace.measurements,
                                         self.output_objects_name.value,
                                         labels)
        add_object_count_measurements(workspace.measurements,
                                      self.output_objects_name.value,
                                      len(np.unique(labels[labels!=0])))
        if self.wants_outlines:
            outlines = outline(labels!=0)
            outline_image = cpi.Image(outlines)
            workspace.image_set.add(self.outlines_name.value, outline_image)
            
    def run_rectangle(self, workspace, gridding):
        '''Return a labels matrix composed of the grid rectangles'''
        return self.fill_grid(workspace, gridding)
    
    def fill_grid(self, workspace, gridding):
        '''Fill a labels matrix by labeling each rectangle in the grid'''
        assert isinstance(gridding, cpg.CPGridInfo)
        i_min = int(gridding.y_location_of_lowest_y_spot -
                    gridding.y_spacing / 2)
        j_min = int(gridding.x_location_of_lowest_x_spot -
                    gridding.x_spacing / 2)
        labels=np.zeros((i_min + gridding.total_height,
                         j_min + gridding.total_width), int)
        i,j = np.mgrid[0:gridding.total_height,0:gridding.total_width]
        i /= gridding.y_spacing
        j /= gridding.x_spacing
        labels[i_min:,j_min:] = gridding.spot_table[i.astype(int),j.astype(int)]
        return labels
    
    def run_forced_circle(self, workspace, gridding):
        '''Return a labels matrix composed of circles centered in the grids'''
        i,j = np.mgrid[0:gridding.rows,0:gridding.columns]
        
        return self.run_circle(workspace, gridding,
                               gridding.y_locations[i],
                               gridding.x_locations[j])
    
    def run_circle(self, workspace,gridding, spot_center_i, spot_center_j):
        '''Return a labels matrix compose of circles centered on the x,y locs
        
        workspace - workspace for the run
        gridding - an instance of CPGridInfo giving the details of the grid
        spot_center_i, spot_center_j - the locations of the grid centers. 
                   This should have one coordinate per grid cell.
        '''

        assert isinstance(gridding,cpg.CPGridInfo)
        radius = self.get_radius(workspace, gridding)
        labels = self.fill_grid(workspace,gridding)
        labels = self.fit_labels_to_guiding_objects(workspace, labels)
        i_min = int(gridding.y_location_of_lowest_y_spot -
                    gridding.y_spacing / 2)
        j_min = int(gridding.x_location_of_lowest_x_spot -
                    gridding.x_spacing / 2)
        spot_center_i_flat = np.zeros(gridding.spot_table.max()+1)
        spot_center_j_flat = np.zeros(gridding.spot_table.max()+1)
        spot_center_i_flat[gridding.spot_table.flatten()] = spot_center_i.flatten()
        spot_center_j_flat[gridding.spot_table.flatten()] = spot_center_j.flatten()
        
        centers_i = spot_center_i_flat[labels]
        centers_j = spot_center_j_flat[labels]
        i,j = np.mgrid[0:labels.shape[0],0:labels.shape[1]]
        #
        # Add .5 to measure from the center of the pixel
        #
        mask = (i-centers_i)**2 + (j-centers_j)**2 <= (radius+.5)**2
        labels[~mask] = 0
        return labels
    
    def run_natural_circle(self, workspace, gridding):
        '''Return a labels matrix composed of circles found from objects'''
        #
        # Find the centroid of any guide label in a grid
        #
        guide_label = self.filtered_labels(workspace, gridding)
        labels = self.fill_grid(workspace,gridding)
        labels[guide_label[0:labels.shape[0],0:labels.shape[1]] == 0] = 0
        centers_i, centers_j = centers_of_labels(labels)
        #
        # Broadcast these using the spot table
        #
        centers_i = centers_i[gridding.spot_table-1]
        centers_j = centers_j[gridding.spot_table-1]
        return self.run_circle(workspace, gridding, centers_i, centers_j)
    
    def run_natural(self, workspace, gridding):
        '''Return a labels matrix made by masking the grid labels with
        the filtered guide labels'''
        guide_label = self.filtered_labels(workspace, gridding)
        labels = self.fill_grid(workspace, gridding)
        labels = self.fit_labels_to_guiding_objects(workspace, labels)
        labels[guide_label == 0] = 0
        return labels
    
    def fit_labels_to_guiding_objects(self, workspace, labels):
        '''Make the labels matrix the same size as the guiding objects matrix
        
        The gridding is typically smaller in extent than the image it's
        based on. This function enlarges the labels matrix to match the
        dimensions of the guiding objects matrix if appropriate.
        '''
        if not self.wants_guiding_objects():
            # No guiding objects? No-op
            return labels
        
        guide_label = self.get_guide_labels(workspace)
        if any(guide_label.shape[i] > labels.shape[i] for i in range(2)):
            result = np.zeros([max(guide_label.shape[i], labels.shape[i])
                               for i in range(2)], int)
            result [0:labels.shape[0],0:labels.shape[1]] = labels
            return result
        return labels
        
    def get_radius(self, workspace, gridding):
        '''Get the radius for circles'''
        if self.diameter_choice == AM_MANUAL:
            return self.diameter.value / 2
        labels = self.filtered_labels(workspace, gridding)
        areas = np.bincount(labels[labels != 0])
        if len(areas) == 0:
            raise RuntimeError("Failed to calculate average radius: no grid objects found in %s"%
                               self.guiding_object_name.value)
        median_area = np.median(areas[areas!=0])
        return max(1, np.sqrt(median_area/ np.pi) )
    
    def filtered_labels(self, workspace, gridding):
        '''Filter labels by proximity to edges of grid'''
        #
        # A label might slightly graze a grid other than its own or
        # a label might be something small in a corner of the grid.
        # This function filters out those parts of the guide labels matrix
        #
        assert isinstance(gridding, cpg.CPGridInfo)
        guide_labels = self.get_guide_labels(workspace)
        labels = self.fill_grid(workspace,gridding)

        centers = np.zeros((2,np.max(guide_labels)+1))
        centers[:,1:] = centers_of_labels(guide_labels)
        bad_centers = ((~ np.isfinite(centers[0,:])) |
                       (~ np.isfinite(centers[1,:])) |
                       (centers[0,:] >= labels.shape[0]) |
                       (centers[1,:] >= labels.shape[1]))
        centers = np.round(centers).astype(int)
        masked_labels = labels.copy()
        x_border = int(np.ceil(gridding.x_spacing /10))
        y_border = int(np.ceil(gridding.y_spacing / 10))
        #
        # erase anything that's not like what's next to it
        #
        ymask = labels[y_border:,:] != labels[:-y_border,:]
        masked_labels[y_border:,:][ymask] = 0
        masked_labels[:-y_border,:][ymask] = 0
        xmask = labels[:,x_border:] != labels[:,:-x_border]
        masked_labels[:,x_border:][xmask] = 0
        masked_labels[:,:-x_border][xmask] = 0
        #
        # Find out the grid that each center falls into. If a center falls
        # into the border region, it will get a grid number of 0 and be
        # erased. The guide objects may fall below or to the right of the
        # grid or there may be gaps in numbering, so we set the center label
        # of bad centers to 0.
        #
        centers[:,bad_centers] = 0
        lcenters = masked_labels[centers[0,:],centers[1,:]]
        lcenters[bad_centers] = 0
        #
        # Use the guide labels to look up the corresponding center for
        # each guide object pixel. Mask out guide labels that don't match
        # centers.
        #
        mask = np.zeros(guide_labels.shape, bool)
        ii_labels = np.index_exp[0:labels.shape[0],0:labels.shape[1]]
        mask[ii_labels] = lcenters[guide_labels[ii_labels]] != labels
        mask[guide_labels == 0] = True
        mask[lcenters[guide_labels] == 0] = True
        filtered_guide_labels = guide_labels.copy()
        filtered_guide_labels[mask] = 0
        return filtered_guide_labels

    def get_guide_labels(self, workspace):
        '''Return the guide labels matrix for this module'''
        guide_labels = workspace.object_set.get_objects(self.guiding_object_name.value)
        guide_labels = guide_labels.segmented
        return guide_labels
        

    def is_interactive(self):
        return False

    def display(self, workspace):
        '''Display the resulting objects'''
        import matplotlib
        gridding = workspace.get_grid(self.grid_name.value)
        assert isinstance(gridding, cpg.CPGridInfo)
        objects_name = self.output_objects_name.value
        o = workspace.object_set.get_objects(objects_name)
        labels = o.segmented
        figure = workspace.create_or_find_figure(subplots=(1,1))
        figure.subplot_imshow_labels(0,0,labels,
                                     title="Identified %s"%objects_name)
        axes = figure.subplot(0,0)
        assert isinstance(axes,matplotlib.axes.Axes)
        for xc, yc in ((gridding.horiz_lines_x, gridding.horiz_lines_y),
                       (gridding.vert_lines_x, gridding.vert_lines_y)):
            for i in range(xc.shape[1]):
                line = matplotlib.lines.Line2D(xc[:,i],yc[:,i],
                                               color="red")
                axes.add_line(line)
                
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
        if from_matlab and variable_revision_number == 2:
            grid_name, new_object_name, shape, old_object_name,\
            diameter, save_outlines, failed_grid_choice = setting_values
            if diameter == AM_AUTOMATIC:
                diameter = "40"
                diameter_choice = AM_AUTOMATIC
            else:
                diameter_choice = AM_MANUAL
            wants_outlines = (cps.NO if save_outlines == cps.DO_NOT_USE
                              else cps.YES)
            setting_values = [grid_name, new_object_name, shape,
                              diameter_choice, diameter, old_object_name,
                              wants_outlines, save_outlines]
            variable_revision_number = 1
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab

    def get_measurement_columns(self, pipeline):
        '''Column definitions for measurements made by IdentifyPrimAutomatic'''
        return get_object_measurement_columns(self.output_objects_name.value)
             
    def get_categories(self,pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        if object_name == 'Image':
            return ['Count']
        elif object_name == self.output_objects_name.value:
            return ['Location','Number']
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
        elif object_name == self.output_objects_name.value and category == 'Number':
            return ['Object_Number']
        return []
