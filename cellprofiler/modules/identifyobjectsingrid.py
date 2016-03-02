'''<b>Identify Objects In Grid</b> identifies objects within each section of a grid 
that has been defined by the <b>DefineGrid</b> module.
<hr>
This module identifies objects that are contained within in a grid pattern, allowing
you to measure the objects using <b>Measure</b> modules. It requires you to have
defined a grid earlier in the pipeline, using the <b>DefineGrid</b> module.

For several of the automatic options, you will need to enter the names of previously
identified objects. Typically, this module is used to refine locations and/or
shapes of objects of interest that you roughly identified in a previous <b>Identify</b>
module. Within this module, objects are re-numbered according to the grid definitions
rather than their original numbering from the earlier <b>Identify</b> module.

If placing the objects within the grid is impossible for some reason (the
grid compartments are too close together to fit the proper sized circles,
for example) the grid will fail and processing will be canceled unless
you choose to re-use a grid from a previous successful image cycle.

<i>Special note on saving images:</i> You can use the settings in this module to
pass object outlines along to the <b>OverlayOutlines</b>module and then
save them with the <b>SaveImages</b> module. You can also pass along objects themselves
to the object processing module <b>ConvertToImage</b> and then save them with the
<b>SaveImages</b> module.

<h4>Available measurements</h4>
<b>Image measurements:</b>
<ul>
<li><i>Count:</i> The number of objects identified.</li>
</ul>
<b>Object measurements:</b>
<ul>
<li><i>Location_X, Location_Y:</i> The pixel (X,Y) coordinates of the center of
mass of the identified objects.</li>
<li><i>Number:</i> The numeric label assigned to each identified object
according to the arrangement order specified by the user.</li>
</ul>

<p>See also <b>DefineGrid</b>.
'''

import numpy as np
from centrosome.cpmorphology import centers_of_labels, relabel
from centrosome.outline import outline

import cellprofiler.cpgridinfo as cpg
import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
from cellprofiler.gui.help import HELP_ON_MEASURING_DISTANCES
from cellprofiler.gui.help import RETAINING_OUTLINES_HELP, NAMING_OUTLINES_HELP
from cellprofiler.modules.identify import add_object_count_measurements
from cellprofiler.modules.identify import add_object_location_measurements
from cellprofiler.modules.identify import get_object_measurement_columns

SHAPE_RECTANGLE = "Rectangle Forced Location"
SHAPE_CIRCLE_FORCED = "Circle Forced Location"
SHAPE_CIRCLE_NATURAL = "Circle Natural Location"
SHAPE_NATURAL = "Natural Shape and Location"

AM_AUTOMATIC = "Automatic"
AM_MANUAL = "Manual"

FAIL_NO = "No"
FAIL_ANY_PREVIOUS = "Any Previous"
FAIL_FIRST = "The First"

class IdentifyObjectsInGrid(cpm.CPModule):

    module_name = "IdentifyObjectsInGrid"
    variable_revision_number = 2
    category = "Object Processing"

    def create_settings(self):
        """Create your settings by subclassing this function

        create_settings is called at the end of initialization.
        """
        self.grid_name = cps.GridNameSubscriber(
            "Select the defined grid",cps.NONE,doc="""
            Select the name of a grid created by a previous <b>DefineGrid</b>
            module.""")

        self.output_objects_name = cps.ObjectNameProvider(
            "Name the objects to be identified","Wells",doc="""
            Enter the name of the grid objects identified by this module. These objects
            will be available for further measurement and processing in subsequent modules.""")

        self.shape_choice = cps.Choice(
            "Select object shapes and locations",
            [SHAPE_RECTANGLE, SHAPE_CIRCLE_FORCED, SHAPE_CIRCLE_NATURAL, SHAPE_NATURAL],doc="""
            Use this setting to choose the method to be used to determine the
            grid objects' shapes and locations:
            <ul>
            <li><i>%(SHAPE_RECTANGLE)s:</i> Each object will be created as a rectangle,
            completely occupying the entire grid compartment (rectangle). This option creates
            the rectangular objects based solely on the grid's specifications,
            not on any previously identified guiding objects.</li>
            <li><i>%(SHAPE_CIRCLE_FORCED)s:</i> Each object will be created as a circle, centered
            in the middle of each grid compartment. This option places
            the circular objects' locations based solely on the grid's specifications,
            not on any previously identified guiding objects. The radius of all circles in a grid
            will be constant for the entire grid in each image cycle, and can
            be determined automatically for each image cycle based on the average radius of previously
            identified guiding objects for that image cycle, or instead it can be specified
            as a single radius for all circles in all grids in the entire analysis run.</li>
            <li><i>%(SHAPE_CIRCLE_NATURAL)s:</i> Each object will be created as a circle,
            and each circle's location within its grid compartment will be determined
            based on the location of any previously identified guiding objects within
            that grid compartment. Thus, if a guiding object lies within a particular grid compartment,
            that object's center will be the center of the created circular object.
            If no guiding objects lie within a particular grid compartment, the circular object
            is placed within the center of that grid compartment.  If more than one
            guiding object lies within the grid compartment, they will be combined and the centroid of
            this combined object will be the location of the created circular object.
            Note that guiding objects whose centers are close to the grid edge are ignored.</li>
            <li><i>%(SHAPE_NATURAL)s:</i> Within each grid compartment, the object
            will be identified based on combining all of the parts of guiding objects, if any,
            that fall within the grid compartment.
            Note that guiding objects whose centers are close to the grid edge are ignored.
            If a guiding object does not exist within a grid compartment, an object consisting
            of one single pixel in the middle of the grid compartment will be created.</li>
            </ul>"""%globals())

        self.diameter_choice = cps.Choice(
            "Specify the circle diameter automatically?",
            [AM_AUTOMATIC, AM_MANUAL], doc="""
            <i>(Used only if Circle is selected as object shape)</i><br>
            There are two methods for selecting the circle diameter:
            <ul>
            <li><i>%(AM_AUTOMATIC)s:</i> Uses the average diameter of previously identified guiding
            objects as the diameter.</li>
            <li><i>%(AM_MANUAL)s:</i> Lets you specify the diameter directly, as a number.</li>
            </ul>"""%globals())

        self.diameter = cps.Integer(
            "Circle diameter", 20, minval=2, doc="""
            <i>(Used only if Circle is selected as object shape and diameter is
            specified manually)</i><br>
            Enter the diameter to be used for each grid circle, in pixels.
            %(HELP_ON_MEASURING_DISTANCES)s"""%globals())

        self.guiding_object_name = cps.ObjectNameSubscriber(
            "Select the guiding objects", cps.NONE,doc="""
            <i>(Used only if Circle is selected as object shape and diameter is specified
            automatically, or if Natural Location is selected as the object shape)</i><br>
            Select the names of previously identified objects that
            will be used to guide the shape and/or location of the objects created by this
            module, depending on the method chosen.""")

        self.wants_outlines = cps.Binary(
            "Retain outlines of the identified objects?", False,doc="""
            %(RETAINING_OUTLINES_HELP)s"""%globals())

        self.outlines_name = cps.OutlineNameProvider(
            "Name the outline image","GridOutlines",doc="""
            %(NAMING_OUTLINES_HELP)s"""%globals())

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
        object_count = gridding.rows * gridding.columns
        workspace.object_set.add_objects(objects,
                                         self.output_objects_name.value)
        add_object_location_measurements(workspace.measurements,
                                         self.output_objects_name.value,
                                         labels, object_count)
        add_object_count_measurements(workspace.measurements,
                                      self.output_objects_name.value,
                                      object_count)
        if self.wants_outlines:
            outlines = outline(labels!=0)
            outline_image = cpi.Image(outlines)
            workspace.image_set.add(self.outlines_name.value, outline_image)
        if self.show_window:
            workspace.display_data.gridding = gridding
            workspace.display_data.labels = labels

    def run_rectangle(self, workspace, gridding):
        '''Return a labels matrix composed of the grid rectangles'''
        return self.fill_grid(workspace, gridding)

    def fill_grid(self, workspace, gridding):
        '''Fill a labels matrix by labeling each rectangle in the grid'''
        assert isinstance(gridding, cpg.CPGridInfo)
        i, j = np.mgrid[0:gridding.image_height,
                        0:gridding.image_width]
        i_min = int(gridding.y_location_of_lowest_y_spot -
                    gridding.y_spacing / 2)
        j_min = int(gridding.x_location_of_lowest_x_spot -
                    gridding.x_spacing / 2)
        i = np.floor((i - i_min) / gridding.y_spacing).astype(int)
        j = np.floor((j - j_min) / gridding.x_spacing).astype(int)
        mask = ((i >= 0) & (j >= 0) &
                (i < gridding.spot_table.shape[0]) &
                (j < gridding.spot_table.shape[1]))
        labels = np.zeros((gridding.image_height,
                          gridding.image_width), int)
        labels[mask] = gridding.spot_table[i[mask], j[mask]]
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
        #
        # Remove any label with a bogus center (no guiding object)
        #
        labels[np.isnan(centers_i) | np.isnan(centers_j)] = 0
        #labels, count = relabel(labels)
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
        nmissing = np.max(gridding.spot_table) - len(centers_i)
        if nmissing > 0:
            centers_i = np.hstack((centers_i, [np.NaN] * nmissing))
            centers_j = np.hstack((centers_j, [np.NaN] * nmissing))
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
        #labels, count = relabel(labels)
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


    def display(self, workspace, figure):
        '''Display the resulting objects'''
        import matplotlib
        gridding = workspace.display_data.gridding
        labels = workspace.display_data.labels
        objects_name = self.output_objects_name.value
        figure.set_subplots((1, 1))
        figure.subplot_imshow_labels(0, 0, labels,
                                     title="Identified %s" % objects_name)
        axes = figure.subplot(0, 0)
        for xc, yc in ((gridding.horiz_lines_x, gridding.horiz_lines_y),
                       (gridding.vert_lines_x, gridding.vert_lines_y)):
            for i in range(xc.shape[1]):
                line = matplotlib.lines.Line2D(xc[:, i], yc[:, i],
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

        if (not from_matlab) and variable_revision_number == 1:
            # Change shape_choice names: Rectangle > Rectangle Forced Location, Natural Shape > Natural Shape and Location
            if setting_values[2] == "Rectangle":
                setting_values[2] = SHAPE_RECTANGLE
            elif setting_values[2] == "Natural Shape":
                setting_values[2] = SHAPE_NATURAL
            variable_revision_number = 2

        return setting_values, variable_revision_number, from_matlab

    def get_measurement_columns(self, pipeline):
        '''Column definitions for measurements made by IdentifyPrimaryObjects'''
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
