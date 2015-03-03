#!/usr/bin/env python
import cellprofiler.icons 
from cellprofiler.gui.help import PROTIP_RECOMEND_ICON, PROTIP_AVOID_ICON, TECH_NOTE_ICON
__doc__ = """<b>YeastSegmentation</b> identifies yeast (or other round) cells in an image.

<p>This module was prepared by Filip Mroz, Adam Kaczmarek and Szymon Stoma. Please reach us at <a href="http://let-your-data-speak.com/">Scopem, ETH</a> for inquires. The method uses star approach developed by Versari et al., in prep. For more details related to yeast segmentation in CellProfiler, please refer to <a href="http://www.cellprofiler.org/yeasttoolbox/">Yeast Toolbox</a>.
<hr>

<h4>What do I need as input?</h4>
To use this module, you will need to make sure that your input image has the following qualities:
<ul>
<li>The image should be grayscale.</li>
<li>The foreground (i.e, regions of interest) are lighter than the background.</li>
</ul>
If this is not the case, other modules can be used to pre-process the images to ensure they are in 
the proper form:
<ul>
<li>If the objects in your images are dark on a light background, you 
should invert the images using the Invert operation in the <b>ImageMath</b> module.</li>
<li>If you are working with color images, they must first be converted to
grayscale using the <b>ColorToGray</b> module.</li>
</ul>
<p>If you have images in which the foreground and background cannot be distinguished by intensity alone
(e.g, brightfield or DIC images), you can use the <a href="http://www.ilastik.org/">ilastik</a> package
bundled with CellProfiler to perform pixel-based classification (Windows only). You first train a classifier 
by identifying areas of images that fall into one of several classes, such as cell body, nucleus, 
background, etc. Then, the <b>ClassifyPixels</b> module takes the classifier and applies it to each image 
to identify areas that correspond to the trained classes. The result of <b>ClassifyPixels</b> is 
an image in which the region that falls into the class of interest is light on a dark background. Since 
this new image satisfies the constraints above, it can be used as input in <b>IdentifyPrimaryObjects</b>. 
See the <b>ClassifyPixels</b> module for more information.</p>

<h4>What do the settings mean?</h4>
See below for help on the individual settings. The following icons are used to call attention to
key items:
<ul>
<li><img src="memory:%(PROTIP_RECOMEND_ICON)s">&nbsp;Our recommendation or example use case
for which a particular setting is best used.</li>
<li><img src="memory:%(PROTIP_AVOID_ICON)s">&nbsp;Indicates a condition under which 
a particular setting may not work well.</li>
<li><img src="memory:%(TECH_NOTE_ICON)s">&nbsp;Technical note. Provides more
detailed information on the setting.</li>
</ul>

<h4>What do I get as output?</h4>
A set of primary objects are produced by this module, which can be used in downstream modules
for measurement purposes or other operations. 
See the section <a href="#Available_measurements">"Available measurements"</a> below for 
the measurements that are produced by this module.

Once the module has finished processing, the module display window 
will show the following panels:
<ul>
<li><i>Upper left:</i> The raw, original image.</li>
<li><i>Upper right:</i> The identified objects shown as a color
image where connected pixels that belong to the same object are assigned the same
color (<i>label image</i>). It is important to note that assigned colors are 
arbitrary; they are used simply to help you distingush the various objects. </li>
<li><i>Lower left:</i> The raw image overlaid with the colored outlines of the 
identified objects. Each object is assigned one of three (default) colors:
<ul>
<li>Green: Acceptable; passed all criteria</li>
<li>Magenta: Discarded based on size</li>
<li>Yellow: Discarded due to touching the border</li>
</ul>
If you need to change the color defaults, you can 
make adjustments in <i>File > Preferences</i>.</li>
<li><i>Lower right:</i> A table showing some of the settings selected by the user, as well as
those calculated by the module in order to produce the objects shown.</li>
</ul>

<a name="Available_measurements">
<h4>Available measurements</h4>
<b>Image measurements:</b>
<ul>
<li><i>Count:</i> The number of primary objects identified.</li>
<li><i>OriginalThreshold:</i> The global threshold for the image.</li>
<li><i>FinalThreshold:</i> For the global threshold methods, this value is the
same as <i>OriginalThreshold</i>. For the adaptive or per-object methods, this
value is the mean of the local thresholds.</li>
<li><i>WeightedVariance:</i> The sum of the log-transformed variances of the 
foreground and background pixels, weighted by the number of pixels in 
each distribution.</li>
<li><i>SumOfEntropies:</i> The sum of entropies computed from the foreground and
background distributions.</li>
</ul>

<b>Object measurements:</b>
<ul>
<li><i>Location_X, Location_Y:</i> The pixel (X,Y) coordinates of the primary 
object centroids. The centroid is calculated as the center of mass of the binary 
representation of the object.</li>
</ul>

<h3>Credits (coding)</h3>
Filip Mroz, Adam Kaczmarek, Szymon Stoma.

<h3>Credits (method)</h3>
Filip Mroz, Adam Kaczmarek, Szymon Stoma,
Artemis Llamosi,
Kirill Batmanov,
Cristian Versari,
Cedric Lhoussaine,
Gabor Csucs,
Simon F. Noerrelykke,
Gregory Batt,
Pawel Rychlikowski,
Pascal Hersen.

<h3>Publications</h3>
<p>

<h3>Technical notes</h3>
"""%globals()

# Module documentation variables:
__authors__="""Filip Mroz,
Adam Kaczmarek,
Szymon Stoma    
"""
__contact__=""
__license__="Cecill-C"
__date__="<Timestamp>"
__version__="0.1"
__docformat__= "restructuredtext en"
__revision__="$Id$"


#################################
#
# Imports from useful Python libraries
#
#################################
import random
from os.path import expanduser, exists
from os.path import join as pj
from os import makedirs

import numpy as np
import scipy as sp
from scipy.ndimage.filters import *
import scipy.ndimage.measurements as spmeasure

#################################
#
# Imports from CellProfiler
#
##################################
try:
    from cellprofiler.modules import identify as cpmi
    import cellprofiler.cpmodule as cpm
    from cellprofiler.modules.identifyobjectsmanually import IdentifyObjectsManually
    import cellprofiler.cpimage as cpi
    import cellprofiler.measurements as cpmeas
    import cellprofiler.settings as cps
    import cellprofiler.preferences as cpp
    import cellprofiler.cpmath as cpmath
    import cellprofiler.cpmath.outline
    import cellprofiler.objects
    from cellprofiler.gui.help import HELP_ON_MEASURING_DISTANCES
    import cellprofiler.preferences as pref

#################################
#
# Specific imports
#
##################################

    from contrib.cell_star.process.segmentation import Segmentation

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
F_BACKGROUND = "Background"
BKG_CURRENT = 'Actual image'
BKG_FIRST = 'First image'
BKG_FILE = 'File'

###################################
#
# The module class
#
###################################


class YeastCellSegmentation(cpmi.Identify):
    module_name = "IdentifyYeastCells"
    category = "Yeast Toolbox"
    variable_revision_number = 6
    current_workspace = ''
    
    def create_settings(self):
        self.input_image_name = cps.ImageNameSubscriber(
            "Select the input image",doc="""
            How do you call the images you want to use to identify objects?""")

        self.object_name = cps.ObjectNameProvider(
            "Name the primary objects to be identified",
            "YeastCells",doc="""
            How do you want to call the objects identified by this module?""")
            
        self.background_image_name = cps.ImageNameSubscriber(
            "Select the empty field image",doc="""
            How do you call the image you want to use as background 
            image (same image will be used for every image in the workflow)?
            """)

        self.background_elimination_strategy = cps.Choice(
            'Select the background calculation mode',
            [BKG_CURRENT, BKG_FIRST, BKG_FILE],doc = """
            You can choose from the following options:
            <ul>
            <li><i>loaded from file</i>: TODO</li>
            <li><i>computed from first image</i>: TODO</li>
            <li><i>computed from actual image</i>: TODO</li>
            </ul>""")
        
        self.average_cell_diameter = cps.Float(
            "Average cell diameter in pixels",
            30.0, minval=10, doc ='''\
            The average cell diameter is used to scale many algorithm parameters. 
            Please use e.g. ImageJ to measure the average cell size in pixels.
            '''
            )

        self.advanced_cell_filtering = cps.Binary(
            'Do you want to filter cells by area?', False, doc="""
            This parameter is using for final acceptance of cells.

            TODO:
            1. explain the difference from filtering using CP module.
            """)

        self.min_cell_area = cps.Integer(
            "Minimal area of accepted cell in pixels",
            900, minval=10, doc ='''\
            The minimum cell area is used while final filtering of cells. 
            Please use e.g. ImageJ to measure the average cell size in pixels.
            '''
            )

        self.max_cell_area = cps.Integer(
            "Maximum area of accepted cell in pixels",
            5*900, minval=10, doc ='''\
            The maximum cell area is used while final filtering of cells. 
            Please use e.g. ImageJ to measure the average cell size in pixels.
            '''
            )

        
        self.background_brighter_then_cell_inside = cps.Binary(
            'Is the area without cells (background) brighter then cell interiors?', True, doc="""
            Please check if the area inside of the cells is <b>darker</b> then the area without the cells (background).
            """
            )

        self.bright_field_image = cps.Binary(
            'Do you want to segment brightfield images?', True, doc="""
            Choose this option if you want to segment a brightfiled image.

            TODO:
            If you have DIC image, please remember to use a ... module before.
            """
            )

        self.advanced_parameters = cps.Binary(
            'Use advanced configuration parameters', False, doc="""
            Do you want to use advanced parameters to configure plugin? They allow for more flexibility,
            however you need to know what you are doing.
            """
            )

        self.segmentation_precision = cps.Integer(
            "Segmentation precision",
            9,minval=2,maxval=15,doc = '''\
            Describes how thouroughly we want to look for cells. Higher values should 
            make it easier to find smaller cells because the more parameters sets are searched. 
            The cost is higher runtime.
            '''
            )
            
        self.maximal_cell_overlap = cps.Float(
            "Maximal overlap of resulting cells",
            0.2,minval=0,maxval=1,doc='''\
            If higher value is used the closed (NOT CLEAR??) cells might be missed.
            In the case of the other extreme the resulting cells might be overlapping.
            '''
            )
        
        self.autoadaptation_steps = cps.Integer(
            "Number of steps in the autoadaptation procedure",
            9,minval=1,maxval=1000,doc = '''
            Describes how thouroughly we want to adapt the algorithm to current image sets. Higher values should 
            make it easier to correctly discover cells, however you will have to wait longer for the autoadaptation
            procedure to finish. Remember that you do it once for all images, and you can copy the values from other
            pipeline, if you have already found the parameters before. 
            '''
            )

        self.use_ground_truth_to_set_params = cps.DoSomething("","Autoadapt parameters",
        self.ground_truth_editor, doc="""
        Use this option to autoadapt parameters required for correct contour identification.

        TODO:
        1. Explain that parameters are rather non understandable/
        2. Explain what is the strategy.
        """)

        self.show_autoadapted_params = cps.Binary(
            'Do you want to see autoadapted parameters?', False, doc="""
            Use this option to discplay autoadapted parameters.

            TODO:
            1. explain the procedure.
            2. explain how to move it
            """)

        self.autoadapted_params = cps.Text(text="Autoadapted parameters: ", value="[[0.1, 0.0442, 304.45, 15.482, 189.40820000000002, 7.0], [300, 10, 0, 18, 10]]", doc="""
            Autoadapted parameters are pasted here from the procedure. These parameters are used to characterize cell borders. Edit them only if you know what you are doing.
            """)

        self.should_save_outlines = cps.Binary(
            'Retain outlines of the identified objects?', False, doc="""
            Use this method to automatically estimate advanced parameters required for correct recognition of cell borders.
            """)
        
        self.save_outlines = cps.OutlineNameProvider(
            'Name the outline image',"PrimaryOutlines", doc="""\
            <i>(Used only if outlines are to be saved)</i><br>
            You can use the outlines of the identified objects in modules downstream,
            by selecting them from any drop-down image list.""")

    def settings(self):
        return [self.input_image_name, 
                self.object_name,
                self.average_cell_diameter,
                self.segmentation_precision,
                self.maximal_cell_overlap,
                self.background_image_name,
                self.should_save_outlines,
                self.save_outlines,
                self.advanced_parameters,
                self.background_brighter_then_cell_inside,
                self.bright_field_image,
                self.min_cell_area,
                self.max_cell_area,
                self.advanced_cell_filtering, 
                self.background_elimination_strategy,
                self.show_autoadapted_params,
                self.autoadapted_params,
                self.autoadaptation_steps
                ]
    
    def visible_settings(self):
        list = [self.input_image_name, 
                self.object_name, 
                self.average_cell_diameter,
                ]

        list+=[self.bright_field_image,
                self.background_brighter_then_cell_inside,
                self.autoadaptation_steps,
                self.use_ground_truth_to_set_params,
                self.show_autoadapted_params]
        
        if self.show_autoadapted_params:
            list.append( self.autoadapted_params )

        list.append( self.advanced_parameters )
        
        #
        # Show the user the background only if self.provide_background is checked
        #
        if self.advanced_parameters:
            list.append( self.segmentation_precision )
            list.append( self.maximal_cell_overlap )
            list.append( self.advanced_cell_filtering )
            if self.advanced_cell_filtering:
                list.append( self.min_cell_area )
                list.append( self.max_cell_area )
            # Show the user the background only if self.provide_background is checked
            list.append( self.background_elimination_strategy ) #
            
            if self.background_elimination_strategy == BKG_FILE:
                list.append(self.background_image_name) 
            
        list.append(self.should_save_outlines)
        #
        # Show the user the scale only if self.should_save_outlines is checked
        #
        if self.should_save_outlines:
            list.append(self.save_outlines)
        
        return list
    
    def is_interactive(self):
        return False

    def prepare_group(self, workspace, grouping, image_numbers):
        '''Erase module information at the start of a run'''
        d = self.get_dictionary(workspace.image_set_list)
        d.clear()
        return True

    def get_ws_dictionary(self, workspace):
        return self.get_dictionary(workspace.image_set_list)

    def __get(self, field, workspace, default):
        if self.get_ws_dictionary(workspace).has_key(field):
            return self.get_ws_dictionary(workspace)[field]
        return default

    def __set(self, field, workspace, value):
        self.get_ws_dictionary(workspace)[field] = value
 
    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
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
        if variable_revision_number == 2:
            setting_values = setting_values + [False]
            variable_revision_number = 3
        if variable_revision_number == 3:
            setting_values = setting_values + [True]
            variable_revision_number = 4

        return setting_values, variable_revision_number, from_matlab

    def display(self, workspace, figure=None):
        
        # Important images should be shown always (the same that in case of IdentifyPrimaryObjects),
        # but if DEBUG is on then many more intermediate images can be presented for investigation.
        
        figure = workspace.create_or_find_figure(subplots=(2,1))
        
        figure.subplot_imshow_grayscale(
            0, 0,
            workspace.display_data.input_pixels,
            title = self.input_image_name.value)
        lead_subplot = figure.subplot(0,0)

        figure.subplot_imshow_grayscale(
            1,0 , 
            workspace.display_data.segmentation_pixels,
            title = "Segmentation",
            sharex = lead_subplot, sharey = lead_subplot)
            
        
    # 
    # Measuremets:
    # - objects count
    # - objects location
    
    def get_measurement_columns(self, pipeline):
        '''Column definitions for measurements made by IdentifyPrimAutomatic'''
        columns = cpmi.get_object_measurement_columns(self.object_name.value)
        return columns
    
    def get_categories(self,pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        result = self.get_object_categories(pipeline, object_name,
                                             {self.object_name.value: [] })
        return result
      
    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """

        result = self.get_object_measurements(pipeline, object_name, category,
                                               {self.object_name.value: [] })
        return result
    
    def run(self, workspace):
        ##clock import time
        ##clock start = time.clock()
        input_image_name = self.input_image_name.value
        self.current_workspace = workspace
        image_set = workspace.image_set

        input_image = image_set.get_image(input_image_name,must_be_grayscale = True)
        self.input_image_file_name = input_image.file_name
        input_pixels = input_image.pixel_data

        # Load previously computed background.
        if self.background_elimination_strategy == BKG_FILE:
            background_pixels = image_set.get_image(self.background_image_name.value, must_be_grayscale=True).pixel_data
        elif self.background_elimination_strategy == BKG_FIRST:
            background_pixels = self.__get(F_BACKGROUND, workspace, None)
        else:
            background_pixels = None

        # Invert images if required.
        if not self.background_brighter_then_cell_inside:
            input_pixels = 1 - input_pixels
            if self.background_elimination_strategy == BKG_FILE:
                background_pixels = 1 - background_pixels

        #
        # Preprocessing (only normalization)
        #
        normalized_image, background_pixels = self.preprocessing(input_pixels, background_pixels)

        #
        # Segmentation
        #
        objects, objects_qualities, background_pixels = self.segmentation(normalized_image, background_pixels)

        if self.__get(F_BACKGROUND, workspace, None) is None and self.background_elimination_strategy == BKG_FIRST:
            self.__set(F_BACKGROUND, workspace, background_pixels)
        
        #
        # Postprocessing
        #
        self.postprocessing(objects)
        workspace.object_set.add_objects(objects, self.object_name.value)

        # Make outlines
        outline_image = cellprofiler.cpmath.outline.outline(objects.segmented)
        if self.should_save_outlines.value:
            out_img = cpi.Image(outline_image.astype(bool),
                                parent_image = normalized_image)
            workspace.image_set.add(self.save_outlines.value, out_img)
            
        # Save measurements

                                           
        cpmi.add_object_location_measurements(workspace.measurements, 
                                      self.object_name.value,
                                      objects.segmented)
                                      
        cpmi.add_object_count_measurements(workspace.measurements,
                                           self.object_name.value, np.max(objects.segmented))

        ##clock end = time.clock()
        ##clock print "segmentation_plugin", end - start
        
    # 
    # Preprocessing of the input bright field image data.
    # Returns: normalized_image 
    #
    def preprocessing(self,input_pixels,background_pixels):
        def adam_normalization(image):
            width = image.shape[1]
            height = image.shape[0]
            image1d = np.array(list(image.reshape(-1)))
            image2d = np.zeros(image.shape)
            
            for y in xrange(height):
                for x in xrange(width):
                    image2d[y,x] = image1d[y*width + x]
            return image2d

        #if self.current_workspace.frame is not None:
        self.current_workspace.display_data.input_pixels = input_pixels
            
        
        if background_pixels != None:
            background_pixels_normalized = adam_normalization(background_pixels)
        else:
            background_pixels_normalized = None
            
        return adam_normalization(input_pixels), background_pixels_normalized

    #
    # Segmentation of the image into yeast cells.
    # Returns: yeast cells, yeast cells qualities, background
    #
    def segmentation(self, normalized_image, background_pixels):
        cellstar = Segmentation(self.segmentation_precision.value, self.average_cell_diameter.value)
        cellstar.parameters["segmentation"]["maxOverlap"] = self.maximal_cell_overlap.value
        if self.advanced_cell_filtering.value:
            areas_range = self.min_cell_area.value, self.max_cell_area.value
            cellstar.parameters["segmentation"]["minSize"] = areas_range[0]
            cellstar.parameters["segmentation"]["maxSize"] = areas_range[1]

        cellstar.decode_auto_params(self.autoadapted_params.value)

        if self.input_image_file_name is not None:
            dedicated_image_folder = pj(pref.get_default_output_directory(), self.input_image_file_name)
            if not exists(dedicated_image_folder):
                makedirs(dedicated_image_folder)
            if dedicated_image_folder is not None:
                cellstar.debug_output_image_path = dedicated_image_folder

        cellstar.set_frame(normalized_image)
        cellstar.set_background(background_pixels)
        segmented_image, snakes = cellstar.run_segmentation()
            
        objects = cellprofiler.objects.Objects()
        objects.segmented = segmented_image
        #print "segmented", segmented_image.shape
        objects.unedited_segmented = segmented_image
        objects.small_removed_segmented = np.zeros(normalized_image.shape)
        objects.parent_image = normalized_image
        
        #objects = cellprofiler.objects.Objects()
        #objects.segmented = labeled_image
        #objects.unedited_segmented = unedited_labels
        #objects.small_removed_segmented = small_removed_labels
        #objects.parent_image = image
        
        #if self.current_workspace.frame is not None:
        self.current_workspace.display_data.segmentation_pixels = objects.segmented

        return objects, [s.rank for s in snakes], cellstar.images.background

    def ground_truth_editor( self ):
        '''Display a UI for GT editing'''
        from cellprofiler.gui.editobjectsdlg import EditObjectsDialog
        from wx import OK
        import wx
        #title = "%s #%d, image cycle #%d: " % (self.module_name,
        #                                     self.module_num,
        #                                     image_set_number)
        
        ### opening file dialog
        dlg = wx.FileDialog(None,
                            message = "Open an image file",
                            wildcard = "Image file (*.tif,*.tiff,*.jpg,*.jpeg,*.png,*.gif,*.bmp)|*.tif;*.tiff;*.jpg;*.jpeg;*.png;*.gif;*.bmp|*.* (all files)|*.*",
                            style = wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            from cellprofiler.modules.loadimages import LoadImagesImageProvider
            from cellprofiler.gui.cpfigure import CPFigureFrame
            from bioformats import load_image
            image = load_image( dlg.Path) #lip.provide_image(None).pixel_data
        else:
            return

        ### opening GT editor
        title = "Please mark few representative cells to allow for autoadapting parameters. \n"
        title += " \n"
        title += 'Press "F" to being freehand drawing.\n'
        title += "Click Help for full instructions."
        self.pixel_data = image #np.zeros((1000, 1000),np.float)
        # TODO think what to do if the user chooses new image (and we load old cells)
        #if not hasattr(self, 'labels'):
        labels = [np.zeros(self.pixel_data.shape[:2], np.uint32)]  #[np.array(self.pixel_data[i,:], int) for i in range(self.pixel_data.shape[0])]

        with EditObjectsDialog(
            self.pixel_data, labels, False, title) as dialog_box:
            result = dialog_box.ShowModal()
            if result != OK:
                return None
            labels = dialog_box.labels[0]

        # check if the user provided GT
        # TODO check for con. comp. and e.g. let it go if more then 3 cells were added
        if not labels.any():
            dlg = wx.MessageDialog(None, "Please correctly select at least one cell. Otherwise, parameters can not be autoadapted!", "Warning!", wx.OK | wx.ICON_WARNING)
            dlg.ShowModal()
            dlg.Destroy()
            return

        ### fitting params 
        # reading GT from dialog_box.labels[0] and image from self.pixel
        progressMax = self.autoadaptation_steps.value
        dialog = wx.ProgressDialog("Fitting parameters..", "Iterations remaining", progressMax,
        style=wx.PD_CAN_ABORT | wx.PD_CAN_SKIP | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME )
        keepGoing = True
        count = 0
        while (keepGoing and count < progressMax):# and dialog.WasCancelled():
            count = count + 1
            # here put one it. of fitting instead
            wx.Sleep(1)
            
            # here update params. in the GUI
            keepGoing, skip = dialog.Update(count)
        dialog.Update(progressMax)
        #dialog.Close()
        #dialog.Destroy()
    
    #
    # Postprocess objects found by CellStar
    #
    def postprocessing(self, objects):
        pass

