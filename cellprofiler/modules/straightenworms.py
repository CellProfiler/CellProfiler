'''<b>StraightenWorms</b> Straightens untangled worms
<hr>

<b>StraightenWorms</b> uses the objects produced by <b>UntangleWorms</b>
to create images and objects of straight worms from the angles and control
points as computed by <b>UntangleWorms</b>. The resulting images can then
be uniformly analyzed to find features that correlate with position in
an ideal representation of the worm, such as the head or gut.

<b>StraightenWorms</b> works by calculating a transform on the image that
translates points in the image to points on the ideal worm. <b>UntangleWorms</b>
idealizes a worm as a series of control points that define the worm's shape
and length. The training set contains measurements of the width of an ideal
worm at each control point. Together, these can be used to reconstruct the
worm's shape and correlate between the worm's location and points on
the body of an ideal worm.

<b>StraightenWorms</b> produces two kinds of outputs: objects representing
the straight worms and images representing the intensity values of a source
image mapped onto the straight worms. The objects and images can then be
used to compute measurements using any of the object measurement modules,
for instance, <b>MeasureTexture</b>.

The idea of straightening worms was inspired by the paper, <i>Straightening
Caenorhabditis elegans images</i>, Hanchuan Peng, Fuhui Long, Xiao Liu,
Stuart Kim, Eugene Myers, Bioinformatics Vol 24 # 2, 2008, pp 234-242.
'''

__version__="$Revision: 10717 %"

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.cpmath.cpmorphology as morph
import cellprofiler.preferences as cpprefs
from cellprofiler.preferences import IO_FOLDER_CHOICE_HELP_TEXT

from untangleworms import C_WORM, F_CONTROL_POINT_X, F_CONTROL_POINT_Y
from untangleworms import F_LENGTH, ATTR_WORM_MEASUREMENTS
from untangleworms import read_params

from identify import get_object_measurement_columns
from identify import add_object_count_measurements
from identify import add_object_location_measurements
from identify import C_COUNT, C_LOCATION, FTR_CENTER_X, FTR_CENTER_Y
from identify import C_NUMBER, FTR_OBJECT_NUMBER

'''The index of the image count setting (# of images to process)'''
IDX_IMAGE_COUNT = 5

class StraightenWorms(cpm.CPModule):
    
    variable_revision_number = 1
    category = "Object Processing"
    module_name = "StraightenWorms"
    
    def create_settings(self):
        '''Create the settings for the module'''
        self.objects_name = cps.ObjectNameSubscriber(
            'Select the input untangled worm objects', 'OverlappingWorms',
            required_attributes = { ATTR_WORM_MEASUREMENTS:True},
            doc = """This is the name of the objects produced by the
            <b>UntangleWorms</b> module. <b>StraightenWorms</b> can use
            either the overlapping or non-overlapping objects as input. It
            will use the control point measurements associated with the objects
            to reconstruct the straight worms.""")
        
        self.straightened_objects_name = cps.ObjectNameProvider(
            "Name the output straightened worm objects", "StraightenedWorms",
            doc = """This is the name that will be given to the straightened
            worm objects. These objects can then be used in a subsequent
            measurement module""")
        
        self.width = cps.Integer(
            "Worm width", 20, minval = 3,
            doc = """This setting determines the width of the image of each
            worm. The width should be set to at least the maximum width of
            any untangled worm, but can be set to be larger to include the
            worm's background in the straightened image.""")
        
        self.training_set_directory = cps.DirectoryPath(
            "Training set file location", support_urls = True,
            allow_metadata = False,
            doc = """Select the folder containing the training set to be loaded.
            %(IO_FOLDER_CHOICE_HELP_TEXT)s
            <p>An additional option is the following:
            <ul>
            <li><i>URL</i>: Use the path part of a URL. For instance, your
            training set might be hosted at 
            <i>http://university.edu/~johndoe/TrainingSet.mat</i>
            To access this file, you would choose <i>URL</i> and enter
            <i>https://svn.broadinstitute.org/CellProfiler/trunk/ExampleImages/ExampleSBSImages</i>
            as the path location.</li>
            </ul></p>"""%globals())
        def get_directory_fn():
            '''Get the directory for the CSV file name'''
            return self.training_set_directory.get_absolute_path()
        def set_directory_fn(path):
            dir_choice, custom_path = self.training_set_directory.get_parts_from_path(path)
            self.training_set_directory.join_parts(dir_choice, custom_path)
            
        self.training_set_file_name = cps.FilenameText(
            "Training set file name", "TrainingSet.mat",
            doc = "This is the name of the training set file.",
            get_directory_fn = get_directory_fn,
            set_directory_fn = set_directory_fn,
            browse_msg = "Choose training set",
            exts = [("Worm training set (*.xml)", "*.xml"),
                    ("All files (*.*)", "*.*")])
        
        self.images = []
        
        self.image_count = cps.HiddenCount(self.images, "Image count")
        
        self.add_image(False)
        
        self.add_image_button = cps.DoSomething(
            "", "Add another image", self.add_image,
            doc = """Press this button to add another image to be straightened""")
        
    def add_image(self, can_delete = True):
        '''Add an image to the list of images to be straightened'''
        
        group = cps.SettingsGroup()
        group.append("divider", cps.Divider())
        group.append("image_name", cps.ImageNameSubscriber(
            'Select an input image to straighten', 'None',
            doc = '''This is the name of an image that will be straightened
            similarly to the worm. The straightened image and objects can
            then be used in subsequent modules such as
            <b>MeasureObjectIntensity</b>'''))
        group.append("straightened_image_name", cps.ImageNameProvider(
            'Name the output straightened image', 'StraightenedImage',
            doc = '''This is the name that will be given to the image
            of the straightened worms.'''))
        if can_delete:
            group.append("remover", cps.RemoveSettingButton(
                "", "Remove above image", self.images, group))
        self.images.append(group)
        
    def settings(self):
        '''Return the settings, in the order they appear in the pipeline'''
        result = ([ self.objects_name, self.straightened_objects_name,
                    self.width, self.training_set_directory, 
                    self.training_set_file_name, self.image_count] + 
                  sum([ group.pipeline_settings() for group in self.images], []))
        return result
    
    def visible_settings(self):
        '''Return the settings as displayed in the module view'''
        result = ([ self.objects_name, self.straightened_objects_name,
                    self.width, self.training_set_directory,
                    self.training_set_file_name ] +
                  sum([ group.visible_settings() for group in self.images], []) +
                  [ self.add_image_button ])
        return result
    
    def prepare_settings(self, setting_values):
        nimages = int(setting_values[IDX_IMAGE_COUNT])
        del self.images[1:]
        for i in range(1,nimages):
            self.add_image()
            
    def is_interactive(self):
        return False
    
    def run(self, workspace):
        '''Process one image set'''
        objects_name = self.objects_name.value
        if not hasattr(self, "training_params"):
            self.training_params = {}
        params = read_params(self.training_set_directory,
                             self.training_set_file_name,
                             self.training_params)
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        #
        # Sort the features by control point number:
        # Worm_ControlPointX_2 < Worm_ControlPointX_10
        #
        features = m.get_feature_names(objects_name)
        cpx = [ f for f in features
                if f.startswith("_".join((C_WORM, F_CONTROL_POINT_X)))]
        cpy = [ f for f in features
                if f.startswith("_".join((C_WORM, F_CONTROL_POINT_Y)))]
        ncontrolpoints = len(cpx)
        
        def sort_fn(a,b):
            '''Sort by control point number'''
            acp = int(a.split("_")[-1])
            bcp = int(b.split("_")[-1])
            return cmp(acp, bcp)
        
        cpx.sort(sort_fn)
        cpy.sort(sort_fn)
        control_points = np.array([
            [m.get_current_measurement(objects_name, f) for f in cp]
            for cp in (cpy, cpx)])
        m_length = "_".join((C_WORM, F_LENGTH))
        lengths = np.ceil(m.get_current_measurement(objects_name, m_length))
        
        nworms = len(lengths)
        half_width = self.width.value / 2
        width = 2*half_width + 1
        if nworms == 0:
            shape = (2 * half_width + 1, width)
        else:
            shape = (int(np.max(lengths)) + 2*half_width + 1, 
                     nworms * width)
        labels = np.zeros(shape, int)
        #
        # ix and jx are the coordinates of the straightened pixel in the
        # original space.
        #
        ix = np.zeros(shape)
        jx = np.zeros(shape)
        #
        # Handle each of the worm splines separately
        #
        for i in range(nworms):
            ii = control_points[0, :, i]
            jj = control_points[1, :, i]
            
            si = interp1d(np.linspace(0, lengths[i], ncontrolpoints), ii)
            sj = interp1d(np.linspace(0, lengths[i], ncontrolpoints), jj)
            sw = interp1d(np.linspace(0, lengths[i], ncontrolpoints),
                          params.radii_from_training)
            #
            # The coordinates of "length" points along the worm
            #
            ci = si(np.arange(0, int(lengths[i])+1))
            cj = sj(np.arange(0, int(lengths[i])+1))
            #
            # Find the normals at each point by taking the derivative,
            # and twisting by 90 degrees. 
            #
            di = ci[1:] - ci[:-1]
            di = np.hstack([[di[0]], di])
            dj = cj[1:] - cj[:-1]
            dj = np.hstack([[dj[0]], dj])
            ni = -dj / np.sqrt(di**2 + dj**2)
            nj = di / np.sqrt(di**2 + dj**2)
            #
            # The widths at each point
            #
            w = sw(np.arange(0, int(lengths[i])+1))
            #
            # Extend the worm out from the head and tail by the width
            #
            ci = np.hstack([np.arange(-half_width,0) * nj[0] + ci[0],
                            ci,
                            np.arange(1, half_width + 1) * nj[-1] + ci[-1]])
            cj = np.hstack([np.arange(-half_width,0) * (-ni[0]) + cj[0],
                            cj,
                            np.arange(1, half_width + 1) * (-ni[-1]) + cj[-1]])
            ni = np.hstack([[ni[0]] * half_width, ni, [ni[-1]] * half_width])
            nj = np.hstack([[nj[0]] * half_width, nj, [nj[-1]] * half_width])
            iii, jjj = np.mgrid[0:len(ci), -half_width : (half_width+1)]
            #
            # Create a mapping of i an j in straightened space to
            # the coordinates in real space
            #
            jslice = slice(width * i,width * (i+1))
            ix[0:len(ci), jslice] = ci[iii] + ni[iii] * jjj
            jx[0:len(ci), jslice] = cj[iii] + nj[iii] * jjj
            islice = slice(half_width, int(lengths[i] + half_width + 1))
            mask = np.abs(jjj[islice,:]) <= w[:,np.newaxis]
            labels[islice, jslice][mask] = i+1
        if workspace.frame is not None:
            workspace.display_data.image_pairs = []
        #
        # Now create one straightened image for each input image
        #
        image_set = workspace.image_set
        assert(isinstance(image_set, cpi.ImageSet))
        for group in self.images:
            image_name = group.image_name.value
            straightened_image_name = group.straightened_image_name.value
            image = image_set.get_image(image_name)
            if image.pixel_data.ndim == 2:
                straightened_pixel_data = map_coordinates(
                    image.pixel_data, [ix, jx])
            else:
                straightened_pixel_data = np.zeros(
                    (ix.shape[0], ix.shape[1], image.pixel_data.shape[2]))
                for d in range(image.pixel_data.shape[2]):
                    straightened_pixel_data[:,:,d] = map_coordinates(
                        image.pixel_data[:,:,d], [ix, jx])
            straightened_mask = map_coordinates(image.mask, [ix, jx]) > .5
            straightened_image = cpi.Image(straightened_pixel_data,
                                           straightened_mask,
                                           parent_image = image)
            image_set.add(straightened_image_name, straightened_image)
            if workspace.frame is not None:
                workspace.display_data.image_pairs.append(
                    ((image.pixel_data, image_name),
                     (straightened_pixel_data, straightened_image_name)))
        #
        # Finally, we need to make the objects
        #
        straightened_objects_name = self.straightened_objects_name.value
        object_set = workspace.object_set
        assert isinstance(object_set, cpo.ObjectSet)
        straightened_objects = cpo.Objects()
        straightened_objects.segmented = labels
        object_set.add_objects(straightened_objects, straightened_objects_name)
        add_object_count_measurements(m, straightened_objects_name, nworms)
        add_object_location_measurements(m, straightened_objects_name,
                                         labels, nworms)
        
    def display(self, workspace):
        '''Display the results of the worm straightening'''
        image_pairs = workspace.display_data.image_pairs
        figure = workspace.create_or_find_figure(subplots=(2,len(image_pairs)))
        src_axis = None
        for i, ((src_pix, src_name), (dest_pix, dest_name)) in enumerate(image_pairs):
            if src_pix.ndim == 2:
                imshow = figure.subplot_imshow_grayscale
            else:
                imshow = figure.subplot_imshow_color
            axis = imshow(0, i, src_pix, title = src_name,
                          sharex = src_axis, sharey = src_axis)
            if src_axis is None:
                src_axis = axis
            if dest_pix.ndim == 2:
                imshow = figure.subplot_imshow_grayscale
            else:
                imshow = figure.subplot_imshow_color
            imshow(1, i, dest_pix, title = dest_name)
        
    def get_measurement_columns(self, pipeline):
        '''Return columns that define the measurements produced by this module'''
        return get_object_measurement_columns(self.straightened_objects_name.value)
    
    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return [ C_COUNT ]
        elif object_name == self.straightened_objects_name:
            return [ C_LOCATION, C_NUMBER]
        return []
    
    def get_measurements(self, pipeline, object_name, category):
        if object_name == cpmeas.IMAGE and category == C_COUNT:
            return [ self.straightened_objects_name.value]
        elif object_name == self.straightened_objects_name.value:
            if category == C_LOCATION:
                return [ FTR_CENTER_X, FTR_CENTER_Y ]
            elif category == C_NUMBER:
                return [ FTR_OBJECT_NUMBER ]
        return []
    
        