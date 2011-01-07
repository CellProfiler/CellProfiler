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
from scipy.ndimage import map_coordinates, extrema
from scipy.ndimage import mean as nd_mean
from scipy.ndimage import standard_deviation as nd_standard_deviation

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

FTR_MEAN_INTENSITY = "MeanIntensity"
FTR_STD_INTENSITY = "StdIntensity"

FLIP_NONE = "Do not align"
FLIP_TOP = "Top brightest"
FLIP_BOTTOM = "Bottom brightest"

'''The index of the image count setting (# of images to process)'''
IDX_IMAGE_COUNT_V1 = 5
IDX_IMAGE_COUNT_V2 = 5
IDX_IMAGE_COUNT = 5

FIXED_SETTINGS_COUNT_V1 = 6
VARIABLE_SETTINGS_COUNT_V1 = 2

class StraightenWorms(cpm.CPModule):
    
    variable_revision_number = 2
    category = "Object Processing"
    module_name = "StraightenWorms"
    
    def create_settings(self):
        '''Create the settings for the module'''
        self.images = []

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
        
        self.wants_measurements = cps.Binary(
            "Measure intensity distribution?", True,
            doc = """<b>StraightenWorms</b> can divide a worm into sections
            and measure the intensities of each section in each of the
            straightened images. These measurements can help classify
            phenotypes if the staining pattern across the segments differs
            between phenotypes.""")
        self.number_of_segments = cps.Integer(
            "Number of segments", 4, 2,
            doc = """(<i>Only displayed if intensities are measured</i>)<br>
            This setting controls the number of segments measured. The module
            will make the following measurements:
            <table><tr><th>Measurement</th><th>Description</th></tr>
            <tr><th>%(C_WORM)s_%(FTR_MEAN_INTENSITY)s_<i>image</i>_<i>segment</i></th>
            <td>The mean intensity of <i>image</i> in the segment with
            segment number <i>segment</i>, excluding pixels shared by two
            crossing worms.</td></tr>
            <tr><th>%(C_WORM)s_%(FTR_STD_INTENSITY)s_<i>image</i>_<i>segment</i></th>
            <th>The standard deviation of <i>image</i> in the segment with
            segment number <i>segment</i>, excluding pixels shared by two
            crossing worms.</td></tr>
            </table>""" % globals())
        self.flip_worms = cps.Choice(
            "Align worms?", [FLIP_NONE, FLIP_TOP, FLIP_BOTTOM],
            doc = """(<i>Only displayed if intensities are measured</i>)<br>
            <b>StraightenWorms</b> can align worms so that the brightest
            half of the worm (the half with the highest mean intensity) is
            at the top of the image or at the bottom of the image. This
            can be used to align all worms similarly if some feature,
            such as the larynx, is stained and is always at the same end
            of the worm. Choose <i>%(FLIP_TOP)s</i> if the brightest part of the
            worm should be at the top of the image, <i>%(FLIP_BOTTOM)s</i> if the
            brightest part of the worm should be at the bottom or
            <i>%(FLIP_NONE)s</i> if the worm should not be aligned.""")
        
        def image_choices_fn(pipeline):
            '''Return the image choices for the alignment image'''
            return [ group.image_name.value
                     for group in self.images ]
        
        self.flip_image = cps.Choice(
            "Alignment image", [ "None" ], choices_fn = image_choices_fn,
            doc = """(<i>Only displayed if aligning worms</i>)<br>
            This is the image whose intensity will be used to align the worms.
            You must use one of the straightened images below.""")
        
        
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
                    self.training_set_file_name, self.image_count,
                    self.wants_measurements, self.number_of_segments,
                    self.flip_worms, self.flip_image] + 
                  sum([ group.pipeline_settings() for group in self.images], []))
        return result
    
    def visible_settings(self):
        '''Return the settings as displayed in the module view'''
        result = [ self.objects_name, self.straightened_objects_name,
                   self.width, self.training_set_directory,
                   self.training_set_file_name, self.wants_measurements]
        if self.wants_measurements:
            result += [ self.number_of_segments, self.flip_worms]
            if self.flip_worms in (FLIP_BOTTOM, FLIP_TOP):
                result += [ self.flip_image ]
        result += sum([ group.visible_settings() for group in self.images], []) 
        result += [ self.add_image_button ]
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
        object_set = workspace.object_set
        assert isinstance(object_set, cpo.ObjectSet)

        image_set = workspace.image_set
        assert(isinstance(image_set, cpi.ImageSet))

        objects_name = self.objects_name.value
        orig_objects = object_set.get_objects(objects_name)
        assert isinstance(orig_objects, cpo.Objects)
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
        # This is a list of tuples - first element in the tuples is
        # a labels matrix, second is a list of indexes in the matrix.
        # We need this for overlapping worms.
        #
        orig_labels_and_indexes = orig_objects.get_labels()
        #
        # Handle each of the worm splines separately
        #
        for i in range(nworms):
            orig_labels = [x for x,y in orig_labels_and_indexes
                           if i+1 in y]
            if len(orig_labels) == 0:
                continue
            orig_labels = orig_labels[0]
            
            ii = control_points[0, :, i]
            jj = control_points[1, :, i]
            
            si = interp1d(np.linspace(0, lengths[i], ncontrolpoints), ii)
            sj = interp1d(np.linspace(0, lengths[i], ncontrolpoints), jj)
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
            islice =slice(0,len(ci))
            jslice = slice(width * i,width * (i+1))
            ix[islice, jslice] = ci[iii] + ni[iii] * jjj
            jx[islice, jslice] = cj[iii] + nj[iii] * jjj
            #
            # We may need to flip the worm
            #
            if self.flip_worms != FLIP_NONE:
                ixs = ix[islice,jslice]
                jxs = jx[islice,jslice]
                image_name = self.flip_image.value
                image = image_set.get_image(image_name, must_be_grayscale = True)
                simage = map_coordinates(image.pixel_data, [ixs, jxs])
                halfway = int(len(ci)) / 2
                smask = map_coordinates(orig_labels == i+1, [ixs, jxs])
                if image.has_mask:
                    smask *= map_coordinates(image.mask, [ixs, jxs])
                simage *= smask
                #
                # Compute the mean intensity of the top and bottom halves
                # of the worm.
                #
                area_top = np.sum(smask[:halfway,:])
                area_bottom = np.sum(smask[halfway:,:])
                top_intensity = np.sum(simage[:halfway,:]) / area_top
                bottom_intensity = np.sum(simage[halfway:,:]) / area_bottom
                if ((top_intensity > bottom_intensity) !=
                    (self.flip_worms == FLIP_TOP)):
                    # Flip worm if it doesn't match user expectations
                    iii = len(ci) - iii - 1
                    jjj = - jjj
                    ix[islice, jslice] = ci[iii] + ni[iii] * jjj
                    jx[islice, jslice] = cj[iii] + nj[iii] * jjj
            mask = map_coordinates((orig_labels == i+1).astype(np.float32), 
                                   [ix[islice, jslice], jx[islice,jslice]]) > .5
            labels[islice, jslice][mask] = i+1
        if workspace.frame is not None:
            workspace.display_data.image_pairs = []
        #
        # Now create one straightened image for each input image
        #
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
        # Measure the worms if appropriate
        #
        if self.wants_measurements:
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            object_name = self.straightened_objects_name.value
            nbins = self.number_of_segments.value
            if nworms == 0:
                for ftr in (FTR_MEAN_INTENSITY, FTR_STD_INTENSITY):
                    for group in self.images:
                        for b in range(nbins):
                            image_name = group.straightened_image_name.value
                            measurement = "_".join(
                                (C_WORM, ftr, image_name, str(b+1)))
                            m.add_measurement(object_name, measurement,
                                              np.zeros((0)))
            else:
                #
                # Find the minimum and maximum i coordinate of each worm
                #
                i,j = np.mgrid[0:labels.shape[0], 0:labels.shape[1]]
                min_i, max_i, _, _ = extrema(i, labels, orig_objects.indices)
                min_i = np.hstack(([0], min_i))
                max_i = np.hstack(([1], max_i)) + 1
                bin_lengths = (max_i - min_i).astype(float) / float(nbins)
                #
                # Find the bin for each pixel in the straightened image
                #
                mbin = ((i - min_i[labels]).astype(float) / bin_lengths[labels]).astype(int)
                #
                # Multiplex the label and bin number together
                # 
                mbin += labels * nbins
                mbin[labels == 0] = 0
                for group in self.images:
                    image_name = group.straightened_image_name.value
                    straightened_image = image_set.get_image(image_name).pixel_data
                    indices = np.arange(nbins, nbins * (nworms + 1))
                    bin_means = nd_mean(straightened_image, mbin, indices)
                    bin_stds = nd_standard_deviation(straightened_image, mbin, indices)
                    bin_means.shape = (nworms, nbins)
                    bin_stds.shape = (nworms, nbins)
                    for b in range(nbins):
                        for values, ftr in (
                            (bin_means, FTR_MEAN_INTENSITY),
                            (bin_stds, FTR_STD_INTENSITY)):
                            measurement = "_".join((C_WORM, ftr, image_name, str(b+1)))
                            m.add_measurement(object_name, measurement,
                                              values[:,b])
        # Finally, we need to make the objects
        #
        straightened_objects_name = self.straightened_objects_name.value
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
        result = get_object_measurement_columns(self.straightened_objects_name.value)
        if self.wants_measurements:
            nsegments = self.number_of_segments.value
            result += sum([
                sum([ [ ( self.straightened_objects_name.value, 
                          "_".join((C_WORM, ftr, 
                                    group.straightened_image_name.value, 
                                    str(segment))), 
                         cpmeas.COLTYPE_FLOAT)
                        for ftr in (FTR_MEAN_INTENSITY, FTR_STD_INTENSITY)]
                      for segment in range(1, nsegments + 1)], [])
                for group in self.images], [])
        return result
    
    def get_categories(self, pipeline, object_name):
        result = []
        if object_name == cpmeas.IMAGE:
            result += [ C_COUNT ]
        elif object_name == self.straightened_objects_name:
            result += [ C_LOCATION, C_NUMBER]
            if self.wants_measurements:
                result += [ C_WORM ]
        return result
    
    def get_measurements(self, pipeline, object_name, category):
        if object_name == cpmeas.IMAGE and category == C_COUNT:
            return [ self.straightened_objects_name.value]
        elif object_name == self.straightened_objects_name.value:
            if category == C_LOCATION:
                return [ FTR_CENTER_X, FTR_CENTER_Y ]
            elif category == C_NUMBER:
                return [ FTR_OBJECT_NUMBER ]
            elif category == C_WORM:
                return [ FTR_MEAN_INTENSITY, FTR_STD_INTENSITY ]
        return []
    
    def get_measurement_images(self, pipeline, object_name, category, measurement):
        if (object_name == self.straightened_objects_name and
            category == C_WORM and
            measurement in (FTR_MEAN_INTENSITY, FTR_STD_INTENSITY)):
            return [group.straightened_image_name.value
                    for group in self.images]
        return []
    
    def get_measurement_scales(self, pipeline, object_name, category,
                               measurement, image_name):
        if image_name in self.get_measurement_images(
            pipeline, object_name, category, measurement):
            return [str(x) for x in range(1, self.number_of_segments.value + 1)]
        return []
    
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        '''Modify the settings to match the current version
        
        This method takes the settings from a previous revision of
        StraightenWorms and modifies them so that they match
        the settings that would be output by the current version.
        
        setting_values - setting value strings, possibly output by prev version
        
        variable_revision_number - revision of version of StraightenWorms that
        output the settings
        
        module_name, from_matlab - not used, see CPModule for use elsewhere.

        Overriding modules should return a tuple of setting_values,
        variable_revision_number and True if upgraded to CP 2.0, otherwise
        they should leave things as-is so that the caller can report
        an error.
        '''
        
        if variable_revision_number == 1:
            #
            # Added worm measurement and flipping
            #
            setting_values = (
                setting_values[:FIXED_SETTINGS_COUNT_V1] +
                [ cps.NO, "4", cps.NO, "None" ] +
                setting_values[FIXED_SETTINGS_COUNT_V1:])
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab
    
    def prepare_to_create_batch(self, pipeline, image_set_list, fn_alter_path):
        '''Prepare to create a batch file
        
        This function is called when CellProfiler is about to create a
        file for batch processing. It will pickle the image set list's
        "legacy_fields" dictionary. This callback lets a module prepare for
        saving.
        
        pipeline - the pipeline to be saved
        image_set_list - the image set list to be saved
        fn_alter_path - this is a function that takes a pathname on the local
                        host and returns a pathname on the remote host. It
                        handles issues such as replacing backslashes and
                        mapping mountpoints. It should be called for every
                        pathname stored in the settings or legacy fields.
        '''
        self.training_set_directory.alter_for_create_batch_files(fn_alter_path)
