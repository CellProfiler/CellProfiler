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

<b>StraightenWorms</b> produces objects representing
the straight worms and images representing the intensity values of a source
image mapped onto the straight worms. The objects and images can then be
used to compute measurements using any of the object measurement modules,
for instance, <b>MeasureTexture</b>.

The module can be configured to make intensity measurements on parts of the
worm, dividing the worm up into pieces of equal width and/or height. 
Measurements are made longitudally in stripes from head to tail and transversely
in segments across the width of the worm. Longitudinal stripes are numbered
from left to right and transverse segments are numbered from top to bottom.
The module will divide the worm into a checkerboard of sections if configured
to measure more than one longitudinal stripe and transverse segment. These
are numbered by longitudinal stripe number, then transverse segment number. For
instance, "Worm_MeanIntensity_GFP_L2of3_T1of4", is a measurement of the
mean GFP intensity of the center stripe (second of 3 stripes) of the topmost band
(first of four bands). Measurements of longitudinal stripes are designated as 
"T1of1" indicating that the whole worm is one transverse segment. Likewise 
measurements of transverse segments are designated as "L1of1" indicating that
there is only one longitudinal stripe. Both mean intensity and standard
deviation of intensity are measured per worm sub-area.

The idea of straightening worms was inspired by the paper, <i>Straightening
Caenorhabditis elegans images</i>, Hanchuan Peng, Fuhui Long, Xiao Liu,
Stuart Kim, Eugene Myers, Bioinformatics Vol 24 # 2, 2008, pp 234-242.
'''

__version__="$Revision: 10717 %"

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate.fitpack import bisplrep, dblint
from scipy.ndimage import map_coordinates, extrema
from scipy.ndimage import mean as nd_mean
from scipy.ndimage import standard_deviation as nd_standard_deviation

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.cpmath.cpmorphology as morph
import cellprofiler.cpmath.index as INDEX
import cellprofiler.preferences as cpprefs
from cellprofiler.preferences import IO_FOLDER_CHOICE_HELP_TEXT
from cellprofiler.utilities import product

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

'''The horizontal scale label - T = Transverse, a transverse strip'''
SCALE_HORIZONTAL = "T"

'''The vertical scale label - L = Longitudinal, a longitudinal strip'''
SCALE_VERTICAL = "L"

FLIP_NONE = "Do not align"
FLIP_TOP = "Top brightest"
FLIP_BOTTOM = "Bottom brightest"

'''The index of the image count setting (# of images to process)'''
IDX_IMAGE_COUNT_V1 = 5
IDX_IMAGE_COUNT_V2 = 5
IDX_IMAGE_COUNT_V3 = 5
IDX_IMAGE_COUNT = 5
IDX_FLIP_WORMS_V2 = 8

FIXED_SETTINGS_COUNT_V1 = 6
VARIABLE_SETTINGS_COUNT_V1 = 2
FIXED_SETTINGS_COUNT_V2 = 10
VARIABLE_SETTINGS_COUNT_V2 = 2
FIXED_SETTINGS_COUNT_V3 = 11
VARIABLE_SETTINGS_COUNT_V3 = 2

class StraightenWorms(cpm.CPModule):
    
    variable_revision_number = 3
    category = ["Object Processing", "Worm Toolbox"]
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
            "Number of transverse segments", 4, 1,
            doc = """(<i>Only displayed if intensities are measured</i>)<br>
            This setting controls the number of segments measured, dividing
            the worm longitudally into transverse segments starting at the head 
            and ending at the tail.
            These measurements might be used to identify a phenotype in which
            a stain is localized longitudally, for instance, in the head.
            
            Set the number of vertical segments to 1 to only measure intensity
            in the horizontal direction.""")

        self.number_of_stripes = cps.Integer(
            "Number of longitudinal stripes", 3, 1,
            doc = """(<i>Only displayed if intensities are measured</i>)<br>
            This setting controls the number of stripes measured, dividing
            the worm transversely into areas that run longitudally. These
            measurements might be used to identify a phenotype in which a
            stain is localized transversely, for instance in the gut of the
            worm.
            
            Set the number of horizontal stripes to 1 to only measure intensity
            in the vertical direction.""")
        
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
            <i>%(FLIP_NONE)s</i> if the worm should not be aligned."""%globals())
        
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
                    self.number_of_stripes,
                    self.flip_worms, self.flip_image] + 
                  sum([ group.pipeline_settings() for group in self.images], []))
        return result
    
    def visible_settings(self):
        '''Return the settings as displayed in the module view'''
        result = [ self.objects_name, self.straightened_objects_name,
                   self.width, self.training_set_directory,
                   self.training_set_file_name, self.wants_measurements]
        if self.wants_measurements:
            result += [ self.number_of_segments, self.number_of_stripes,
                        self.flip_worms]
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
            self.measure_worms(workspace, labels, nworms, width)
        #
        # Record the objects
        #
        self.make_objects(workspace, labels, nworms)
            
    def measure_worms(self, workspace, labels, nworms, width):
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        object_name = self.straightened_objects_name.value
        nbins_vertical = self.number_of_segments.value
        nbins_horizontal = self.number_of_stripes.value
        if not hasattr(self, "training_params"):
            self.training_params = {}
        params = read_params(self.training_set_directory,
                             self.training_set_file_name,
                             self.training_params)
        if nworms == 0:
            # # # # # # # # # # # # # # # # # # # # # #
            #
            # Record measurements if no worms
            #
            # # # # # # # # # # # # # # # # # # # # # #
            for ftr in (FTR_MEAN_INTENSITY, FTR_STD_INTENSITY):
                for group in self.images:
                    image_name = group.straightened_image_name.value
                    if nbins_vertical > 1:
                        for b in range(nbins_vertical):
                            measurement = "_".join(
                                (C_WORM, ftr, image_name, 
                                 self.get_scale_name(None, b)))
                            m.add_measurement(object_name, measurement,
                                              np.zeros((0)))
                    if nbins_horizontal > 1:
                        for b in range(nbins_horizontal):
                            measurement = "_".join(
                                (C_WORM, ftr, image_name, 
                                 self.get_scale_name(b, None)))
                            m.add_measurement(object_name, measurement,
                                              np.zeros((0)))
                        if nbins_vertical > 1:
                            for v in range(nbins_vertical):
                                for h in range(nbins_horizontal):
                                    measurement = "_".join(
                                        (C_WORM, ftr, image_name, 
                                         self.get_scale_name(h, v)))
                                    m.add_measurement(object_name, measurement,
                                                      np.zeros((0)))
                                    
        else:
            #
            # Find the minimum and maximum i coordinate of each worm
            #
            object_set = workspace.object_set
            assert isinstance(object_set, cpo.ObjectSet)
            orig_objects = object_set.get_objects(self.objects_name.value)
    
            i,j = np.mgrid[0:labels.shape[0], 0:labels.shape[1]]
            min_i, max_i, _, _ = extrema(i, labels, orig_objects.indices)
            min_i = np.hstack(([0], min_i))
            max_i = np.hstack(([labels.shape[0]], max_i)) + 1
            heights = max_i - min_i
            
            # # # # # # # # # # # # # # # # #
            #
            # Create up to 3 spaces which represent the gridding
            # of the worm and create a coordinate mapping into
            # this gridding for each straightened worm
            #
            # # # # # # # # # # # # # # # # #
            griddings = []
            if nbins_vertical > 1:
                scales = np.array([self.get_scale_name(None, b)
                                   for b in range(nbins_vertical)])
                scales.shape = (nbins_vertical, 1)
                griddings += [(nbins_vertical, 1, scales)]
            if nbins_horizontal > 1:
                scales = np.array([self.get_scale_name(b, None)
                                   for b in range(nbins_horizontal)])
                scales.shape = (1, nbins_horizontal)
                griddings += [(1, nbins_horizontal, scales)]
                if nbins_vertical > 1:
                    scales = np.array([
                        [self.get_scale_name(h,v) for h in range(nbins_horizontal)]
                        for v in range(nbins_vertical)])
                    griddings += [(nbins_vertical, nbins_horizontal, scales)]
            
            for i_dim, j_dim, scales in griddings:
                # # # # # # # # # # # # # # # # # # # # # #
                #
                # Start out mapping every point to a 1x1 space
                #
                # # # # # # # # # # # # # # # # # # # # # #
                labels1 = labels.copy()
                i,j = np.mgrid[0:labels.shape[0], 0:labels.shape[1]]
                i_frac = (i - min_i[labels]).astype(float) / heights[labels]
                i_frac_end = i_frac + 1.0 / heights[labels].astype(float)
                i_radius_frac = (i - min_i[labels]).astype(float) / (heights[labels] - 1)
                labels1[(i_frac >= 1) | (i_frac_end <= 0)] = 0
                # # # # # # # # # # # # # # # # # # # # # #
                #
                # Map the horizontal onto the grid.
                #
                # # # # # # # # # # # # # # # # # # # # # #
                radii = np.array(params.radii_from_training)
                #
                # For each pixel in the image, find the center of its worm
                # in the j direction (the width)
                #
                j_center = int(width / 2) + width * (labels-1)
                #
                # Find which segment (from the training set) per pixel in
                # a fractional form
                #
                i_index = i_radius_frac * (len(radii) - 1)
                #
                # Interpolate
                #
                i_index_frac = i_index - np.floor(i_index)
                i_index_frac[i_index >= len(radii) - 1] = 1
                i_index = np.minimum(i_index.astype(int), len(radii) - 2)
                r = np.ceil((radii[i_index] * (1 - i_index_frac) + 
                             radii[i_index+1] * i_index_frac))
                #
                # Map the worm width into the space 0-1
                #
                j_frac = (j - j_center + r) / (r * 2 + 1)
                j_frac_end = j_frac + 1.0 / (r * 2 + 1)
                labels1[(j_frac >= 1) | (j_frac_end <= 0)] = 0
                #
                # Map the worms onto the gridding.
                #
                i_mapping = np.maximum(i_frac * i_dim, 0)
                i_mapping_end = np.minimum(i_frac_end * i_dim, i_dim)
                j_mapping = np.maximum(j_frac * j_dim, 0)
                j_mapping_end = np.minimum(j_frac_end  * j_dim, j_dim)
                i_mapping = i_mapping[labels1 > 0]
                i_mapping_end = i_mapping_end[labels1 > 0]
                j_mapping = j_mapping[labels1 > 0]
                j_mapping_end = j_mapping_end[labels1 > 0]
                labels_1d = labels1[labels1 > 0]
                i = i[labels1 > 0]
                j = j[labels1 > 0]
                
                #
                # There are easy cases and hard cases. The easy cases are
                # when a pixel in the input space wholly falls in the
                # output space.
                #
                easy = ((i_mapping.astype(int) == i_mapping_end.astype(int)) &
                        (j_mapping.astype(int) == j_mapping_end.astype(int)))
                
                i_src = i[easy]
                j_src = j[easy]
                i_dest = i_mapping[easy].astype(int)
                j_dest = j_mapping[easy].astype(int)
                weight = np.ones(i_src.shape)
                labels_src = labels_1d[easy]
                #
                # The hard cases start in one pixel in the binning space,
                # possibly continue through one or more intermediate pixels
                # in horribly degenerate cases and end in a final
                # partial pixel.
                #
                # More horribly, a pixel in the straightened space
                # might span two or more in the binning space in the I
                # direction, the J direction or both.
                #
                if not np.all(easy):
                    i = i[~ easy]
                    j = j[~ easy]
                    i_mapping = i_mapping[~ easy]
                    j_mapping = j_mapping[~ easy]
                    i_mapping_end = i_mapping_end[~ easy]
                    j_mapping_end = j_mapping_end[~ easy]
                    labels_1d = labels_1d[~ easy]
                    #
                    # A pixel in the straightened space can be wholly within
                    # a pixel in the bin space, it can straddle two pixels
                    # or straddle two and span one or more. It can do different
                    # things in the I and J direction.
                    #
                    # --- The number of pixels wholly spanned ---
                    #
                    i_span = np.maximum(np.floor(i_mapping_end) - np.ceil(i_mapping), 0)
                    j_span = np.maximum(np.floor(j_mapping_end) - np.ceil(j_mapping), 0)
                    #
                    # --- The fraction of a pixel covered by the lower straddle
                    #
                    i_low_straddle = i_mapping.astype(int) + 1 - i_mapping
                    j_low_straddle = j_mapping.astype(int) + 1 - j_mapping
                    #
                    # Segments that start at exact pixel boundaries and span
                    # whole pixels have low fractions that are 1. The span
                    # length needs to have these subtracted from it.
                    #
                    i_span[i_low_straddle == 1] -= 1
                    j_span[j_low_straddle == 1] -= 1
                    #
                    # --- the fraction covered by the upper straddle
                    #
                    i_high_straddle = i_mapping_end - i_mapping_end.astype(int)
                    j_high_straddle = j_mapping_end - j_mapping_end.astype(int)
                    #
                    # --- the total distance across the binning space
                    #
                    i_total = i_low_straddle + i_span + i_high_straddle
                    j_total = j_low_straddle + j_span + j_high_straddle
                    #
                    # --- The fraction in the lower straddle
                    #
                    i_low_frac = i_low_straddle / i_total
                    j_low_frac = j_low_straddle / j_total
                    #
                    # --- The fraction in the upper straddle
                    #
                    i_high_frac = i_high_straddle / i_total
                    j_high_frac = j_high_straddle / j_total
                    #
                    # later on, the high fraction will overwrite the low fraction
                    # for i and j hitting on a single pixel in the bin space
                    #
                    i_high_frac[(i_mapping.astype(int) == i_mapping_end.astype(int))] = 1
                    j_high_frac[(j_mapping.astype(int) == j_mapping_end.astype(int))] = 1
                    #
                    # --- The fraction in spans
                    #
                    i_span_frac = i_span / i_total
                    j_span_frac = j_span / j_total
                    #
                    # --- The number of bins touched by each pixel
                    #
                    i_count = (np.ceil(i_mapping_end) - np.floor(i_mapping)).astype(int)
                    j_count = (np.ceil(j_mapping_end) - np.floor(j_mapping)).astype(int)
                    #
                    # --- For I and J, calculate the weights for each pixel
                    #     along each axis.
                    #
                    i_idx = INDEX.Indexes([i_count])
                    j_idx = INDEX.Indexes([j_count])
                    i_weights = i_span_frac[i_idx.rev_idx]
                    j_weights = j_span_frac[j_idx.rev_idx]
                    i_weights[i_idx.fwd_idx] = i_low_frac
                    j_weights[j_idx.fwd_idx] = j_low_frac
                    mask = i_high_frac > 0
                    i_weights[i_idx.fwd_idx[mask]+ i_count[mask] - 1] = \
                             i_high_frac[mask]
                    mask = j_high_frac > 0
                    j_weights[j_idx.fwd_idx[mask] + j_count[mask] - 1] = \
                             j_high_frac[mask]
                    #
                    # Get indexes for the 2-d array, i_count x j_count
                    #
                    idx = INDEX.Indexes([i_count, j_count])
                    #
                    # The coordinates in the straightened space
                    #
                    i_src_hard = i[idx.rev_idx]
                    j_src_hard = j[idx.rev_idx]
                    #
                    # The coordinates in the bin space
                    #
                    i_dest_hard = i_mapping[idx.rev_idx].astype(int) + idx.idx[0]
                    j_dest_hard = j_mapping[idx.rev_idx].astype(int) + idx.idx[1]
                    #
                    # The weights are the i-weight times the j-weight
                    #
                    # The i-weight can be found at the nth index of
                    # i_weights relative to the start of the i_weights
                    # for the pixel in the straightened space.
                    #
                    # The start is found at i_idx.fwd_idx[idx.rev_idx]
                    # the I offset is found at idx.idx[0]
                    #
                    # Similarly for J.
                    #
                    weight_hard = (i_weights[i_idx.fwd_idx[idx.rev_idx] +
                                             idx.idx[0]] *
                                   j_weights[j_idx.fwd_idx[idx.rev_idx] +
                                             idx.idx[1]])
                    i_src = np.hstack((i_src, i_src_hard))
                    j_src = np.hstack((j_src, j_src_hard))
                    i_dest = np.hstack((i_dest, i_dest_hard))
                    j_dest = np.hstack((j_dest, j_dest_hard))
                    weight = np.hstack((weight, weight_hard))
                    labels_src = np.hstack((labels_src, labels_1d[idx.rev_idx]))
                
                self.measure_bins(workspace, i_src, j_src, i_dest, j_dest,
                                  weight, labels_src, scales, nworms)

    def measure_bins(self, workspace, i_src, j_src, i_dest, j_dest,
                     weight, labels_src, scales, nworms):
        '''Measure the intensity in the worm by binning
        
        Consider a transformation from the space of images of straightened worms
        to the space of a grid (the worm gets stretched to fit into the grid).
        This function takes the coordinates of each labeled pixel in the
        straightened worm and computes per-grid-cell measurements on
        the pixels that fall into each grid cell for each straightened image.
        
        A pixel might span bins. In this case, it appears once per overlapped
        bin and it is given a weight proportional to the amount of it's area
        that falls in the bin.
        
        workspace - the workspace for the current image set
        i_src, j_src - the coordinates of the pixels in the straightened space
        i_dest, j_dest - the coordinates of the bins for those pixels
        weight - the fraction of the pixel that falls into the bin
        labels_src - the label for the pixel
        scales - the "scale" portion of the measurement for each of the bins
                 shaped the same as the i_dest, j_dest coordinates
        nworms - # of labels.
        '''
        image_set = workspace.image_set
        assert(isinstance(image_set, cpi.ImageSet))
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        object_name = self.straightened_objects_name.value
        nbins = len(scales)
        for group in self.images:
            image_name = group.straightened_image_name.value
            straightened_image = image_set.get_image(image_name).pixel_data
            straightened_image = straightened_image[i_src, j_src]
            bin_number = (labels_src - 1 + 
                          nworms * j_dest + 
                          nworms * scales.shape[1] * i_dest)
            bin_counts = np.bincount(bin_number)
            bin_weights = np.bincount(bin_number, weight)
            bin_means = (np.bincount(bin_number, weight * straightened_image) / 
                         bin_weights)
            deviances = straightened_image - bin_means[bin_number]
            #
            # Weighted variance = 
            # sum(weight * (x - mean(x)) ** 2) 
            # ---------------------------------
            #  N - 1
            #  ----- sum(weight)
            #    N
            #
            bin_vars = (np.bincount(bin_number, weight * deviances * deviances) /
                        (bin_weights * (bin_counts - 1) / bin_counts))
            bin_stds = np.sqrt(bin_vars)
            nexpected = np.prod(scales.shape) * nworms
            bin_means = np.hstack((bin_means, [np.nan] * (nexpected - len(bin_means))))
            bin_means.shape = (scales.shape[0], scales.shape[1], nworms)
            bin_stds = np.hstack((bin_stds, [np.nan] * (nexpected - len(bin_stds))))
            bin_stds.shape = (scales.shape[0], scales.shape[1], nworms)
            for i in range(scales.shape[0]):
                for j in range(scales.shape[1]):
                    for values, ftr in (
                        (bin_means, FTR_MEAN_INTENSITY),
                        (bin_stds, FTR_STD_INTENSITY)):
                        measurement = "_".join(
                            (C_WORM, ftr, image_name, scales[i][j]))
                        m.add_measurement(object_name, measurement,
                                          values[i,j])
        
                
    def make_objects(self, workspace, labels, nworms):
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        object_set = workspace.object_set
        assert isinstance(object_set, cpo.ObjectSet)
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
    
    def get_scale_name(self, longitudinal, transverse):
        '''Create a scale name, given a longitudinal and transverse band #
        
        longitudinal - band # (0 to # of stripes) or None for transverse-only
        transverse - band # (0 to # of stripes) or None  for longitudinal-only
        '''
        if longitudinal is None:
            longitudinal = 0
            lcount = 1
        else:
            lcount = self.number_of_stripes.value
        if transverse is None:
            transverse = 0
            tcount = 1
        else:
            tcount = self.number_of_segments.value
        return "%s%dof%d_%s%dof%d" % (
            SCALE_HORIZONTAL, transverse+1, tcount,
            SCALE_VERTICAL, longitudinal+1, lcount)
    
    def get_measurement_columns(self, pipeline):
        '''Return columns that define the measurements produced by this module'''
        result = get_object_measurement_columns(self.straightened_objects_name.value)
        if self.wants_measurements:
            nsegments = self.number_of_segments.value
            nstripes = self.number_of_stripes.value
            if nsegments > 1:
                result += [(self.straightened_objects_name.value,
                            "_".join((C_WORM, ftr, 
                                      group.straightened_image_name.value,
                                      self.get_scale_name(None, segment))),
                            cpmeas.COLTYPE_FLOAT)
                           for ftr, group, segment
                           in product((FTR_MEAN_INTENSITY, FTR_STD_INTENSITY),
                                      self.images,
                                      range(nsegments))]
            if nstripes > 1:
                result += [(self.straightened_objects_name.value,
                            "_".join((C_WORM, ftr, 
                                      group.straightened_image_name.value,
                                      self.get_scale_name(stripe, None))),
                            cpmeas.COLTYPE_FLOAT)
                           for ftr, group, stripe
                           in product((FTR_MEAN_INTENSITY, FTR_STD_INTENSITY),
                                      self.images,
                                      range(nstripes))]
            if nsegments > 1 and nstripes > 1:
                result += [(self.straightened_objects_name.value,
                            "_".join((C_WORM, ftr,
                                      group.straightened_image_name.value,
                                      self.get_scale_name(stripe, segment))),
                            cpmeas.COLTYPE_FLOAT)
                           for ftr, group, stripe, segment
                           in product((FTR_MEAN_INTENSITY, FTR_STD_INTENSITY),
                                      self.images,
                                      range(nstripes),
                                      range(nsegments))]
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
        result = []
        if image_name in self.get_measurement_images(
            pipeline, object_name, category, measurement):
            nsegments = self.number_of_segments.value
            nstripes = self.number_of_stripes.value
            if nsegments > 1:
                result += [self.get_scale_name(None, segment)
                           for segment in range(nsegments)]
            if nstripes > 1:
                result += [self.get_scale_name(stripe, None)
                           for stripe in range(nstripes)]
            if nstripes > 1 and nsegments > 1:
                result += [self.get_scale_name(h, v)
                           for h,v in product(
                               range(nstripes),
                               range(nsegments))]
        return result
    
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
        if variable_revision_number == 2:
            #
            # Added horizontal worm measurements
            #
            setting_values = (
                setting_values[:IDX_FLIP_WORMS_V2] + ["1"] +
                setting_values[IDX_FLIP_WORMS_V2:])
            variable_revision_number = 3
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
