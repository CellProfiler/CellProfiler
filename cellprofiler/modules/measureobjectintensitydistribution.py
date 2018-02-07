"""<b>Measure Object Intensity Distribution</b> measures the distribution 
of intensities within each object.
<hr>
Given an image with objects identified, this module measures the
intensity distribution from each object's center to its boundary
within a user-controlled number of bins, i.e. rings.

<p>The distribution is measured from the center of the object, where
the center is defined as the point farthest from any edge.  The numbering
is from 1 (innermost) to <i>N</i> (outermost), where <i>N</i> is the
number of bins specified by the user.
Alternatively, if primary objects exist within the object of interest
(e.g. nuclei within cells), you can choose the center of the primary
objects as the center from which to measure the radial distribution.
This might be useful in cytoplasm-to-nucleus translocation experiments,
for example.  Note that the ring widths are normalized per-object,
i.e., not necessarily a constant width across objects.</p>

<h4>Available measurements</h4>
<ul>
<li><i>FracAtD:</i> Fraction of total stain in an object at a given radius.</li>
<li><i>MeanFrac:</i> Mean fractional intensity at a given radius; calculated
as fraction of total intensity normalized by fraction of pixels at a given radius.</li>
<li><i>RadialCV:</i> Coefficient of variation of intensity within a ring, calculated
over 8 slices.</li>
<li><i>Zernike:</i> The Zernike features characterize the distribution of
intensity across the object. For instance, Zernike 1,1 has a high value if the
tensity is low on one side of the object and high on the other. The
ZernikeMagnitude feature holds the rotationally invariant degree magnitude
of the moment and the ZernikePhase feature gives the moment's orientation.</li>
</ul>

See also <b>MeasureObjectIntensity</b>.
"""

import sys

import centrosome.zernike as cpmz
import matplotlib.cm
import numpy as np
import scipy.ndimage as scind
from centrosome.cpmorphology import centers_of_labels
from centrosome.cpmorphology import color_labels
from centrosome.cpmorphology import distance_to_edge
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.cpmorphology import maximum_position_of_labels
from centrosome.cpmorphology import minimum_enclosing_circle
from centrosome.propagate import propagate
from numpy.ma import masked_array
from scipy.sparse import coo_matrix

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw
from cellprofiler.settings import YES, NO

C_SELF = 'These objects'
C_CENTERS_OF_OTHER_V2 = 'Other objects'
C_CENTERS_OF_OTHER = 'Centers of other objects'
C_EDGES_OF_OTHER = 'Edges of other objects'
C_ALL = [C_SELF, C_CENTERS_OF_OTHER, C_EDGES_OF_OTHER]

Z_NONE = 'None'
Z_MAGNITUDES = "Magnitudes only"
Z_MAGNITUDES_AND_PHASE = "Magnitudes and phase"
Z_ALL = [Z_NONE, Z_MAGNITUDES, Z_MAGNITUDES_AND_PHASE]

M_CATEGORY = 'RadialDistribution'
F_FRAC_AT_D = 'FracAtD'
F_MEAN_FRAC = 'MeanFrac'
F_RADIAL_CV = 'RadialCV'
F_ALL = [F_FRAC_AT_D, F_MEAN_FRAC, F_RADIAL_CV]

FF_SCALE = '%dof%d'
FF_OVERFLOW = 'Overflow'
FF_GENERIC = '_%s_' + FF_SCALE
FF_FRAC_AT_D = F_FRAC_AT_D + FF_GENERIC
FF_MEAN_FRAC = F_MEAN_FRAC + FF_GENERIC
FF_RADIAL_CV = F_RADIAL_CV + FF_GENERIC

FF_ZERNIKE_MAGNITUDE = "ZernikeMagnitude"
FF_ZERNIKE_PHASE = "ZernikePhase"

MF_FRAC_AT_D = '_'.join((M_CATEGORY,FF_FRAC_AT_D))
MF_MEAN_FRAC = '_'.join((M_CATEGORY,FF_MEAN_FRAC))
MF_RADIAL_CV = '_'.join((M_CATEGORY,FF_RADIAL_CV))
OF_FRAC_AT_D = '_'.join((M_CATEGORY, F_FRAC_AT_D, "%s", FF_OVERFLOW))
OF_MEAN_FRAC = '_'.join((M_CATEGORY, F_MEAN_FRAC, "%s", FF_OVERFLOW))
OF_RADIAL_CV = '_'.join((M_CATEGORY, F_RADIAL_CV, "%s", FF_OVERFLOW))

'''# of settings aside from groups'''
SETTINGS_STATIC_COUNT = 3
'''# of settings in image group'''
SETTINGS_IMAGE_GROUP_COUNT = 1
'''# of settings in object group'''
SETTINGS_OBJECT_GROUP_COUNT = 3
'''# of settings in bin group, v1'''
SETTINGS_BIN_GROUP_COUNT_V1 = 1
'''# of settings in bin group, v2'''
SETTINGS_BIN_GROUP_COUNT_V2 = 3
SETTINGS_BIN_GROUP_COUNT = 3
'''# of settings in heatmap group, v4'''
SETTINGS_HEATMAP_GROUP_COUNT_V4 = 7
SETTINGS_HEATMAP_GROUP_COUNT = 7
'''Offset of center choice in object group'''
SETTINGS_CENTER_CHOICE_OFFSET = 1

A_FRAC_AT_D = "Fraction at Distance"
A_MEAN_FRAC = "Mean Fraction"
A_RADIAL_CV = "Radial CV"
MEASUREMENT_CHOICES = [A_FRAC_AT_D, A_MEAN_FRAC, A_RADIAL_CV]

MEASUREMENT_ALIASES = {
    A_FRAC_AT_D: MF_FRAC_AT_D,
    A_MEAN_FRAC: MF_MEAN_FRAC,
    A_RADIAL_CV: MF_RADIAL_CV }

class MeasureObjectIntensityDistribution(cpm.CPModule):

    module_name = "MeasureObjectIntensityDistribution"
    category = "Measurement"
    variable_revision_number = 5

    def create_settings(self):
        self.images = []
        self.objects = []
        self.bin_counts = []
        self.heatmaps = []
        self.image_count = cps.HiddenCount(self.images)
        self.object_count = cps.HiddenCount(self.objects)
        self.bin_counts_count = cps.HiddenCount(self.bin_counts)
        self.heatmap_count = cps.HiddenCount(self.heatmaps)
        self.wants_zernikes = cps.Choice(
            "Calculate intensity Zernikes?", Z_ALL,
            doc="""This setting determines whether the intensity Zernike
            moments are calculated. Choose <i>%(Z_NONE)s</i> to save computation
            time by not calculating the Zernike moments. Choose
            <i>%(Z_MAGNITUDES)s</i> to only save the magnitude information
            and discard information related to the object's angular orientation.
            Choose <i>%(Z_MAGNITUDES_AND_PHASE)s</i> to save the phase information
            as well. The last option lets you recover each object's rough
            appearance from the Zernikes but may not contribute useful
            information if used to classify phenotypes.""" %globals())
        self.zernike_degree = cps.Integer(
            "Maximum zernike moment", value=9, minval=1, maxval=20,
            doc="""(<i>Only if "%s" is %s or %s</i>)<br>
            This is the maximum radial moment that will be calculated.
            There are increasing numbers of azimuthal moments as you increase
            the radial moment, so higher values are increasingly expensive
            to calculate.""" %
            (self.wants_zernikes.text, Z_MAGNITUDES, Z_MAGNITUDES_AND_PHASE))
        self.add_image_button = cps.DoSomething("", "Add another image", self.add_image)
        self.spacer_1 = cps.Divider()
        self.add_object_button = cps.DoSomething("", "Add another object",
                                                 self.add_object)
        self.spacer_2 = cps.Divider()
        self.add_bin_count_button = cps.DoSomething(
            "", "Add another set of bins", self.add_bin_count)
        self.spacer_3 = cps.Divider()
        self.add_heatmap_button = cps.DoSomething(
            "", "Add another heatmap display", self.add_heatmap,
            doc = """
        Press this button to add a display of one of the radial distribution
        measurements. Each radial band of the object is colored using a
        heatmap according to the measurement value for that band.""")
        self.add_image(can_remove = False)
        self.add_object(can_remove = False)
        self.add_bin_count(can_remove = False)

    def add_image(self, can_remove = True):
        '''Add an image to be measured'''
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))
        group.append("image_name", cps.ImageNameSubscriber(
                "Select an image to measure", cps.NONE, doc="""
                Select the image that you want to measure the intensity from."""))

        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove this image", self.images, group))
        self.images.append(group)

    def add_object(self, can_remove = True):
        '''Add an object to be measured (plus optional centers)'''
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))
        group.append("object_name", cps.ObjectNameSubscriber(
                "Select objects to measure", cps.NONE,doc="""
                Select the objects that you want to measure the intensity from."""))

        group.append("center_choice", cps.Choice(
                "Object to use as center?", C_ALL,doc="""
                There are three ways to specify the center of the radial measurement:
                <ul>
                <li><i>%(C_SELF)s:</i> Use the centers of these objects for the
                radial measurement.</li>
                <li><i>%(C_CENTERS_OF_OTHER)s:</i> Use the centers of other objects
                for the radial measurement.</li>
                <li><i>%(C_EDGES_OF_OTHER)s:</i> Measure distances from the
                edge of the other object to each pixel outside of the
                centering object. Do not include pixels within the centering
                object in the radial measurement calculations.</li>
                </ul>
                For example, if measuring the radial distribution in a Cell
                object, you can use the center of the Cell objects (<i>%(C_SELF)s</i>)
                or you can use previously identified Nuclei objects as
                the centers (<i>%(C_CENTERS_OF_OTHER)s</i>)."""%globals()))

        group.append("center_object_name", cps.ObjectNameSubscriber(
                "Select objects to use as centers", cps.NONE, doc="""
                <i>(Used only if "%(C_CENTERS_OF_OTHER)s" are selected for centers)</i><br>
                Select the object to use as the center, or select <i>None</i> to
                use the input object centers (which is the same as selecting
                <i>%(C_SELF)s</i> for the object centers)."""%globals()))

        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove this object", self.objects, group))
        self.objects.append(group)

    def add_bin_count(self, can_remove = True):
        '''Add another radial bin count at which to measure'''
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))

        group.append("wants_scaled", cps.Binary(
                "Scale the bins?", True,doc ="""
                <p>Select <i>%(YES)s</i> to divide the object radially into the number
                of bins that you specify. </p>
                <p>Select <i>%(NO)s</i> to create the number of bins you specify based
                on distance. For this option, the user will be
                asked to specify a maximum distance so that each object will have the
                same measurements (which might be zero for small objects) and so that
                the measurements can be taken without knowing the maximum object radius
                before the run starts.</p>"""%globals()))

        group.append("bin_count", cps.Integer(
                "Number of bins", 4, 2, doc="""
                Specify the number of bins that you want to use to measure
                the distribution. Radial distribution is measured with respect to a series
                of concentric rings starting from the object center (or
                more generally, between contours at a normalized distance
                from the object center). This number
                specifies the number of rings into which the distribution is to
                be divided. Additional ring counts can be specified
                by clicking the <i>Add another set of bins</i> button."""))

        group.append("maximum_radius", cps.Integer(
                "Maximum radius", 100, minval = 1,doc = """
                Specify the maximum radius for the unscaled bins. The unscaled binning
                method creates the number of bins that you
                specify and creates equally spaced bin boundaries up to the maximum
                radius. Parts of the object that are beyond this radius will be
                counted in an overflow bin. The radius is measured in pixels."""))

        group.can_remove = can_remove
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove this set of bins", self.bin_counts, group))
        self.bin_counts.append(group)

    def get_bin_count_choices(self, pipeline=None):
        choices = []
        for bin_count in self.bin_counts:
            nbins = str(bin_count.bin_count.value)
            if nbins != choices:
                choices.append(nbins)
        return choices

    def add_heatmap(self):
        group = cps.SettingsGroup()
        if len(self.heatmaps) > 0:
            group.append("divider", cps.Divider(line=False))
        group.append("image_name", MORDImageNameSubscriber(
            "Image", doc="""
            The heatmap will be displayed with measurements taken using this image.
            The setting will let you choose from among the images you have
            specified in "Select image to measure".
            """))
        group.image_name.set_module(self)
        group.append("object_name", MORDObjectNameSubscriber(
            "Objects to display", doc = """
            The objects to display in the heatmap. You can select any of the
            objects chosen in "Select objects to measure".
            """))
        group.object_name.set_module(self)
        group.append("bin_count", cps.Choice(
            "Number of bins", self.get_bin_count_choices(),
            choices_fn = self.get_bin_count_choices))
        def get_number_of_bins(module = self, group=group):
            if len(module.bin_counts) == 1:
                return module.bin_counts[0].bin_count.value
            else:
                return int(group.bin_count.value)
        group.get_number_of_bins = get_number_of_bins

        group.append("measurement", cps.Choice(
            "Measurement", MEASUREMENT_CHOICES,
            doc="""The measurement to display."""))
        group.append("colormap", cps.Colormap(
            "Color map",
            doc="""
            The color map setting chooses the color palette that will be
            used to render the different values for your measurement. If you
            choose "gray", the image will label each of the bins with the
            actual image measurement.
            """))
        group.append("wants_to_save_display", cps.Binary(
            "Save display as image?", False,
            doc = """This setting allows you to save the heatmap display as
            an image that can be output using the <b>SaveImages</b> module.
            Choose <i>%(YES)s</i> to save the display or <i>%(NO)s</i> if the
            display is not needed."""%globals()))
        group.append("display_name", cps.ImageNameProvider(
            "Output image name", "Heatmap",
            doc = """
            <i>(Only used if "Save display as image?" is "%(YES)s")</i><br>
            This setting names the heatmap image so that the name you enter
            here can be selected in a later <b>SaveImages</b> or other module.
            """%globals()))
        group.append("remover", cps.RemoveSettingButton(
            "", "Remove this heatmap display", self.heatmaps, group))
        self.heatmaps.append(group)

    def validate_module(self, pipeline):
        """Make sure chosen objects, images and bins are selected only once"""
        images = set()
        for group in self.images:
            if group.image_name.value in images:
                raise cps.ValidationError(
                    "%s has already been selected" %group.image_name.value,
                    group.image_name)
            images.add(group.image_name.value)

        objects = set()
        for group in self.objects:
            if group.object_name.value in objects:
                raise cps.ValidationError(
                    "%s has already been selected" %group.object_name.value,
                    group.object_name)
            objects.add(group.object_name.value)

        bins = set()
        for group in self.bin_counts:
            if group.bin_count.value in bins:
                raise cps.ValidationError(
                    "%s has already been selected" %group.bin_count.value,
                    group.bin_count)
            bins.add(group.bin_count.value)

    def settings(self):
        result = [self.image_count, self.object_count, self.bin_counts_count,
                  self.heatmap_count, self.wants_zernikes, self.zernike_degree]
        for x in (self.images, self.objects, self.bin_counts, self.heatmaps):
            for settings in x:
                temp = settings.pipeline_settings()
                result += temp
        return result

    def visible_settings(self):
        result = [self.wants_zernikes]
        if self.wants_zernikes != Z_NONE:
            result.append(self.zernike_degree)

        for settings in self.images:
            result += settings.visible_settings()
        result += [self.add_image_button, self.spacer_1]

        for settings in self.objects:
            temp = settings.visible_settings()
            if settings.center_choice.value == C_SELF:
                temp.remove(settings.center_object_name)
            result += temp
        result += [self.add_object_button, self.spacer_2]

        for settings in self.bin_counts:
            result += [settings.wants_scaled, settings.bin_count]
            if not settings.wants_scaled:
                result += [settings.maximum_radius]
            if settings.can_remove:
                result += [settings.remover]
        result += [self.add_bin_count_button, self.spacer_3]

        for settings in self.heatmaps:
            if hasattr(settings, "divider"):
                result.append(settings.divider)
            if settings.image_name.is_visible():
                result.append(settings.image_name)
            if settings.object_name.is_visible():
                result.append(settings.object_name)
            if len(self.bin_counts) > 1:
                result.append(settings.bin_count)
            result += [settings.measurement, settings.colormap,
                       settings.wants_to_save_display]
            if settings.wants_to_save_display:
                result.append(settings.display_name)
            result.append(settings.remover)
        result += [self.add_heatmap_button]
        return result

    def prepare_settings(self, setting_values):
        '''Adjust the numbers of images, objects and bin counts'''
        image_count, objects_count, bin_counts_count, heatmap_count = \
                   [int(x) for x in setting_values[:4]]
        for sequence, add_fn, count in \
            ((self.images, self.add_image, image_count),
             (self.objects, self.add_object, objects_count),
             (self.bin_counts, self.add_bin_count, bin_counts_count),
             (self.heatmaps, self.add_heatmap, heatmap_count)):
            while len(sequence) > count:
                del sequence[-1]
            while len(sequence) < count:
                add_fn()


    def run(self, workspace):
        header = ("Image","Objects","Bin # (innermost=1)","Bin count","Fraction","Intensity","COV")
        stats = []
        d = {}
        for image in self.images:
            for o in self.objects:
                for bin_count_settings in self.bin_counts:
                    stats += \
                    self.do_measurements(workspace,
                                         image.image_name.value,
                                         o.object_name.value,
                                         o.center_object_name.value
                                         if o.center_choice != C_SELF
                                         else None,
                                         o.center_choice.value,
                                         bin_count_settings,
                                         d)
        if self.wants_zernikes != Z_NONE:
            self.calculate_zernikes(workspace)

        if self.show_window:
            workspace.display_data.header = header
            workspace.display_data.stats = stats
            workspace.display_data.heatmaps = []
        for heatmap in self.heatmaps:
            heatmap_img = d.get(id(heatmap))
            if heatmap_img is not None:
                if self.show_window or heatmap.wants_to_save_display:
                    labels = workspace.object_set.get_objects(
                        heatmap.object_name.get_objects_name()).segmented
                if self.show_window:
                    workspace.display_data.heatmaps.append(
                        (heatmap_img, labels != 0))
                if heatmap.wants_to_save_display:
                    colormap = heatmap.colormap.value
                    if colormap == matplotlib.cm.gray.name:
                        output_pixels = heatmap_img
                    else:
                        if colormap == cps.DEFAULT:
                            colormap = cpprefs.get_default_colormap()
                        cm = matplotlib.cm.ScalarMappable(
                            cmap = colormap)
                        output_pixels = cm.to_rgba(heatmap_img)[:, :, :3]
                        output_pixels[labels == 0, :] = 0
                    parent_image = workspace.image_set.get_image(
                        heatmap.image_name.get_image_name())
                    output_img = cpi.Image(
                        output_pixels,
                        parent_image = parent_image)
                    img_name = heatmap.display_name.value
                    workspace.image_set.add(img_name, output_img)

    def display(self, workspace, figure):
        header = workspace.display_data.header
        stats = workspace.display_data.stats
        n_plots = len(workspace.display_data.heatmaps) + 1
        n_vert = int(np.sqrt(n_plots))
        n_horiz = int(np.ceil(float(n_plots) / n_vert))
        figure.set_subplots((n_horiz, n_vert))
        figure.subplot_table(0, 0, stats, col_labels=header)
        idx = 1
        sharexy = None
        for heatmap, (heatmap_img, mask) in zip(
            self.heatmaps, workspace.display_data.heatmaps):
            heatmap_img = np.ma.array(heatmap_img, mask=~mask)
            if heatmap_img is not None:
                title = "%s %s %s" % (
                    heatmap.image_name.get_image_name(),
                    heatmap.object_name.get_objects_name(),
                    heatmap.measurement.value)
                x = idx % n_horiz
                y = int(idx / n_horiz)
                colormap = heatmap.colormap.value
                if colormap == cps.DEFAULT:
                    colormap = cpprefs.get_default_colormap()
                if sharexy is None:
                    sharexy = figure.subplot_imshow(
                        x, y, heatmap_img,
                        title = title,
                        colormap = colormap,
                        normalize = False,
                        vmin = np.min(heatmap_img),
                        vmax = np.max(heatmap_img),
                        colorbar = True)
                else:
                    figure.subplot_imshow(
                        x, y, heatmap_img,
                        title = title,
                        colormap = colormap,
                        colorbar = True,
                        normalize = False,
                        vmin = np.min(heatmap_img),
                        vmax = np.max(heatmap_img),
                        sharexy = sharexy)
                idx += 1

    def do_measurements(self, workspace, image_name, object_name,
                        center_object_name, center_choice,
                        bin_count_settings, dd):
        '''Perform the radial measurements on the image set

        workspace - workspace that holds images / objects
        image_name - make measurements on this image
        object_name - make measurements on these objects
        center_object_name - use the centers of these related objects as
                      the centers for radial measurements. None to use the
                      objects themselves.
        center_choice - the user's center choice for this object:
                      C_SELF, C_CENTERS_OF_OBJECTS or C_EDGES_OF_OBJECTS.
        bin_count_settings - the bin count settings group
        d - a dictionary for saving reusable partial results

        returns one statistics tuple per ring.
        '''
        assert isinstance(workspace, cpw.Workspace)
        assert isinstance(workspace.object_set, cpo.ObjectSet)
        bin_count = bin_count_settings.bin_count.value
        wants_scaled = bin_count_settings.wants_scaled.value
        maximum_radius = bin_count_settings.maximum_radius.value

        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale=True)
        objects = workspace.object_set.get_objects(object_name)
        labels, pixel_data = cpo.crop_labels_and_image(objects.segmented,
                                                       image.pixel_data)
        nobjects = np.max(objects.segmented)
        measurements = workspace.measurements
        assert isinstance(measurements, cpmeas.Measurements)
        heatmaps = {}
        for heatmap in self.heatmaps:
            if heatmap.object_name.get_objects_name() == object_name and \
               image_name == heatmap.image_name.get_image_name() and \
               heatmap.get_number_of_bins() == bin_count:
                dd[id(heatmap)] = \
                heatmaps[MEASUREMENT_ALIASES[heatmap.measurement.value]] = \
                np.zeros(labels.shape)
        if nobjects == 0:
            for bin in range(1, bin_count+1):
                for feature in (F_FRAC_AT_D, F_MEAN_FRAC, F_RADIAL_CV):
                    feature_name = (
                        (feature + FF_GENERIC) % (image_name, bin, bin_count))
                    measurements.add_measurement(
                        object_name, "_".join([M_CATEGORY, feature_name]),
                        np.zeros(0))
                    if not wants_scaled:
                        measurement_name = "_".join([M_CATEGORY, feature,
                                                     image_name, FF_OVERFLOW])
                        measurements.add_measurement(
                            object_name, measurement_name, np.zeros(0))
            return [(image_name, object_name, "no objects","-","-","-","-")]
        name = (object_name if center_object_name is None
                else "%s_%s"%(object_name, center_object_name))
        if dd.has_key(name):
            normalized_distance, i_center, j_center, good_mask = dd[name]
        else:
            d_to_edge = distance_to_edge(labels)
            if center_object_name is not None:
                #
                # Use the center of the centering objects to assign a center
                # to each labeled pixel using propagation
                #
                center_objects=workspace.object_set.get_objects(center_object_name)
                center_labels, cmask = cpo.size_similarly(
                    labels, center_objects.segmented)
                pixel_counts = fix(scind.sum(
                    np.ones(center_labels.shape),
                    center_labels,
                    np.arange(1, np.max(center_labels)+1,dtype=np.int32)))
                good = pixel_counts > 0
                i,j = (centers_of_labels(center_labels) + .5).astype(int)
                ig = i[good]
                jg = j[good]
                lg = np.arange(1, len(i)+1)[good]
                if center_choice == C_CENTERS_OF_OTHER:
                    #
                    # Reduce the propagation labels to the centers of
                    # the centering objects
                    #
                    center_labels = np.zeros(center_labels.shape, int)
                    center_labels[ig,jg] = lg
                cl,d_from_center = propagate(np.zeros(center_labels.shape),
                                             center_labels,
                                             labels != 0, 1)
                #
                # Erase the centers that fall outside of labels
                #
                cl[labels == 0] = 0
                #
                # If objects are hollow or crescent-shaped, there may be
                # objects without center labels. As a backup, find the
                # center that is the closest to the center of mass.
                #
                missing_mask = (labels != 0) & (cl == 0)
                missing_labels = np.unique(labels[missing_mask])
                if len(missing_labels):
                    all_centers = centers_of_labels(labels)
                    missing_i_centers, missing_j_centers = \
                                     all_centers[:, missing_labels-1]
                    di = missing_i_centers[:, np.newaxis] - ig[np.newaxis, :]
                    dj = missing_j_centers[:, np.newaxis] - jg[np.newaxis, :]
                    missing_best = lg[np.argsort((di*di + dj*dj, ))[:, 0]]
                    best = np.zeros(np.max(labels) + 1, int)
                    best[missing_labels] = missing_best[0]
                    cl[missing_mask] = best[labels[missing_mask]]
                    #
                    # Now compute the crow-flies distance to the centers
                    # of these pixels from whatever center was assigned to
                    # the object.
                    #
                    iii, jjj = np.mgrid[0:labels.shape[0], 0:labels.shape[1]]
                    di = iii[missing_mask] - i[cl[missing_mask] - 1]
                    dj = jjj[missing_mask] - j[cl[missing_mask] - 1]
                    d_from_center[missing_mask] = np.sqrt(di*di + dj*dj)
            else:
                # Find the point in each object farthest away from the edge.
                # This does better than the centroid:
                # * The center is within the object
                # * The center tends to be an interesting point, like the
                #   center of the nucleus or the center of one or the other
                #   of two touching cells.
                #
                i,j = maximum_position_of_labels(d_to_edge, labels, objects.indices)
                center_labels = np.zeros(labels.shape, int)
                center_labels[i,j] = labels[i,j]
                #
                # Use the coloring trick here to process touching objects
                # in separate operations
                #
                colors = color_labels(labels)
                ncolors = np.max(colors)
                d_from_center = np.zeros(labels.shape)
                cl = np.zeros(labels.shape, int)
                for color in range(1,ncolors+1):
                    mask = colors == color
                    l,d = propagate(np.zeros(center_labels.shape),
                                    center_labels,
                                    mask, 1)
                    d_from_center[mask] = d[mask]
                    cl[mask] = l[mask]
            good_mask = cl > 0
            if center_choice == C_EDGES_OF_OTHER:
                # Exclude pixels within the centering objects
                # when performing calculations from the centers
                good_mask = good_mask & (center_labels == 0)
            i_center = np.zeros(cl.shape)
            i_center[good_mask] = i[cl[good_mask]-1]
            j_center = np.zeros(cl.shape)
            j_center[good_mask] = j[cl[good_mask]-1]

            normalized_distance = np.zeros(labels.shape)
            if wants_scaled:
                total_distance = d_from_center + d_to_edge
                normalized_distance[good_mask] = (d_from_center[good_mask] /
                                                  (total_distance[good_mask] + .001))
            else:
                normalized_distance[good_mask] = \
                    d_from_center[good_mask] / maximum_radius
            dd[name] = [normalized_distance, i_center, j_center, good_mask]
        ngood_pixels = np.sum(good_mask)
        good_labels = labels[good_mask]
        bin_indexes = (normalized_distance * bin_count).astype(int)
        bin_indexes[bin_indexes > bin_count] = bin_count
        labels_and_bins = (good_labels-1, bin_indexes[good_mask])
        histogram = coo_matrix((pixel_data[good_mask], labels_and_bins),
                               (nobjects, bin_count+1)).toarray()
        sum_by_object = np.sum(histogram, 1)
        sum_by_object_per_bin = np.dstack([sum_by_object]*(bin_count + 1))[0]
        fraction_at_distance = histogram / sum_by_object_per_bin
        number_at_distance = coo_matrix((np.ones(ngood_pixels),labels_and_bins),
                                        (nobjects, bin_count+1)).toarray()
        object_mask = number_at_distance > 0
        sum_by_object = np.sum(number_at_distance, 1)
        sum_by_object_per_bin = np.dstack([sum_by_object]*(bin_count+1))[0]
        fraction_at_bin = number_at_distance / sum_by_object_per_bin
        mean_pixel_fraction = fraction_at_distance / (fraction_at_bin +
                                                      np.finfo(float).eps)
        masked_fraction_at_distance = masked_array(fraction_at_distance,
                                                   ~object_mask)
        masked_mean_pixel_fraction = masked_array(mean_pixel_fraction,
                                                  ~object_mask)
        # Anisotropy calculation.  Split each cell into eight wedges, then
        # compute coefficient of variation of the wedges' mean intensities
        # in each ring.
        #
        # Compute each pixel's delta from the center object's centroid
        i,j = np.mgrid[0:labels.shape[0], 0:labels.shape[1]]
        imask = i[good_mask] > i_center[good_mask]
        jmask = j[good_mask] > j_center[good_mask]
        absmask = (abs(i[good_mask] - i_center[good_mask]) >
                   abs(j[good_mask] - j_center[good_mask]))
        radial_index = (imask.astype(int) + jmask.astype(int)*2 +
                        absmask.astype(int)*4)
        statistics = []

        for bin in range(bin_count + (0 if wants_scaled else 1)):
            bin_mask = (good_mask & (bin_indexes == bin))
            bin_pixels = np.sum(bin_mask)
            bin_labels = labels[bin_mask]
            bin_radial_index = radial_index[bin_indexes[good_mask] == bin]
            labels_and_radii = (bin_labels-1, bin_radial_index)
            radial_values = coo_matrix((pixel_data[bin_mask],
                                        labels_and_radii),
                                       (nobjects, 8)).toarray()
            pixel_count = coo_matrix((np.ones(bin_pixels), labels_and_radii),
                                     (nobjects, 8)).toarray()
            mask = pixel_count==0
            radial_means = masked_array(radial_values / pixel_count, mask)
            radial_cv = np.std(radial_means,1) / np.mean(radial_means, 1)
            radial_cv[np.sum(~mask,1)==0] = 0
            for measurement, feature, overflow_feature in (
                (fraction_at_distance[:,bin], MF_FRAC_AT_D, OF_FRAC_AT_D),
                (mean_pixel_fraction[:,bin], MF_MEAN_FRAC, OF_MEAN_FRAC),
                (np.array(radial_cv), MF_RADIAL_CV, OF_RADIAL_CV)):

                if bin == bin_count:
                    measurement_name = overflow_feature % image_name
                else:
                    measurement_name = feature % (image_name, bin+1, bin_count)
                measurements.add_measurement(object_name,
                                             measurement_name,
                                             measurement)
                if feature in heatmaps:
                    heatmaps[feature][bin_mask] = measurement[bin_labels-1]
            radial_cv.mask = np.sum(~mask,1)==0
            bin_name = str(bin+1) if bin < bin_count else "Overflow"
            statistics += [(image_name, object_name, bin_name, str(bin_count),
                            round(np.mean(masked_fraction_at_distance[:,bin]),4),
                            round(np.mean(masked_mean_pixel_fraction[:, bin]),4),
                            round(np.mean(radial_cv),4))]
        return statistics

    def calculate_zernikes(self, workspace):
        zernike_indexes = cpmz.get_zernike_indexes(self.zernike_degree.value+1)
        meas = workspace.measurements
        for o in self.objects:
            object_name = o.object_name.value
            objects = workspace.object_set.get_objects(object_name)
            #
            # First, get a table of centers and radii of minimum enclosing
            # circles per object
            #
            ij = np.zeros((objects.count+1, 2))
            r = np.zeros(objects.count+1)
            for labels, indexes in objects.get_labels():
                ij_, r_ = minimum_enclosing_circle(labels, indexes)
                ij[indexes] = ij_
                r[indexes] = r_
            #
            # Then compute x and y, the position of each labeled pixel
            # within a unit circle around the object
            #
            ijv = objects.ijv
            l = ijv[:, 2]
            yx = (ijv[:, :2] - ij[l, :]) / r[l, np.newaxis]
            z = cpmz.construct_zernike_polynomials(
                yx[:, 1], yx[:, 0], zernike_indexes)
            for image_group in self.images:
                image_name = image_group.image_name.value
                image = workspace.image_set.get_image(
                    image_name, must_be_grayscale=True)
                pixels = image.pixel_data
                mask = (ijv[:, 0] < pixels.shape[0]) &\
                    (ijv[:, 1] < pixels.shape[1])
                mask[mask] = image.mask[ijv[mask, 0], ijv[mask, 1]]
                yx_ = yx[mask, :]
                l_ = l[mask]
                z_ = z[mask, :]
                if len(l_) == 0:
                    for i, (n, m) in enumerate(zernike_indexes):
                        ftr = self.get_zernike_magnitude_name(image_name, n, m)
                        meas[object_name, ftr] = np.zeros(0)
                        if self.wants_zernikes == Z_MAGNITUDES_AND_PHASE:
                            ftr = self.get_zernike_phase_name(image_name, n, m)
                            meas[object_name, ftr] = np.zeros(0)
                    continue
                areas = scind.sum(
                    np.ones(l_.shape, int), labels=l_, index=objects.indices)
                for i, (n, m) in enumerate(zernike_indexes):
                    vr = scind.sum(
                        pixels[ijv[mask, 0], ijv[mask, 1]] * z_[:, i].real,
                        labels=l_, index = objects.indices)
                    vi = scind.sum(
                        pixels[ijv[mask, 0], ijv[mask, 1]] * z_[:, i].imag,
                        labels=l_, index = objects.indices)
                    magnitude = np.sqrt(vr*vr + vi*vi) / areas
                    ftr = self.get_zernike_magnitude_name(image_name, n, m)
                    meas[object_name, ftr] = magnitude
                    if self.wants_zernikes == Z_MAGNITUDES_AND_PHASE:
                        phase = np.arctan2(vr, vi)
                        ftr = self.get_zernike_phase_name(image_name, n, m)
                        meas[object_name, ftr] = phase

    def get_zernike_magnitude_name(self, image_name, n, m):
        '''The feature name of the magnitude of a Zernike moment

        image_name - the name of the image being measured
        n - the radial moment of the Zernike
        m - the azimuthal moment of the Zernike
        '''
        return "_".join(
            (M_CATEGORY, FF_ZERNIKE_MAGNITUDE, image_name, str(n), str(m)))

    def get_zernike_phase_name(self, image_name, n, m):
        '''The feature name of the phase of a Zernike moment

        image_name - the name of the image being measured
        n - the radial moment of the Zernike
        m - the azimuthal moment of the Zernike
        '''
        return "_".join(
            (M_CATEGORY, FF_ZERNIKE_PHASE, image_name, str(n), str(m)))

    def get_measurement_columns(self, pipeline):
        columns = []
        for image in self.images:
            image_name = image.image_name.value
            for o in self.objects:
                object_name = o.object_name.value
                for bin_count_obj in self.bin_counts:
                    bin_count = bin_count_obj.bin_count.value
                    wants_scaling = bin_count_obj.wants_scaled.value
                    for feature, ofeature in (
                        (MF_FRAC_AT_D, OF_FRAC_AT_D),
                        (MF_MEAN_FRAC, OF_MEAN_FRAC),
                        (MF_RADIAL_CV, OF_RADIAL_CV)):
                        for bin in range(1,bin_count+1):
                            columns.append(
                                (object_name,
                                 feature % (image_name, bin, bin_count),
                                 cpmeas.COLTYPE_FLOAT))
                        if not wants_scaling:
                            columns.append(
                                (object_name,
                                 ofeature % image.image_name.value,
                                 cpmeas.COLTYPE_FLOAT))
                    if self.wants_zernikes != Z_NONE:
                        name_fns = [self.get_zernike_magnitude_name]
                        if self.wants_zernikes == Z_MAGNITUDES_AND_PHASE:
                            name_fns.append(self.get_zernike_phase_name)
                        max_n = self.zernike_degree.value
                        for name_fn in name_fns:
                            for n, m in cpmz.get_zernike_indexes(max_n+1):
                                ftr = name_fn(image_name, n, m)
                                columns.append(
                                    (object_name, ftr, cpmeas.COLTYPE_FLOAT))
        return columns

    def get_categories(self, pipeline, object_name):
        if object_name in [x.object_name.value for x in self.objects]:
            return [M_CATEGORY]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if category in self.get_categories(pipeline, object_name):
            if self.wants_zernikes == Z_NONE:
                return F_ALL
            elif self.wants_zernikes == Z_MAGNITUDES:
                return F_ALL + [FF_ZERNIKE_MAGNITUDE]
            else:
                return F_ALL + [FF_ZERNIKE_MAGNITUDE, FF_ZERNIKE_PHASE]
        return []

    def get_measurement_images(self, pipeline, object_name, category, feature):
        if feature in self.get_measurements(pipeline, object_name, category):
            return [image.image_name.value for image in self.images]
        return []

    def get_measurement_scales(self, pipeline, object_name, category, feature,
                               image_name):
        if image_name in self.get_measurement_images(pipeline, object_name,
                                                     category, feature):
            if feature in (FF_ZERNIKE_MAGNITUDE, FF_ZERNIKE_PHASE):
                n_max = self.zernike_degree.value
                result = ["%d_%d" % (n, m)
                          for n, m in cpmz.get_zernike_indexes(n_max+1)]
            else:
                result = [FF_SCALE % (bin,bin_count.bin_count.value)
                          for bin_count in self.bin_counts
                          for bin in range(1, bin_count.bin_count.value+1)]
                if any([not bin_count.wants_scaled.value
                        for bin_count in self.bin_counts]):
                    result += [FF_OVERFLOW]
            return result
        return []

    def upgrade_settings(self,setting_values,variable_revision_number,
                         module_name,from_matlab):
        if from_matlab and variable_revision_number == 1:
            image_name, object_name, center_name, bin_count = setting_values[:4]
            if center_name == cps.DO_NOT_USE:
                center_choice = C_SELF
            else:
                center_choice = C_CENTERS_OF_OTHER
            setting_values = ["1","1","1",image_name,
                              object_name, center_choice, center_name,
                              bin_count]
            variable_revision_number = 1
            from_matlab = False
        if variable_revision_number == 1:
            n_images, n_objects, n_bins = [
                int(setting) for setting in setting_values[:3]]
            off_bins = (SETTINGS_STATIC_COUNT +
                        n_images * SETTINGS_IMAGE_GROUP_COUNT +
                        n_objects * SETTINGS_OBJECT_GROUP_COUNT)
            new_setting_values = setting_values[:off_bins]
            for bin_count in setting_values[off_bins:]:
                new_setting_values += [ cps.YES, bin_count, "100"]
            setting_values = new_setting_values
            variable_revision_number = 2
        if variable_revision_number == 2:
            n_images, n_objects = [
                int(setting) for setting in setting_values[:2]]
            off_objects = (SETTINGS_STATIC_COUNT +
                           n_images * SETTINGS_IMAGE_GROUP_COUNT)
            setting_values = list(setting_values)
            for i in range(n_objects):
                offset = (off_objects + i * SETTINGS_OBJECT_GROUP_COUNT +
                          SETTINGS_CENTER_CHOICE_OFFSET)
                if setting_values[offset] == C_CENTERS_OF_OTHER_V2:
                    setting_values[offset] = C_CENTERS_OF_OTHER
            variable_revision_number = 3
        if variable_revision_number == 3:
            # added heatmaps
            # Need a heatmap_count = 0
            #
            setting_values = setting_values[:3] + ["0"] + setting_values[3:]
            variable_revision_number = 4

        if variable_revision_number == 4:
            #
            # Added zernikes
            #
            setting_values = setting_values[:4] + [Z_NONE, "9"] +\
                setting_values[4:]
            variable_revision_number = 5

        return setting_values, variable_revision_number, from_matlab

class MORDObjectNameSubscriber(cps.ObjectNameSubscriber):
    '''An object name subscriber limited by the objects in the objects' group'''
    def set_module(self, module):
        assert isinstance(module, MeasureObjectIntensityDistribution)
        self.__module = module

    def __is_valid_choice(self, choice_tuple):
        for object_group in self.__module.objects:
            if choice_tuple[0] == object_group.object_name:
                return True
        return False

    def get_choices(self, pipeline):
        super_choices = super(self.__class__, self).get_choices(pipeline)
        return filter(self.__is_valid_choice, super_choices)

    def is_visible(self):
        '''Return True if a choice should be displayed'''
        return len(self.__module.objects) > 1

    def get_objects_name(self):
        '''Return the name of the objects to use in the display'''
        if len(self.__module.objects) == 1:
            return self.__module.objects[0].object_name.value
        else:
            return self.value

class MORDImageNameSubscriber(cps.ImageNameSubscriber):
    '''An image name subscriber limited by the images in the image group'''
    def set_module(self, module):
        assert isinstance(module, MeasureObjectIntensityDistribution)
        self.__module = module

    def __is_valid_choice(self, choice_tuple):
        for image_group in self.__module.images:
            if choice_tuple[0] == image_group.image_name:
                return True
        return False

    def get_choices(self, pipeline):
        super_choices = super(self.__class__, self).get_choices(pipeline)
        return filter(self.__is_valid_choice, super_choices)

    def is_visible(self):
        '''Return True if a choice should be displayed'''
        return len(self.__module.images) > 1

    def get_image_name(self):
        '''Return the name of the image to use in the display'''
        if len(self.__module.images) == 1:
            return self.__module.images[0].image_name.value
        else:
            return self.value
