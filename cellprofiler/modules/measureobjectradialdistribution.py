"""<b>Measure Object Radial Distribution</b> measures the radial distribution 
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
</ul>

See also <b>MeasureObjectIntensity</b>.
"""

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2014 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org


import numpy as np
import matplotlib.cm
from numpy.ma import masked_array
from scipy.sparse import coo_matrix
import scipy.ndimage as scind
import sys

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.settings as cps
from cellprofiler.settings import YES, NO
import cellprofiler.workspace as cpw

from cellprofiler.cpmath.cpmorphology import distance_to_edge
from cellprofiler.cpmath.cpmorphology import centers_of_labels
from cellprofiler.cpmath.cpmorphology import maximum_position_of_labels
from cellprofiler.cpmath.cpmorphology import color_labels
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.cpmath.propagate import propagate

C_SELF = 'These objects'
C_CENTERS_OF_OTHER_V2 = 'Other objects'
C_CENTERS_OF_OTHER = 'Centers of other objects'
C_EDGES_OF_OTHER = 'Edges of other objects'
C_ALL = [C_SELF, C_CENTERS_OF_OTHER, C_EDGES_OF_OTHER]

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
'''Offset of center choice in object group'''
SETTINGS_CENTER_CHOICE_OFFSET = 1

class MeasureObjectRadialDistribution(cpm.CPModule):
 
    module_name = "MeasureObjectRadialDistribution"
    category = "Measurement"
    variable_revision_number = 3
    
    def create_settings(self):
        self.images = []
        self.objects = []
        self.bin_counts = []
        self.image_count = cps.HiddenCount(self.images)
        self.object_count = cps.HiddenCount(self.objects)
        self.bin_counts_count = cps.HiddenCount(self.bin_counts)
        self.add_image_button = cps.DoSomething("", "Add another image", self.add_image)
        self.spacer_1 = cps.Divider()
        self.add_object_button = cps.DoSomething("", "Add another object",
                                                 self.add_object)
        self.spacer_2 = cps.Divider()
        self.add_bin_count_button = cps.DoSomething("",
                                                    "Add another set of bins", self.add_bin_count)
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
        result = [self.image_count, self.object_count, self.bin_counts_count]
        for x in (self.images, self.objects, self.bin_counts):
            for settings in x:
                temp = settings.pipeline_settings() 
                result += temp
        return result
    
    def visible_settings(self):
        result = []
        
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
        result += [self.add_bin_count_button]
        
        return result
    
    def prepare_settings(self, setting_values):
        '''Adjust the numbers of images, objects and bin counts'''
        image_count, objects_count, bin_counts_count = \
                   [int(x) for x in setting_values[:3]]
        for sequence, add_fn, count in \
            ((self.images, self.add_image, image_count),
             (self.objects, self.add_object, objects_count),
             (self.bin_counts, self.add_bin_count, bin_counts_count)):
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
        if self.show_window:
            workspace.display_data.header = header
            workspace.display_data.stats = stats

    def display(self, workspace, figure):
        header = workspace.display_data.header
        stats = workspace.display_data.stats
        figure.set_subplots((1,1))
        figure.subplot_table(0, 0, stats, col_labels=header)

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
                    best[missing_labels] = missing_best
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
        labels_and_bins = (good_labels-1,bin_indexes[good_mask])
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
            radial_cv.mask = np.sum(~mask,1)==0
            bin_name = str(bin+1) if bin < bin_count else "Overflow"
            statistics += [(image_name, object_name, bin_name, str(bin_count),
                            round(np.mean(masked_fraction_at_distance[:,bin]),4),
                            round(np.mean(masked_mean_pixel_fraction[:, bin]),4),
                            round(np.mean(radial_cv),4))]
        return statistics
    
    def get_measurement_columns(self, pipeline):
        columns = []
        for image in self.images:
            for o in self.objects:
                for bin_count_obj in self.bin_counts:
                    bin_count = bin_count_obj.bin_count.value
                    wants_scaling = bin_count_obj.wants_scaled.value
                    for feature, ofeature in (
                        (MF_FRAC_AT_D, OF_FRAC_AT_D),
                        (MF_MEAN_FRAC, OF_MEAN_FRAC),
                        (MF_RADIAL_CV, OF_RADIAL_CV)):
                        for bin in range(1,bin_count+1):
                            columns.append((o.object_name.value,
                                            feature % (image.image_name.value,
                                                       bin, bin_count),
                                            cpmeas.COLTYPE_FLOAT))
                        if not wants_scaling:
                            columns.append(
                                (o.object_name.value,
                                 ofeature % image.image_name.value,
                                 cpmeas.COLTYPE_FLOAT))
        return columns

    def get_categories(self, pipeline, object_name):
        if object_name in [x.object_name.value for x in self.objects]:
            return [M_CATEGORY]
        return []
    
    def get_measurements(self, pipeline, object_name, category):
        if category in self.get_categories(pipeline, object_name):
            return F_ALL
        return []
    
    def get_measurement_images(self, pipeline, object_name, category, feature):
        if feature in self.get_measurements(pipeline, object_name, category):
            return [image.image_name.value for image in self.images]
        return []
    
    def get_measurement_scales(self, pipeline, object_name, category, feature,
                               image_name):
        if image_name in self.get_measurement_images(pipeline, object_name,
                                                     category, feature):
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
            
        return setting_values, variable_revision_number, from_matlab
    
