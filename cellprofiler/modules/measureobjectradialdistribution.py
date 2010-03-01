"""<b>Measure Object Radial Distribution</b> measures the radial distribution 
of intensities within each object
<hr>

Given an image with objects identified, this module measures the
intensity distribution from each object's center to its boundary 
within a user-controlled number of bins.

The distribution is measured from the center of the object, where 
the center is defined as the point farthest from any edge.
Alternatively, if primary objects exist within the object of interest
(e.g. nuclei within cells), you can choose the center of the the primary
objects as the center from which to measure the radial distribution.
This might be useful in cytoplasm-to-nucleus translocation experiments, 
for example.

<h4>Available measurements</h4>
<ul>
<li>FracAtD: Fraction of total stain in an object at a given radius.
<li>MeanFrac: Mean fractional intensity at a given radius; calculated
as fraction of total intensity normalized by fraction of pixels at a given radius.</li>
<li>RadialCV: Coefficient of variation of intensity within a ring, calculated 
over 8 slices.</li>
</ul>
<br>
See also <b>MeasureObjectIntensity</b>.
"""

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
import cellprofiler.workspace as cpw

from cellprofiler.cpmath.cpmorphology import distance_to_edge
from cellprofiler.cpmath.cpmorphology import centers_of_labels
from cellprofiler.cpmath.cpmorphology import maximum_position_of_labels
from cellprofiler.cpmath.cpmorphology import color_labels
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.cpmath.propagate import propagate

C_SELF = 'These objects'
C_OTHER = 'Other objects'
C_ALL = [C_SELF, C_OTHER]

M_CATEGORY = 'RadialDistribution'
F_FRAC_AT_D = 'FracAtD'
F_MEAN_FRAC = 'MeanFrac'
F_RADIAL_CV = 'RadialCV'
F_ALL = [F_FRAC_AT_D, F_MEAN_FRAC, F_RADIAL_CV]

FF_SCALE = '%dof%d'
FF_GENERIC = '_%s_' + FF_SCALE
FF_FRAC_AT_D = F_FRAC_AT_D + FF_GENERIC
FF_MEAN_FRAC = F_MEAN_FRAC + FF_GENERIC
FF_RADIAL_CV = F_RADIAL_CV + FF_GENERIC

MF_FRAC_AT_D = '_'.join((M_CATEGORY,FF_FRAC_AT_D))
MF_MEAN_FRAC = '_'.join((M_CATEGORY,FF_MEAN_FRAC))
MF_RADIAL_CV = '_'.join((M_CATEGORY,FF_RADIAL_CV))

class MeasureObjectRadialDistribution(cpm.CPModule):
 
    module_name = "MeasureObjectRadialDistribution"
    category = "Measurement"
    variable_revision_number = 1
    
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
                "Select an image to measure", "None",doc="""
                What did you call the images you want to process?"""))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove this image", self.images, group))
        self.images.append(group)

    def add_object(self, can_remove = True):
        '''Add an object to be measured (plus optional centers)'''
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))
        group.append("object_name", cps.ObjectNameSubscriber(
                "Select objects to meaasure", "None",doc="""
                What did you call the objects you want to measure?"""))
        group.append("center_choice", cps.Choice(
                "Object to use as center?", C_ALL,doc="""
                There are two ways to specify the center of the radial measurement:
                <ul>
                <li><i>These objects</i>: Use the centers of these objects for the 
                radial measurement.</li> 
                <li><i>Other objects</i>: Use the centers of other objects
                for the radial measurement.</li>
                </ul>
                For example, if measuring the radial distribution in a Cell
                object, you can use the center of the Cell objects (<i>These
                objects</i>) or you can use previously identified Nuclei objects as 
                the centers (<i>Other objects</i>)."""))
        group.append("center_object_name", cps.ObjectNameSubscriber(
                "Select objects to use as centers", "None",doc="""
                Select the object to use as the center, or select <i>None</i> to
                use the input object centers (which is the same as selecting
                <i>These objects</i> for the object centers)."""))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove this object", self.objects, group))
        self.objects.append(group)

    def add_bin_count(self, can_remove = True):
        '''Add another radial bin count at which to measure'''
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))
        group.append("bin_count", cps.Integer(
                    "Number of bins",4, 2, doc="""How many bins do you want to use to measure 
                        the distribution?
                        Radial distribution is measured with respect to a series
                        of concentric rings starting from the object center (or 
                        more generally, between contours at a normalized distance
                        from the object center). This number
                        specifies the number of rings into which the distribution is to
                        be divided. Additional ring counts can be specified
                        by clicking the <i>Add another set of bins</i> button."""))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove this set of bins", self.bin_counts, group))
        self.bin_counts.append(group)
    
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
            result += settings.visible_settings()
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
        stats = [("Image","Objects","Bin #","Bin count","Fraction","Intensity","COV")]
        d = {}
        for image in self.images:
            for o in self.objects:
                for bin_count in self.bin_counts:
                    stats += \
                    self.do_measurements(workspace,
                                         image.image_name.value,
                                         o.object_name.value,
                                         o.center_object_name.value
                                         if o.center_choice == C_OTHER
                                         else None,
                                         bin_count.bin_count.value,
                                         d)
        if workspace.frame is not None:
            images = [d[key][0] for key in d.keys()]
            names = d.keys()
            figure = workspace.create_or_find_figure(subplots=(1,len(images)))
            figure.figure.clf()
            nimages = len(images)
            shrink = .05
            for i in range(len(images)):
                rect = [shrink + float(i)/float(nimages),
                        .5+shrink,
                        (1.0-2*shrink)/float(nimages),
                        .45*(1.0-2*shrink)]
                axes = figure.figure.add_axes(rect)
                axes.imshow(images[i], matplotlib.cm.Greys_r)
                axes.set_title(names[i],
                               fontname=cpprefs.get_title_font_name(),
                               fontsize=cpprefs.get_title_font_size())
            rect = [0.1,.1,.8,.35]
            axes = figure.figure.add_axes(rect, frameon = False)
            table = axes.table(cellText=stats,
                               colWidths=[1.0/7.0]*7,
                               loc='center',
                               cellLoc='left')
            axes.set_axis_off()
            table.auto_set_font_size(False)
            table.set_fontsize(cpprefs.get_table_font_size())
            
    def do_measurements(self, workspace, image_name, object_name, 
                        center_object_name, bin_count,
                        dd):
        '''Perform the radial measurements on the image set
        
        workspace - workspace that holds images / objects
        image_name - make measurements on this image
        object_name - make measurements on these objects
        center_object_name - use the centers of these related objects as
                      the centers for radial measurements. None to use the
                      objects themselves.
        bin_count - bin the object into this many concentric rings
        d - a dictionary for saving reusable partial results
        
        returns one statistics tuple per ring.
        '''
        assert isinstance(workspace, cpw.Workspace)
        assert isinstance(workspace.object_set, cpo.ObjectSet)
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale=True)
        objects = workspace.object_set.get_objects(object_name)
        nobjects = np.max(objects.segmented)
        measurements = workspace.measurements
        assert isinstance(measurements, cpmeas.Measurements)
        if nobjects == 0:
            for bin in range(1, bin_count+1):
                for feature in (FF_FRAC_AT_D, FF_MEAN_FRAC, FF_RADIAL_CV):
                    measurements.add_measurement(object_name,
                                                 M_CATEGORY + "_" + feature % 
                                                 (image_name, bin, bin_count),
                                                 np.zeros(0))
            return [(image_name, object_name, "no objects","-","-","-","-")]
        name = (object_name if center_object_name is None 
                else "%s_%s"%(object_name, center_object_name))
        if dd.has_key(name):
            normalized_distance, i_center, j_center, good_mask = dd[name]
        else:
            d_to_edge = distance_to_edge(objects.segmented)
            if center_object_name is not None:
                center_objects=workspace.object_set.get_objects(center_object_name)
                center_labels = center_objects.segmented
                pixel_counts = fix(scind.sum(np.ones(center_labels.shape),
                                             center_labels,
                                             np.arange(1, np.max(center_labels)+1)))
                good = pixel_counts > 0
                i,j = (centers_of_labels(center_labels) + .5).astype(int)
                ig = i[good]
                jg = j[good]
                center_labels = np.zeros(center_labels.shape, int)
                center_labels[ig,jg] = objects.segmented[ig,jg]
                cl,d_from_center = propagate(np.zeros(center_labels.shape),
                                             center_labels,
                                             objects.segmented != 0, 1)
            else:
                # Find the point in each object farthest away from the edge.
                # This does better than the centroid:
                # * The center is within the object
                # * The center tends to be an interesting point, like the
                #   center of the nucleus or the center of one or the other
                #   of two touching cells.
                #
                i,j = maximum_position_of_labels(d_to_edge, objects.segmented)
                center_labels = np.zeros(objects.segmented.shape, int)
                center_labels[i,j] = objects.segmented[i,j]
                #
                # Use the coloring trick here to process touching objects
                # in separate operations
                #
                colors = color_labels(objects.segmented)
                ncolors = np.max(colors)
                d_from_center = np.zeros(objects.segmented.shape)
                cl = np.zeros(objects.segmented.shape, int)
                for color in range(1,ncolors+1):
                    mask = colors == color
                    l,d = propagate(np.zeros(center_labels.shape),
                                    center_labels,
                                    mask, 1)
                    d_from_center[mask] = d[mask]
                    cl[mask] = l[mask]
            good_mask = cl > 0
            i_center = np.zeros(cl.shape)
            i_center[good_mask] = i[cl[good_mask]-1]
            j_center = np.zeros(cl.shape)
            j_center[good_mask] = j[cl[good_mask]-1]
            
            normalized_distance = np.zeros(objects.segmented.shape)
            total_distance = d_from_center + d_to_edge
            normalized_distance[good_mask] = (d_from_center[good_mask] /
                                              (total_distance[good_mask] + .001))
            dd[name] = [normalized_distance, i_center, j_center, good_mask]
        ngood_pixels = np.sum(good_mask)
        good_labels = objects.segmented[good_mask]
        bin_indexes = (normalized_distance * bin_count).astype(int)
        labels_and_bins = (good_labels-1,bin_indexes[good_mask])
        histogram = coo_matrix((image.pixel_data[good_mask], labels_and_bins),
                               (nobjects, bin_count)).toarray()
        sum_by_object = np.sum(histogram, 1)
        sum_by_object_per_bin = np.dstack([sum_by_object]*bin_count)[0]
        fraction_at_distance = histogram / sum_by_object_per_bin
        number_at_distance = coo_matrix((np.ones(ngood_pixels),labels_and_bins),
                                        (nobjects, bin_count)).toarray()
        object_mask = number_at_distance > 0
        sum_by_object = np.sum(number_at_distance, 1)
        sum_by_object_per_bin = np.dstack([sum_by_object]*bin_count)[0]
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
        i,j = np.mgrid[0:objects.segmented.shape[0], 
                       0:objects.segmented.shape[1]]
        imask = i[good_mask] > i_center[good_mask]
        jmask = j[good_mask] > j_center[good_mask]
        absmask = (abs(i[good_mask] - i_center[good_mask]) > 
                   abs(j[good_mask] - j_center[good_mask]))
        radial_index = (imask.astype(int) + jmask.astype(int)*2 + 
                        absmask.astype(int)*4)
        statistics = []
        for bin in range(bin_count):
            bin_mask = (good_mask & (bin_indexes == bin))
            bin_pixels = np.sum(bin_mask)
            bin_labels = objects.segmented[bin_mask]
            bin_radial_index = radial_index[bin_indexes[good_mask] == bin]
            labels_and_radii = (bin_labels-1, bin_radial_index)
            radial_values = coo_matrix((image.pixel_data[bin_mask],
                                        labels_and_radii),
                                       (nobjects, 8)).toarray()
            pixel_count = coo_matrix((np.ones(bin_pixels), labels_and_radii),
                                     (nobjects, 8)).toarray()
            mask = pixel_count==0
            radial_means = masked_array(radial_values / pixel_count, mask)
            radial_cv = np.std(radial_means,1) / np.mean(radial_means, 1)
            radial_cv[np.sum(~mask,1)==0] = 0
            for measurement, feature in ((fraction_at_distance[:,bin], MF_FRAC_AT_D),
                                         (mean_pixel_fraction[:,bin], MF_MEAN_FRAC),
                                         (np.array(radial_cv), MF_RADIAL_CV)):
                                         
                measurements.add_measurement(object_name,
                                             feature % 
                                             (image_name, bin+1, bin_count),
                                             measurement)
            radial_cv.mask = np.sum(~mask,1)==0
            statistics += [(image_name, object_name, str(bin+1), str(bin_count),
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
                    for feature in (MF_FRAC_AT_D, MF_MEAN_FRAC, MF_RADIAL_CV):
                        for bin in range(1,bin_count+1):
                            columns.append((o.object_name.value,
                                            feature % (image.image_name.value,
                                                       bin, bin_count),
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
            return [FF_SCALE % (bin,bin_count.bin_count.value)
                    for bin_count in self.bin_counts
                    for bin in range(1, bin_count.bin_count.value+1)]
        return []
            
    def upgrade_settings(self,setting_values,variable_revision_number,
                         module_name,from_matlab):
        if from_matlab and variable_revision_number == 1:
            image_name, object_name, center_name, bin_count = setting_values[:4]
            if center_name == cps.DO_NOT_USE:
                center_choice = C_SELF
            else:
                center_choice = C_OTHER
            setting_values = ["1","1","1",image_name, 
                              object_name, center_choice, center_name,
                              bin_count]
            variable_revision_number = 1
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab
    
