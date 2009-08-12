'''measureobjectradialdistribution.py - Measure object's radial distribution

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import numpy as np
from numpy.ma import masked_array
from scipy.sparse import coo_matrix
import uuid

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw
import cellprofiler.gui.cpfigure as cpf
import relate as R

from cellprofiler.cpmath.cpmorphology import distance_to_edge
from cellprofiler.cpmath.cpmorphology import centers_of_labels

C_SELF = 'These objects'
C_OTHER = 'Other objects'
C_ALL = [C_SELF, C_OTHER]

M_CATEGORY = 'RadialDistribution'
FF_FRAC_AT_D = 'FracAtD_%s_%dOf%d'
FF_MEAN_FRAC = 'MeanFrac_%s_%dOf%d'
FF_RADIAL_CV = 'RadialCV_%s_%dOf%d'
MF_FRAC_AT_D = '_'.join((M_CATEGORY,FF_FRAC_AT_D))
MF_MEAN_FRAC = '_'.join((M_CATEGORY,FF_MEAN_FRAC))
MF_RADIAL_CV = '_'.join((M_CATEGORY,FF_RADIAL_CV))

class MeasureObjectRadialDistribution(cpm.CPModule):
    '''SHORT DESCRIPTION:
Measures radial distribution of one or more proteins within a cell.
*************************************************************************

Given an image with objects identified, this module measures the
intensity distribution from the center of those objects to their
boundary within a user-controlled number of bins, for each object.

The distribution can be measured within a single identified object,
in which case it is relative to the "center" of the object (as
defined as the point farthest from the boundary), or another object
can be used as the center, an example of which would be using Nuclei
for centers within Cells.

Three features are measured for each object:
- FracAtD: Fraction of total stain in an object at a given radius.
- MeanFrac: Mean fractional intensity at a given radius (Fraction of total 
   intenstiy normalized by fraction of pixels at a given radius).
- RadialCV: Coefficient of variation of intensity within a ring, calculated 
  over 8 slices.
'''
    category = "Measurement"
    variable_revision_number = 1
    
    def create_settings(self):
        self.module_name = "MeasureObjectRadialDistribution"
        self.images = []
        self.objects = []
        self.bin_counts = []
        self.image_count = cps.HiddenCount(self.images)
        self.object_count = cps.HiddenCount(self.objects)
        self.bin_counts_count = cps.HiddenCount(self.bin_counts)
        self.add_image_button = cps.DoSomething("Add another image", "Add", 
                                                self.add_image)
        self.add_object_button = cps.DoSomething("Add another object", "Add",
                                                 self.add_object)
        self.add_bin_count_button = cps.DoSomething("Add another bin count",
                                                    "Add", self.add_bin_count)
        self.add_image(can_remove = False)
        self.add_object(can_remove = False)
        self.add_bin_count(can_remove = False)
    
    def add_image(self, can_remove = True):
        '''Add an image to be measured'''
        class ImageSettings(object):
            '''Settings describing an image to be measured'''
            def __init__(self, images):
                self.key = uuid.uuid4()
                self.image_name = cps.ImageNameSubscriber(
                    "What did you call the image from which you want to "
                    "measure the intensity distribution?", "None")
                if can_remove:
                    def remove(images=images, key = self.key):
                        index = [x.key for x in images].index(key)
                        del images[index]
                    self.remove_button = cps.DoSomething("Remove above image",
                                                         "Remove", remove)
            def settings(self):
                '''Return the settings that should be saved in the pipeline'''
                return [self.image_name]
            
            def visible_settings(self):
                '''Return the settings that should be displayed'''
                if can_remove:
                    return [self.image_name, self.remove_button]
                else:
                    return [self.image_name]
        self.images.append(ImageSettings(self.images))
    
    def add_object(self, can_remove = True):
        '''Add an object to be measured (plus optional centers)'''
        class ObjectSettings(object):
            '''Settings describing an object to be measured'''
            def __init__(self, objects):
                self.key = uuid.uuid4()
                self.object_name = cps.ObjectNameSubscriber(
                    "What did you call the objects from which you want "
                    "to measure the intensity distribution?", "None")
                self.center_choice = cps.Choice(
                    "Do you want to measure distribution from the center "
                    "of these objects or some other objects?", C_ALL)
                self.center_object_name = cps.ObjectNameSubscriber(
                    "What objects do you want to use as centers?", "None")
                if can_remove:
                    def remove(objects = objects, key = self.key):
                        index = [x.key for x in objects].index(key)
                        del objects[index]
                    self.remove_button = cps.DoSomething("Remove above object",
                                                         "Remove", remove)
            
            def settings(self):
                '''Return the settings that should be saved in the pipeline'''
                return [self.object_name, self.center_choice, 
                        self.center_object_name]
            
            def visible_settings(self):
                '''Return the settings that should be displayed'''
                result = [self.object_name, self.center_choice]
                if self.center_choice == C_OTHER:
                    result += [self.center_choice]
                if can_remove:
                    result += [self.remove_button]
                return result
        self.objects.append(ObjectSettings(self.objects))
    
    def add_bin_count(self, can_remove = True):
        '''Add another radial bin count at which to measure'''
        class BinCountSettings(object):
            '''Settings describing the number of radial bins'''
            def __init__(self, bin_counts):
                self.key = uuid.uuid4()
                self.bin_count = cps.Integer(
                    "How many bins do you want to use to store "
                        "the distribution?",4, 2)
                if can_remove:
                    def remove(bin_counts = bin_counts, key = self.key):
                        index = [x.key for x in bin_counts].index(key)
                        del bin_counts[index]
                    self.remove_button = cps.DoSomething("Remove above bin count",
                                                         "Remove", remove)
            def settings(self):
                '''Return the settings that should be saved in the pipeline'''
                return [self.bin_count]
            
            def visible_settings(self):
                '''Return the settings that should be displayed'''
                if can_remove:
                    return [self.bin_count, self.remove_button]
                else:
                    return [self.bin_count]
        self.bin_counts.append(BinCountSettings(self.bin_counts))
    
    def settings(self):
        result = [self.image_count, self.object_count, self.bin_counts_count]
        for x in (self.images, self.objects, self.bin_counts):
            for settings in x:
                result += settings.settings()
        return result
    
    def visible_settings(self):
        result = []
        for setting_list, add_button in ((self.images, self.add_image_button),
                                         (self.objects, self.add_object_button),
                                         (self.bin_counts, self.add_bin_count_button)):
            for settings in setting_list:
                result += settings.visible_settings()
            result += [add_button]
        return result
    
    def prepare_to_set_values(self, setting_values):
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
    
    def backwards_compatibilize(self,setting_values,variable_revision_number,
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
    
    def run(self, workspace):
        stats = [("Image","Objects","Bin #","Bin count","Fraction","Intensity","COV")]
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
                                         bin_count.bin_count.value)
        if workspace.frame is not None:
            figure = workspace.create_or_find_figure(subplots=(1,1))
            assert isinstance(figure, cpf.CPFigureFrame)
            figure.subplot_table(0,0, stats, [1.0/7.0]*7)
    
    def do_measurements(self, workspace, image_name, object_name, 
                        center_object_name, bin_count):
        '''Perform the radial measurements on the image set
        
        workspace - workspace that holds images / objects
        image_name - make measurements on this image
        object_name - make measurements on these objects
        center_object_name - use the centers of these related objects as
                      the centers for radial measurements. None to use the
                      objects themselves.
        bin_count - bin the object into this many concentric rings
        
        returns one statistics tuple per ring.
        '''
        assert isinstance(workspace, cpw.Workspace)
        assert isinstance(workspace.object_set, cpo.ObjectSet)
        assert isinstance(workspace.measurements, cpmeas.Measurements)
        image = workspace.image_set.get_image(image_name)
        objects = workspace.object_set.get_objects(object_name)
        nobjects = np.max(objects.segmented)
        measurements = workspace.measurements
        d_to_edge = distance_to_edge(objects.segmented)
        if center_object_name is not None:
            center_objects=workspace.object_set.get_objects(center_object_name)
            ncenters = np.max(center_objects.segmented)
            #
            # By default, the labels on objects and centers match
            # We try to use measurements from relate or identify to find
            # the best center for an object.
            #
            object_to_center = np.arange(1,nobjects+1)
            if measurements.has_current_measurements(
                object_name, R.FF_PARENT % center_object_name):
                object_to_center = measurements.get_current_measurement(
                    object_name, R.FF_PARENT % center_object_name)
            elif measurements.has_current_measurements(
                center_object_name, RR.FF_PARENT % object_name):
                # The reverse is less than ideal. Some objects might have
                # more than one center and some objects might not have
                # a center.
                center_to_object = measurements.get_current_measurement(
                    center_object_name, R.FF_PARENT % object_name)
                object_to_center[center_to_object] = np.arange(1, ncenters+1)
            
            good = object_to_center > 0
            center_centers = centers_of_labels(center_objects.segmented)
            centers = np.zeros((2,nobjects))
            centers[good] = center_centers[object_to_center[good]-1]
        else:
            good = np.array(nobjects, bool)
            centers = centers_of_labels(objects.segmented)
        good0 = np.zeros(nobjects+1, bool)
        good0[1:] = good
        d_from_center = np.zeros(objects.segmented.shape)
        good_mask = good0[objects.segmented]
        ngood_pixels = np.sum(good_mask)
        i,j = np.mgrid[0:objects.segmented.shape[0], 
                       0:objects.segmented.shape[1]]
        good_labels = objects.segmented[good_mask]
        pt_centers = centers[:,good_labels-1]
        d_from_center[good_mask] = np.sqrt((i[good_mask]-pt_centers[0,:])**2 +
                                           (j[good_mask]-pt_centers[1,:])**2)
        normalized_distance = np.zeros(objects.segmented.shape)
        total_distance = d_from_center + d_to_edge
        normalized_distance[good_mask] = (d_from_center[good_mask] /
                                          (total_distance[good_mask] + .001))
        bin_indexes = (normalized_distance * bin_count).astype(int)
        histogram = coo_matrix((image.pixel_data[good_mask],
                                (good_labels,
                                 bin_indexes[good_mask])),
                               (nobjects, bin_count)).toarray()
        sum_by_object = np.sum(histogram, 1)
        fraction_at_distance = histogram / np.tile(sum_by_object, (bin_count,1))
        number_at_distance = coo_matrix((np.ones(ngood_pixels),
                                         (good_labels,
                                          bin_indexes[good_mask])),
                                        (nobjects, bin_count)).toarray()
        sum_by_object = np.sum(number_at_distance, 1)
        fraction_at_bin = number_at_distance / np.tile(sum_by_object,
                                                       (bin_count,1))
        mean_pixel_fraction = fraction_at_distance / (fraction_at_bin +
                                                      np.finfo(float).eps)
        # Anisotropy calculation.  Split each cell into eight wedges, then
        # compute coefficient of variation of the wedges' mean intensities
        # in each ring.
        #
        # Compute each pixel's delta from the center object's centroid
        imask = i[good_mask] > pt_centers[0,:]
        jmask = j[good_mask] > pt_centers[1,:]
        absmask = (i[good_mask] - pt_centers[0,:] > 
                   j[good_mask] - pt_centers[1,:])
        radial_index = (imask.astype(int) + jmask.astype(int)*2 + 
                        absmask.astype(int)*4)
        statistics = []
        for bin in range(bin_count):
            bin_mask = (good_mask & (bin_indexes == bin))
            bin_pixels = np.sum(bin_mask)
            bin_labels = objects.segmented[bin_mask]
            bin_radial_index = radial_index[bin_indexes[good_mask] == bin]
            radial_values = coo_matrix((image.pixel_data[bin_mask],
                                        (bin_labels, bin_radial_index)),
                                       (nobjects, 8)).toarray()
            pixel_count = coo_matrix((np.ones(bin_pixels),
                                      (bin_labels, bin_radial_index))).toarray()
            mask = pixel_count==0
            radial_means = masked_array(radial_values / pixel_count, mask)
            radial_cv = np.std(radial_means,1) / np.mean(radial_means, 1)
            radial_cv[np.sum(~mask,1)==0] = 0
            for measurement, feature in ((fraction_at_distance[1:], MF_FRAC_AT_D),
                                         (mean_pixel_fraction[1:], MF_MEAN_FRAC),
                                         (radial_cv[1:], MF_RADIAL_CV)):
                                         
                measurements.add_measurement(object_name,
                                             feature % 
                                             (image_name, bin, bin_count),
                                             measurement)
            statistics += [image_name, object_name, str(bin), str(bin_count),
                           np.mean(fraction_at_distance[good0,bin]),
                           np.mean(mean_pixel_fraction[good0, bin]),
                           np.mean(radial_cv[~mask])]
        return statistics
                           