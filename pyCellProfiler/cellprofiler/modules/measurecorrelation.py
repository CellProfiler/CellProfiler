'''measure_correlation.py - Measure correlation between two images

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import numpy as np
from scipy.linalg import lstsq
import scipy.ndimage as scind
import uuid

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix

M_IMAGES = "Images"
M_OBJECTS = "Objects"
M_IMAGES_AND_OBJECTS = "Images and objects"
M_ALL = [M_IMAGES, M_OBJECTS, M_IMAGES_AND_OBJECTS]

'''Feature name format for the correlation measurement'''
F_CORRELATION_FORMAT = "Correlation_Correlation_%s_%s"

'''Feature name format for the slope measurement'''
F_SLOPE_FORMAT = "Correlation_Slope_%s_%s"

class MeasureCorrelation(cpm.CPModule):
    '''SHORT DESCRIPTION:
    Measures the correlation between intensities in different images (e.g.
    different color channels) on a pixel by pixel basis, within identified
    objects or across an entire image.
    *************************************************************************
    
    Given two or more images, calculates the correlation between the
    pixel intensities. The correlation can be measured for the entire
    images, or individual correlation measurements can be made within each
    individual object. For example:
                                         Image overall:  In Nuclei:
    OrigBlue_OrigGreen    Correlation:    0.49955        -0.07395
    OrigBlue_OrigRed      Correlation:    0.59886        -0.02752
    OrigGreen_OrigRed     Correlation:    0.83605         0.68489
    
    '''

    category = 'Measurement'
    variable_revision_number = 1
    
    def create_settings(self):
        '''Create the initial settings for the module'''
        self.module_name = 'MeasureCorrelation'
        self.image_groups = []
        self.add_image(can_delete = False)
        self.add_image(can_delete = False)
        self.image_count = cps.HiddenCount(self.image_groups)
        self.add_image_button = cps.DoSomething('Add another image','Add image',
                                                self.add_image)
        self.images_or_objects = cps.Choice('Do you want to measure the correlation within objects, over the whole image or both within objects and over the whole image?',
                                            M_ALL)
        self.object_groups = []
        self.add_object(can_delete = False)
        self.object_count = cps.HiddenCount(self.object_groups)
        self.add_object_button = cps.DoSomething('Add another object','Add object',
                                                 self.add_object)
        
    def add_image(self, can_delete = True):
        '''Add an image to the image_groups collection
        
        can_delete - set this to False to keep from showing the "remove"
                     button for images that must be present.
        '''
        class ImageSettings(object):
            def __init__(self, image_groups, can_delete):
                self.can_delete = can_delete
                self.image_name = cps.ImageNameSubscriber('What is the name of the image to be measured?','None')
                if self.can_delete:
                    self.key = uuid.uuid4()
                    def remove(key = self.key, image_groups = image_groups):
                        index = [x.key for x in image_groups].index(key)
                        del image_groups[index]
                    self.remove_button = cps.DoSomething('Remove this image',
                                                         'Remove', remove)
            
            def settings(self):
                return [self.image_name]
            
            def visible_settings(self):
                return ([self.image_name] + 
                        ([self.remove_button] if self.can_delete else []))
        self.image_groups.append(ImageSettings(self.image_groups, can_delete))
        
    def add_object(self, can_delete = True):
        '''Add an object to the object_groups collection'''
        class ObjectSettings(object):
            def __init__(self, object_groups, can_delete):
                self.can_delete = can_delete
                self.object_name = cps.ObjectNameSubscriber('What is the name of the objects to be measured?','None')
                if self.can_delete:
                    self.key = uuid.uuid4()
                    def remove(key = self.key, object_groups = object_groups):
                        index = [x.key for x in object_groups].index(key)
                        del object_groups[index]
                    self.remove_button = cps.DoSomething('Remove this object',
                                                         'Remove', remove)
            
            def settings(self):
                return [self.object_name]
            
            def visible_settings(self):
                return ([self.object_name] + 
                        ([self.remove_button] if self.can_delete else []))
        self.object_groups.append(ObjectSettings(self.object_groups, can_delete))
            
    def settings(self):
        '''Return the settings to be saved in the pipeline'''
        result = [self.image_count, self.object_count]
        for image_group in self.image_groups:
            result += image_group.settings()
        result += [self.images_or_objects]
        for object_group in self.object_groups:
            result += object_group.settings()
        return result

    def prepare_to_set_values(self, setting_values):
        '''Make sure there are enough image and object slots for the incoming settings'''
        image_count = int(setting_values[0])
        object_count = int(setting_values[1])
        if image_count < 2:
            raise ValueError("The MeasureCorrelate module must have at least two input images. %d found in pipeline file"%image_count)
        if object_count < 1:
            raise ValueError("Inconsistent state in MeasureCorrelate module. Pipeline file must have at least one object, none found")
        while len(self.image_groups) > image_count:
            del self.image_groups[image_count]
        while len(self.image_groups) < image_count:
            self.add_image()
        
        while len(self.object_groups) > object_count:
            del self.object_groups[object_count]
        while len(self.object_groups) < object_count:
            self.add_object()

    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        '''Adjust the setting values for pipelines saved under old revisions'''
        if from_matlab and variable_revision_number == 3:
            image_names = [x for x in setting_values[:4]
                           if x.upper() != cps.DO_NOT_USE.upper()]
            wants_image_measured = np.any([x==cpmeas.IMAGE 
                                           for x in setting_values[4:]])
            object_names = [x for x in setting_values[4:]
                            if not x in (cpmeas.IMAGE, cps.DO_NOT_USE)]
            if wants_image_measured:
                if len(object_names):
                    m = M_IMAGES_AND_OBJECTS
                else:
                    m = M_IMAGES
            elif len(object_names):
                m = M_OBJECTS
            else:
                raise ValueError("Must either measure texture over images or over some set of objects")
            if len(object_names) == 0:
                object_names = ['None']
            setting_values = ([str(len(image_names)), str(len(object_names))] +
                              image_names + [m] + object_names)
            from_matlab = False
            variable_revision_number = 1
        return setting_values, variable_revision_number, from_matlab

    def visible_settings(self):
        result = []
        for image_group in self.image_groups:
            result += image_group.visible_settings()
        result += [self.add_image_button, self.images_or_objects]
        if self.wants_objects:
            for object_group in self.object_groups:
                result += object_group.visible_settings()
            result += [self.add_object_button]
        return result

    def run(self, workspace):
        '''Calculate measurements on an image set'''
        statistics = [["First image","Second image","Objects","Measurement","Value"]]
        for first_name, second_name in self.get_image_pairs():
                statistics += self.run_image_pair(workspace, 
                                                  first_name, 
                                                  second_name)
        if not workspace.frame is None:
            figure = workspace.create_or_find_figure(subplots=(1,1))
            figure.subplot_table(0,0,statistics,(0.2,0.2,0.2,0.2,0.2))
    
    def get_image_pairs(self):
        '''Yield all permutations of pairs of images to correlate
        
        Yields the pairs of images in a canonical order.
        '''
        for i in range(self.image_count.value-1):
            for j in range(i+1, self.image_count.value):
                yield (self.image_groups[i].image_name.value,
                       self.image_groups[j].image_name.value)
    @property
    def wants_images(self):
        '''True if the user wants to measure correlation on whole images'''
        return self.images_or_objects in (M_IMAGES, M_IMAGES_AND_OBJECTS)
    
    @property
    def wants_objects(self):
        '''True if the user wants to measure per-object correlations'''
        return self.images_or_objects in (M_OBJECTS, M_IMAGES_AND_OBJECTS)
    
    def run_image_pair(self, workspace, first_image_name, second_image_name):
        '''Calculate the correlation between two images'''
        statistics = []
        if self.wants_images:
            statistics += self.run_image_pair_image(workspace, 
                                                    first_image_name, 
                                                    second_image_name)
        if self.wants_objects:
            for i in range(self.object_count.value):
                object_name = self.object_groups[i].object_name.value
                statistics += self.run_image_pair_objects(workspace, 
                                                          first_image_name,
                                                          second_image_name, 
                                                          object_name)
        return statistics
    
    def run_image_pair_image(self, workspace, first_image_name, 
                             second_image_name):
        '''Calculate the correlation between the pixels of two images'''
        first_image = workspace.image_set.get_image(first_image_name,
                                                    must_be_grayscale=True)
        second_image = workspace.image_set.get_image(second_image_name,
                                                     must_be_grayscale=True)
        first_pixel_data = first_image.pixel_data
        first_mask = first_image.mask
        first_pixel_count = np.product(first_pixel_data.shape)
        second_pixel_data = second_image.pixel_data
        second_mask = second_image.mask
        second_pixel_count = np.product(second_pixel_data.shape)
        #
        # Crop the larger image similarly to the smaller one
        #
        if first_pixel_count < second_pixel_count:
            second_pixel_data = first_image.crop_image_similarly(second_pixel_data)
            second_mask = first_image.crop_image_similarly(second_mask)
        elif second_pixel_count < first_pixel_count:
            first_pixel_data = second_image.crop_image_similarly(first_pixel_data)
            first_mask = second_image.crop_image_similarly(first_mask)
        mask = (first_mask & second_mask)
        #
        # Perform the correlation, which returns:
        # [ [ii, ij],
        #   [ji, jj] ]
        #
        fi = first_pixel_data[mask]
        si = second_pixel_data[mask]
        corr = np.corrcoef((fi,si))[1,0]
        #
        # Find the slope as a linear regression to
        # A * i1 + B = i2
        #
        coeffs = lstsq(np.array((fi,np.ones_like(fi))).transpose(),si)[0]
        slope = coeffs[0]
        #
        # Add the measurements
        #
        corr_measurement = F_CORRELATION_FORMAT%(first_image_name, 
                                                 second_image_name)
        m = workspace.measurements
        m.add_image_measurement(corr_measurement, corr)
        slope_measurement = F_SLOPE_FORMAT%(first_image_name,
                                            second_image_name)
        m.add_image_measurement(slope_measurement, slope)
        return [[first_image_name, second_image_name,"-",
                 "Correlation","%.2f"%corr],
                [first_image_name, second_image_name,"-",
                 "Slope","%.2f"%slope]]
        
    def run_image_pair_objects(self, workspace, first_image_name,
                               second_image_name, object_name):
        '''Calculate per-object correlations between intensities in two images'''
        first_image = workspace.image_set.get_image(first_image_name,
                                                    must_be_grayscale=True)
        second_image = workspace.image_set.get_image(second_image_name,
                                                     must_be_grayscale=True)
        objects = workspace.object_set.get_objects(object_name)
        #
        # Crop both images to the size of the labels matrix
        #
        first_pixels  = objects.crop_image_similarly(first_image.pixel_data)
        first_mask    = objects.crop_image_similarly(first_image.mask)
        second_pixels = objects.crop_image_similarly(second_image.pixel_data)
        second_mask   = objects.crop_image_similarly(second_image.mask)
        labels = objects.segmented
        mask   = ((labels > 0) & first_mask & second_mask)
        first_pixels = first_pixels[mask]
        second_pixels = second_pixels[mask]
        labels = labels[mask]
        n_objects = np.max(labels)
        if n_objects == 0:
            corr = np.zeros((0,))
        else:
            #
            # The correlation is sum((x-mean(x))(y-mean(y)) /
            #                         ((n-1) * std(x) *std(y)))
            #
            lrange = np.arange(n_objects)+1
            area  = fix(scind.sum(np.ones_like(labels), labels, lrange))
            mean1 = fix(scind.mean(first_pixels, labels, lrange))
            mean2 = fix(scind.mean(second_pixels, labels, lrange))
            std1 = fix(scind.standard_deviation(first_pixels, labels, lrange))
            std2 = fix(scind.standard_deviation(second_pixels, labels, lrange))
            x = first_pixels - mean1[labels-1]  # x - mean(x)
            y = second_pixels - mean2[labels-1] # y - mean(y)
            corr = fix(scind.sum(x * y / 
                                 (std1[labels-1] * std2[labels-1] * 
                                  (area[labels-1]-1)),
                                 labels, lrange))
            corr[~ np.isfinite(corr)] = 0
        measurement = ("Correlation_Correlation_%s_%s" %
                       (first_image_name, second_image_name))
        workspace.measurements.add_measurement(object_name, measurement, corr)
        return [[first_image_name, second_image_name, object_name,
                 "Mean correlation","%.2f"%np.mean(corr)],
                [first_image_name, second_image_name, object_name,
                 "Median correlation","%.2f"%np.median(corr)],
                [first_image_name, second_image_name, object_name,
                 "Min correlation","%.2f"%np.min(corr)],
                [first_image_name, second_image_name, object_name,
                 "Max correlation","%.2f"%np.max(corr)]]
             
    def get_measurement_columns(self, pipeline):
        '''Return column definitions for all measurements made by this module'''
        columns = []
        for first_image, second_image in self.get_image_pairs():
            if self.wants_images:
                columns += [(cpmeas.IMAGE,
                             F_CORRELATION_FORMAT%(first_image, second_image),
                             cpmeas.COLTYPE_FLOAT),
                            (cpmeas.IMAGE,
                             F_SLOPE_FORMAT%(first_image, second_image),
                             cpmeas.COLTYPE_FLOAT)]
            if self.wants_objects:
                for i in range(self.object_count.value):
                    object_name = self.object_groups[i].object_name.value
                    columns += [(object_name,
                                 F_CORRELATION_FORMAT %
                                 (first_image, second_image),
                                 cpmeas.COLTYPE_FLOAT)]
        return columns

    def get_categories(self, pipeline, object_name):
        '''Return the categories supported by this module for the given object
        
        object_name - name of the measured object or "Image"
        '''
        if ((object_name == cpmeas.IMAGE and self.wants_images) or
            (object_name != cpmeas.IMAGE and self.wants_objects and
             object_name in [x.object_name.value for x in self.object_groups])):
            return ["Correlation"]
        return [] 

    def get_measurements(self, pipeline, object_name, category):
        if self.get_categories(pipeline, object_name) == ["Correlation"]:
            if object_name == cpmeas.IMAGE:
                return ["Correlation","Slope"]
            else:
                return ["Correlation"]
        return []

    def get_measurement_images(self, pipeline, object_name, category, 
                               measurement):
        '''Return the joined pairs of images measured'''
        if measurement in self.get_measurements(pipeline, object_name, category):
            return ["%s_%s"%x for x in self.get_image_pairs()]
        return []
