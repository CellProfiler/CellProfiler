'''<b>Measure Correlation</b> measures the correlation between intensities in different images (e.g.,
different color channels) on a pixel-by-pixel basis, within identified
objects or across an entire image.
<hr>
Given two or more images, this module calculates the correlation between the
pixel intensities. The correlation can be measured for entire
images, or a correlation measurement can be made within each
individual object.

Correlations will be calculated between all pairs of images that are selected in 
the module, as well as between selected objects. For example, if correlations 
are to be measured for a set of red, green, and blue images containing identified nuclei, 
measurements will be made between the following:
<ul>
<li>The blue and green, red and green, and red and blue images. </li>
<li>The nuclei in each of the above image pairs.</li>
</ul>

<h4>Available measurements</h4>
<ul>
<li><i>Correlation coefficient:</i> The correlation between a pair of images <i>I</i> and <i>J</i>. 
Calculated as Pearson's correlation coefficient, for which the formula is
covariance(<i>I</i> ,<i>J</i>)/[std(<i>I</i> ) &times; std(<i>J</i>)].</li>
<li><i>Slope:</i> The slope of the least-squares regression between a pair of images
I and J. Calculated using the model <i>A</i> &times; <i>I</i> + <i>B</i> = <i>J</i>, w
here <i>A</i> is the slope.</li>
</ul>

'''
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
from scipy.linalg import lstsq
import scipy.ndimage as scind

import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix

M_IMAGES = "Across entire image"
M_OBJECTS = "Within objects"
M_IMAGES_AND_OBJECTS = "Both"

'''Feature name format for the correlation measurement'''
F_CORRELATION_FORMAT = "Correlation_Correlation_%s_%s"

'''Feature name format for the slope measurement'''
F_SLOPE_FORMAT = "Correlation_Slope_%s_%s"

class MeasureCorrelation(cpm.CPModule):

    module_name = 'MeasureCorrelation'
    category = 'Measurement'
    variable_revision_number = 2
    
    def create_settings(self):
        '''Create the initial settings for the module'''
        self.image_groups = []
        self.add_image(can_delete = False)
        self.spacer_1 = cps.Divider()
        self.add_image(can_delete = False)
        self.image_count = cps.HiddenCount(self.image_groups)
        
        self.add_image_button = cps.DoSomething("", 'Add another image', self.add_image)
        self.spacer_2 = cps.Divider()
        self.images_or_objects = cps.Choice(
            'Select where to measure correlation',
            [M_IMAGES, M_OBJECTS, M_IMAGES_AND_OBJECTS], doc = '''
            You can measure the correlation in several ways: 
            <ul>
            <li><i>%(M_OBJECTS)s:</i> Measure correlation only in those pixels previously
            identified as an object. You will be asked to specify which object to measure from.</li>
            <li><i>%(M_IMAGES)s:</i> Measure the correlation across all pixels in the images.</li>
            <li><i>%(M_IMAGES_AND_OBJECTS)s:</i> Calculate both measurements above.</li>
            </ul>
            All methods measure correlation on a pixel by pixel basis.'''%globals())
        
        self.object_groups = []
        self.add_object(can_delete = False)
        self.object_count = cps.HiddenCount(self.object_groups)
        
        self.spacer_2 = cps.Divider(line=True)
        
        self.add_object_button = cps.DoSomething("", 'Add another object', self.add_object)

    def add_image(self, can_delete = True):
        '''Add an image to the image_groups collection
        
        can_delete - set this to False to keep from showing the "remove"
                     button for images that must be present.
        '''
        group = cps.SettingsGroup()
        if can_delete:
            group.append("divider", cps.Divider(line=False))
        group.append("image_name", cps.ImageNameSubscriber(
            'Select an image to measure',cps.NONE,doc = '''
            Select an image to measure the correlation from.'''))
        
        if len(self.image_groups) == 0: # Insert space between 1st two images for aesthetics
            group.append("extra_divider", cps.Divider(line=False))
        
        if can_delete:
            group.append("remover", cps.RemoveSettingButton("","Remove this image", self.image_groups, group))
            
        self.image_groups.append(group)

    def add_object(self, can_delete = True):
        '''Add an object to the object_groups collection'''
        group = cps.SettingsGroup()
        if can_delete:
            group.append("divider", cps.Divider(line=False))
            
        group.append("object_name", cps.ObjectNameSubscriber(
            'Select an object to measure',cps.NONE, doc = '''
            Select the objects to be measured.'''))
        
        if can_delete:
            group.append("remover", cps.RemoveSettingButton('', 'Remove this object', self.object_groups, group))
        self.object_groups.append(group)

    def settings(self):
        '''Return the settings to be saved in the pipeline'''
        result = [self.image_count, self.object_count]
        result += [image_group.image_name for image_group in self.image_groups]
        result += [self.images_or_objects]
        result += [object_group.object_name for object_group in self.object_groups]
        return result

    def prepare_settings(self, setting_values):
        '''Make sure there are the right number of image and object slots for the incoming settings'''
        image_count = int(setting_values[0])
        object_count = int(setting_values[1])
        if image_count < 2:
            raise ValueError("The MeasureCorrelate module must have at least two input images. %d found in pipeline file"%image_count)
        
        del self.image_groups[image_count:]
        while len(self.image_groups) < image_count:
            self.add_image()
        
        del self.object_groups[object_count:]
        while len(self.object_groups) < object_count:
            self.add_object()

    def visible_settings(self):
        result = []
        for image_group in self.image_groups:
            result += image_group.visible_settings()
        result += [self.add_image_button, self.spacer_2, self.images_or_objects]
        if self.wants_objects():
            for object_group in self.object_groups:
                result += object_group.visible_settings()
            result += [self.add_object_button]
        return result

    def get_image_pairs(self):
        '''Yield all permutations of pairs of images to correlate
        
        Yields the pairs of images in a canonical order.
        '''
        for i in range(self.image_count.value-1):
            for j in range(i+1, self.image_count.value):
                yield (self.image_groups[i].image_name.value,
                       self.image_groups[j].image_name.value)

    def wants_images(self):
        '''True if the user wants to measure correlation on whole images'''
        return self.images_or_objects in (M_IMAGES, M_IMAGES_AND_OBJECTS)

    def wants_objects(self):
        '''True if the user wants to measure per-object correlations'''
        return self.images_or_objects in (M_OBJECTS, M_IMAGES_AND_OBJECTS)

    def run(self, workspace):
        '''Calculate measurements on an image set'''
        col_labels = ["First image","Second image","Objects","Measurement","Value"]
        statistics = []
        for first_image_name, second_image_name in self.get_image_pairs():
            if self.wants_images():
                statistics += self.run_image_pair_images(workspace, 
                                                         first_image_name, 
                                                         second_image_name)
            if self.wants_objects():
                for object_name in [group.object_name.value for group in self.object_groups]:
                    statistics += self.run_image_pair_objects(workspace, 
                                                              first_image_name,
                                                              second_image_name, 
                                                              object_name)
        if self.show_window:
            workspace.display_data.statistics = statistics
            workspace.display_data.col_labels = col_labels

    def display(self, workspace, figure):
        statistics = workspace.display_data.statistics
        figure.set_subplots((1, 1))
        figure.subplot_table(0, 0, statistics, workspace.display_data.col_labels)

    def run_image_pair_images(self, workspace, first_image_name, 
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
        mask = (first_mask & second_mask & 
                (~ np.isnan(first_pixel_data)) &
                (~ np.isnan(second_pixel_data)))
        if np.any(mask):
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
        else:
            corr = np.NaN
            slope = np.NaN
        #
        # Add the measurements
        #
        corr_measurement = F_CORRELATION_FORMAT%(first_image_name, 
                                                 second_image_name)
        slope_measurement = F_SLOPE_FORMAT%(first_image_name,
                                            second_image_name)
        workspace.measurements.add_image_measurement(corr_measurement, corr)
        workspace.measurements.add_image_measurement(slope_measurement, slope)
        return [[first_image_name, second_image_name, "-", "Correlation","%.2f"%corr],
                [first_image_name, second_image_name, "-", "Slope","%.2f"%slope]]

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
        labels = objects.segmented
        try:
            first_pixels  = objects.crop_image_similarly(first_image.pixel_data)
            first_mask    = objects.crop_image_similarly(first_image.mask)
        except ValueError:
            first_pixels, m1 = cpo.size_similarly(labels, first_image.pixel_data)
            first_mask, m1 = cpo.size_similarly(labels, first_image.mask)
            first_mask[~m1] = False
        try:
            second_pixels = objects.crop_image_similarly(second_image.pixel_data)
            second_mask   = objects.crop_image_similarly(second_image.mask)
        except ValueError:
            second_pixels, m1 = cpo.size_similarly(labels, second_image.pixel_data)
            second_mask, m1 = cpo.size_similarly(labels, second_image.mask)
            second_mask[~m1] = False
        mask   = ((labels > 0) & first_mask & second_mask)
        first_pixels = first_pixels[mask]
        second_pixels = second_pixels[mask]
        labels = labels[mask]
        if len(labels)==0:
            n_objects = 0
        else:
            n_objects = np.max(labels)
        if n_objects == 0:
            corr = np.zeros((0,))
        else:
            #
            # The correlation is sum((x-mean(x))(y-mean(y)) /
            #                         ((n-1) * std(x) *std(y)))
            #
            lrange = np.arange(n_objects,dtype=np.int32)+1
            area  = fix(scind.sum(np.ones_like(labels), labels, lrange))
            mean1 = fix(scind.mean(first_pixels, labels, lrange))
            mean2 = fix(scind.mean(second_pixels, labels, lrange))
            #
            # Calculate the standard deviation times the population.
            #
            std1 = np.sqrt(fix(scind.sum((first_pixels-mean1[labels-1])**2,
                                         labels, lrange)))
            std2 = np.sqrt(fix(scind.sum((second_pixels-mean2[labels-1])**2,
                                         labels, lrange)))
            x = first_pixels - mean1[labels-1]  # x - mean(x)
            y = second_pixels - mean2[labels-1] # y - mean(y)
            corr = fix(scind.sum(x * y / (std1[labels-1] * std2[labels-1]),
                                 labels, lrange))
            corr[~ np.isfinite(corr)] = 0
        measurement = ("Correlation_Correlation_%s_%s" %
                       (first_image_name, second_image_name))
        workspace.measurements.add_measurement(object_name, measurement, corr)
        if n_objects == 0:
            return [[first_image_name, second_image_name, object_name,
                     "Mean correlation","-"],
                    [first_image_name, second_image_name, object_name,
                     "Median correlation","-"],
                    [first_image_name, second_image_name, object_name,
                     "Min correlation","-"],
                    [first_image_name, second_image_name, object_name,
                     "Max correlation","-"]]
        else:
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
            if self.wants_images():
                columns += [(cpmeas.IMAGE,
                             F_CORRELATION_FORMAT%(first_image, second_image),
                             cpmeas.COLTYPE_FLOAT),
                            (cpmeas.IMAGE,
                             F_SLOPE_FORMAT%(first_image, second_image),
                             cpmeas.COLTYPE_FLOAT)]
            if self.wants_objects():
                for i in range(self.object_count.value):
                    object_name = self.object_groups[i].object_name.value
                    columns += [(object_name,
                                 F_CORRELATION_FORMAT %
                                 (first_image, second_image),
                                 cpmeas.COLTYPE_FLOAT)]
        return columns

    def get_categories(self, pipeline, object_name):
        '''Return the categories supported by this module for the given object
        
        object_name - name of the measured object or cpmeas.IMAGE
        '''
        if ((object_name == cpmeas.IMAGE and self.wants_images()) or
            ((object_name != cpmeas.IMAGE) and self.wants_objects() and
             (object_name in [x.object_name.value for x in self.object_groups]))):
            return ["Correlation"]
        return [] 

    def get_measurements(self, pipeline, object_name, category):
        if self.get_categories(pipeline, object_name) == [category]:
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

    def upgrade_settings(self, setting_values, variable_revision_number, 
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
                object_names = [cps.NONE]
            setting_values = ([str(len(image_names)), str(len(object_names))] +
                              image_names + [m] + object_names)
            from_matlab = False
            variable_revision_number = 1
        if variable_revision_number == 1:
            #
            # Wording of image / object text changed
            #
            image_count, object_count = [int(x) for x in setting_values[:2]]
            image_names = setting_values[2:(image_count+2)]
            m = setting_values[image_count+2]
            object_names = setting_values[(image_count+3):]
            if m == "Images":
                m = M_IMAGES
            elif m == "Objects":
                m = M_OBJECTS
            elif m == "Images and objects":
                m = M_IMAGES_AND_OBJECTS
            setting_values = ([str(image_count), str(object_count)] +
                              image_names + [m] + object_names)
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

