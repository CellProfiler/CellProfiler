'''<b>Measure image granularity</b> outputs a spectra of size measurements 
of the textures in the image.
<hr>

Image granularity is a texture measure that tries a series of structure elements 
of increasing size and outputs a spectrum of measures of how well these structure 
elements fit in the texture of the image. Granularity is measured as described by 
Ilya Ravkin (ref below). The size of the starting structure element as well as the
length of the spectrum is given as input. The module returns one measurement 
for each instant of the granularity spectrum.
<br><br>
Image sampling: If the textures of interest are larger than a few pixels, it is recommended 
to subsample the image with a factor &lt;1 to speed up the processing. Down sampling the 
image will let you detect larger structures with a smaller sized structure element.
<br><br>
More detail: The subsampling fraction is to compensate for the
usually oversampled images; at least they are oversampled in both
Transfluor library image sets. Subsampling by 1/4 reduces computation time by (1/4)^3 because the size
of the image is (1/4)^2 of original and length of granular spectrum can
be 1/4 of original. Moreover, the results were actually a little better
with subsampling, which is probably because with subsampling the
individual granular spectrum components can be used as features, whereas
without subsampling a feature should be a sum of several adjacent
granular spectrum components. The recommendation on the numerical value
can't be given in advance; an analysis like in the above reference is
required before running the whole set.
See <a href="http://www.ravkin.net/presentations/">
http://www.ravkin.net/presentations/Statistical_properties_of_algorithms_for_analysis_of_cell_images.pdf</a> 
slides 27-31, 49-50.
<br><br>
It is also important to remove low frequency image background variations as 
they will affect the final granularity measure. Any method can be used (as 
a pre-processing step to this module). This module simply subtracts a highly 
open image, where the degree of opening is decided by how much the image is 
down sampled before opening.
<br><br>
References:<i>
<br><br>
J.Serra, Image Analysis and Mathematical Morphology, Vol. 1. Academic
Press, London, 1989 Maragos,P. "Pattern spectrum and multiscale shape
representation", IEEE Transactions on Pattern Analysis and Machine
Intelligence, 11, N 7, pp. 701-716, 1989
<br><br>
L.Vincent "Granulometries and Opening Trees", Fundamenta Informaticae,
41, No. 1-2, pp. 57-90, IOS Press, 2000.
<br><br>
L.Vincent "Morphological Area Opening and Closing for Grayscale Images",
Proc. NATO Shape in Picture Workshop, Driebergen, The Netherlands, pp.
197-208, 1992.
<br><br>
I.Ravkin, V.Temov "Bit representation techniques and image processing",
Applied Informatics, v.14, pp. 41-90, Finances and Statistics, Moskow,
1988 (in Russian)</i>
'''

#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Developed by the Broad Institute
#Copyright 2003-2009
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org


__version__="$Revision$"

import numpy as np
import scipy.ndimage as scind
import uuid

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw
import cellprofiler.cpimage as cpi
import cellprofiler.cpmath.cpmorphology as morph

'Granularity category'
C_GRANULARITY = "Granularity"

class MeasureImageGranularity(cpm.CPModule):
    module_name = 'MeasureImageGranularity'
    category = "Measurement"
    variable_revision_number = 1
    def create_settings(self):
        self.image_settings = []
        self.add_image_setting(False)
        self.add_image_button = cps.DoSomething(
            "Add another image:","Add",self.add_image_setting)
        
    def add_image_setting(self, can_delete = True):
        self.image_settings.append(ImageSetting(self.image_settings, 
                                                can_delete))
    def settings(self):
        result = []
        for image_setting in self.image_settings:
            result += image_setting.settings()
        return result
    
    def prepare_settings(self, setting_values):
        '''Adjust self.image_groups to account for the expected # of images'''
        assert len(setting_values) % ImageSetting.setting_count == 0
        group_count = len(setting_values) / ImageSetting.setting_count
        while len(self.image_settings) > group_count:
            del self.image_settings[-1]
        while len(self.image_settings) < group_count:
            self.add_image_setting()
    
    def visible_settings(self):
        result = []
        for image_setting in self.image_settings:
            result += image_setting.visible_settings()
        return result + [self.add_image_button]
    
    def run(self, workspace):
        max_scale = np.max([image_setting.granular_spectrum_length.value
                            for image_setting in self.image_settings])
        statistics = [[ "Image name" ] + 
                      [ "GS%d"%n for n in range(1,max_scale+1)]]
        for image_setting in self.image_settings:
            statistic = self.run_on_image_setting(workspace, image_setting)
            statistic += ["-"] * (max_scale - image_setting.granular_spectrum_length.value)
            statistics.append(statistic)
        if not workspace.frame is None:
            figure = workspace.create_or_find_figure(subplots=(1,1))
            ratio = [1.0 / float(max_scale+1)] * (max_scale+1)
            figure.subplot_table(0,0,statistics, ratio = ratio)
    
    def run_on_image_setting(self, workspace, image_setting):
        assert isinstance(workspace, cpw.Workspace)
        image_set = workspace.image_set
        assert isinstance(image_set, cpi.ImageSet)
        measurements = workspace.measurements
        assert isinstance(image_setting, ImageSetting)
        image = image_set.get_image(image_setting.image_name.value,
                                    must_be_grayscale=True)
        #
        # Downsample the image and mask
        #
        new_shape = np.array(image.pixel_data.shape)
        if image_setting.subsample_size.value < 1:
            new_shape = new_shape * image_setting.subsample_size.value
            i,j = (np.mgrid[0:new_shape[0],0:new_shape[1]].astype(float) /
                   image_setting.subsample_size.value)
            pixels = scind.map_coordinates(image.pixel_data,(i,j),order=1)
            mask = scind.map_coordinates(image.mask.astype(float), (i,j)) > .9
        else:
            pixels = image.pixel_data
            mask = image.mask
        #
        # Remove background pixels using a greyscale tophat filter
        #
        if image_setting.image_sample_size.value < 1:
            back_shape = new_shape * image_setting.image_sample_size.value
            i,j = (np.mgrid[0:back_shape[0],0:back_shape[1]].astype(float) /
                   image_setting.image_sample_size.value)
            back_pixels = scind.map_coordinates(pixels,(i,j), order=1)
            back_mask = scind.map_coordinates(mask.astype(float), (i,j)) > .9
        else:
            back_pixels = pixels
            back_mask = mask
        radius = image_setting.element_size.value
        back_pixels = morph.grey_erosion(back_pixels, radius, back_mask)
        back_pixels = morph.grey_dilation(back_pixels, radius, back_mask)
        if image_setting.image_sample_size.value < 1:
            i,j = np.mgrid[0:new_shape[0],0:new_shape[1]].astype(float)
            #
            # Make sure the mapping only references the index range of
            # back_pixels.
            #
            i *= float(back_shape[0]-1)/float(new_shape[0]-1)
            j *= float(back_shape[1]-1)/float(new_shape[1]-1)
            back_pixels = scind.map_coordinates(back_pixels,(i,j), order=1)
        pixels -= back_pixels
        pixels[pixels < 0] = 0
        #
        # Transcribed from the Matlab module: granspectr function
        #
        # CALCULATES GRANULAR SPECTRUM, ALSO KNOWN AS SIZE DISTRIBUTION,
        # GRANULOMETRY, AND PATTERN SPECTRUM, SEE REF.:
        # J.Serra, Image Analysis and Mathematical Morphology, Vol. 1. Academic Press, London, 1989
        # Maragos,P. "Pattern spectrum and multiscale shape representation", IEEE Transactions on Pattern Analysis and Machine Intelligence, 11, N 7, pp. 701-716, 1989
        # L.Vincent "Granulometries and Opening Trees", Fundamenta Informaticae, 41, No. 1-2, pp. 57-90, IOS Press, 2000.
        # L.Vincent "Morphological Area Opening and Closing for Grayscale Images", Proc. NATO Shape in Picture Workshop, Driebergen, The Netherlands, pp. 197-208, 1992.
        # I.Ravkin, V.Temov "Bit representation techniques and image processing", Applied Informatics, v.14, pp. 41-90, Finances and Statistics, Moskow, 1988 (in Russian)
        # THIS IMPLEMENTATION INSTEAD OF OPENING USES EROSION FOLLOWED BY RECONSTRUCTION
        #
        ng = image_setting.granular_spectrum_length.value
        startmean = np.mean(pixels[mask])
        ero = pixels.copy()
        # Mask the test image so that masked pixels will have no effect
        # during reconstruction
        #
        ero[~mask] = 0
        currentmean = startmean
        startmean = max(startmean, np.finfo(float).eps)
        
        footprint = np.array([[False,True,False],
                              [True ,True,True],
                              [False,True,False]])
        statistics = [ image_setting.image_name.value]
        for i in range(1,ng+1):
            prevmean = currentmean
            ero = morph.grey_erosion(ero, mask = mask, footprint=footprint)
            rec = morph.grey_reconstruction(ero, pixels, footprint)
            currentmean = np.mean(rec[mask])
            gs = (prevmean - currentmean) * 100 / startmean
            statistics += [ "%.2f"%gs]
            feature = image_setting.granularity_feature(i)
            measurements.add_image_measurement(feature, gs)
        return statistics
    
    def get_measurement_columns(self, pipeline):
        result = []
        for image_setting in self.image_settings:
            result += [(cpmeas.IMAGE, 
                        image_setting.granularity_feature(i),
                        cpmeas.COLTYPE_FLOAT)
                        for i in range(1,image_setting.granular_spectrum_length.value+1)]
        return result
    
    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return [ C_GRANULARITY ]
        else:
            return []
    
    def get_measurements(self, pipeline, object_name, category):
        if object_name == cpmeas.IMAGE and category == C_GRANULARITY:
            max_length = np.max([image_setting.granular_spectrum_length.value
                                 for image_setting in self.image_settings])
            return [str(i) for i in range(1,max_length+1)]
        return []
    
    def get_measurement_images(self, pipeline, object_name, category,
                               measurement):
        result = []
        if object_name == cpmeas.IMAGE and category == C_GRANULARITY:
            try:
                length = int(measurement)
                if length <= 0:
                    return []
            except ValueError:
                return []
            for image_setting in self.image_settings:
                if image_setting.granular_spectrum_length.value >= length:
                    result.append(image_setting.image_name.value)
        return result
        
    def upgrade_settings(self,setting_values,variable_revision_number,
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 1:
            # Matlab and pyCP v1 are identical
            from_matlab = False
            variable_revision_number = 1
        return setting_values, variable_revision_number, from_matlab
    
class ImageSetting(object):
    setting_count = 5
    def __init__(self, image_settings, can_delete):
        self.can_delete = can_delete
        self.key = uuid.uuid4()
        self.image_name = cps.ImageNameSubscriber(
            "Select the input image",
            "None")
        self.subsample_size = cps.Float(
            "Subsampling factor for granularity measurements",
            .25, minval = np.finfo(float).eps, maxval = 1,doc='''Subsampling factor for granularity 
            measurements: If the textures of 
            interest are larger than a few pixels, it is recommended to subsample the image with a factor 
            &lt;1 to speed up the processing. Down sampling the image will let you detect larger 
            structures with a smaller sized structure element. A factor &gt;1 will increase the accuracy 
            but also require more processing time. Images are typically of higher resolution than is 
            required for granularity measurements, so the default is 0.25. For low-resolution images, 
            increase the subsampling fraction and for high-resolution images, decrease the subsampling 
            fraction. ''')
        self.image_sample_size = cps.Float(
            "Subsampling factor for background reduction",
            .25, minval = np.finfo(float).eps, maxval = 1,doc='''Reduce background: It is important to 
            remove low frequency image background variations as they will affect the final granularity 
            measure. Here we simply subtract a highly open image. To do it fast we subsample the image 
            first. The subsampling factor for background reduction is usually [0.125 - 0.25].  This is 
            highly empirical, but a small factor should be use if the structures of interest are large. The 
            significance of background removal in the context of granulometry is that image 
            volume at certain granular size is normalized by the total image volume, which depends on 
            how the background was removed.''')
        self.element_size = cps.Integer(
            "Radius of structuring element",
            10, minval = 1,doc='''Radius of the structuring element: 
            This radius should correspond to the radius of the textures of interest <i>after</i> 
            subsampling. I.e., if textures in the original image scale have a radius of 40 
            pixels, and a subsampling factor of 0.25 is used, the structuring element size should be
            10, or slightly smaller, and the range of the spectrum defined below will cover more sizes.''')
        self.granular_spectrum_length = cps.Integer(
            "Length of the granular spectrum",
            16, minval = 1,doc='''Granular spectrum length: Needs a trial run to see which Granular 
            Spectrum Length yields informative measurements. Start by using a wide spectrum, and 
            narrow it down to the informative range to save time.''')
        if can_delete:
            def remove():
                index = [x.key for x in image_settings].index(self.key)
                del image_settings[index]
                
            self.remove_button = cps.DoSomething("Remove above image",
                                                 "Remove", remove)
    
    def settings(self):
        return [self.image_name, self.subsample_size, 
                self.image_sample_size, self.element_size,
                self.granular_spectrum_length]
    
    def visible_settings(self):
        result = [self.image_name, self.subsample_size, 
                  self.image_sample_size, self.element_size,
                  self.granular_spectrum_length]
        if self.can_delete:
            result += [self.remove_button]
        return result
    
    def granularity_feature(self, length):
        return "%s_%d_%s"% (C_GRANULARITY, length, self.image_name.value)
