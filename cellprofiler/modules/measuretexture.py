"""
<b>Measure Texture</b> measures the degree and nature of textures within 
objects (versus smoothness).
<hr>
This module measures the variations in grayscale images.  An object (or 
entire image) without much texture has a smooth appearance; an
object or image with a lot of texture will appear rough and show a wide 
variety of pixel intensities.

<p>This module can also measure textures of objects against grayscale images. 
Any input objects specified will have their texture measured against <i>all</i> input
images specfied, which may lead to image-object texture combinations that are unneccesary. 
If you do not want this behavior, use multiple <b>MeasureTexture</b> modules to 
specify the particular image-object measures that you want.</p>
                        
<h4>Available measurements</h4>
<ul>
<li><i>Haralick Features:</i> Haralick texture features are derived from the 
co-occurrence matrix, which contains information about how image intensities in pixels with a 
certain position in relation to each other occur together. <b>MeasureTexture</b>
can measure textures at different scales; the scale you choose determines
how the co-occurrence matrix is constructed.
For example, if you choose a scale of 2, each pixel in the image (excluding
some border pixels) will be compared against the one that is two pixels to
the right. <b>MeasureTexture</b> quantizes the image into eight intensity
levels. There are then 8x8 possible ways to categorize a pixel with its
scale-neighbor. <b>MeasureTexture</b> forms the 8x8 co-occurrence matrix
by counting how many pixels and neighbors have each of the 8x8 intensity
combinations. 
<p>Thirteen measurements are then calculated for the image by performing
mathematical operations on the co-occurrence matrix (the formulas can be found 
<a href="http://murphylab.web.cmu.edu/publications/boland/boland_node26.html">here</a>):
<ul>
<li><i>AngularSecondMoment</i></li>
<li><i>Contrast</i></li>
<li><i>Correlation</i></li>
<li><i>Variance</i></li>
<li><i>InverseDifferenceMoment</i></li>
<li><i>SumAverage</i></li>
<li><i>SumVariance</i></li>
<li><i>SumEntropy</i></li>
<li><i>Entropy</i></li>
<li><i>DifferenceVariance</i></li>
<li><i>DifferenceEntropy</i></li>
<li><i>InfoMeas1</i></li>
<li><i>InfoMeas2</i></li>
</ul>
Each measurement is suffixed with the direction of the offset used between
pixels in the co-occurrence matrix:
<ul>
<li><i>0:</i> Horizontal</li>
<li><i>90:</i> Vertical</li>
<li><i>45:</i> Diagonal</li>
<li><i>135:</i> Anti-diagonal</li>
</ul>
</p>
</li>
<li>
<i>Gabor "wavelet" features:</i> These features are similar to wavelet features, 
and they are obtained by applying so-called Gabor filters to the image. The Gabor 
filters measure the frequency content in different orientations. They are very 
similar to wavelets, and in the current context they work exactly as wavelets, but
they are not wavelets by a strict mathematical definition. The Gabor
features detect correlated bands of intensities, for instance, images of
Venetian blinds would have high scores in the horizontal orientation.</li>
</ul>

<h4>Technical notes</h4> 

To calculate the Haralick features, <b>MeasureTexture</b> normalizes the 
co-occurence matrix at the per-object level by basing the intensity levels of the 
matrix on the maximum and minimum intensity observed within each object. This
is beneficial for images in which the maximum intensities of the objects vary
substantially because each object will have the full complement of levels.

<p><b>MeasureTexture</b> performs a vectorized calculation of the Gabor filter, 
properly scaled to the size of the object being measured and covering all 
pixels in the object. The Gabor filter can be calculated at a user-selected 
number of angles by using the following algorithm to compute a score
at each scale using the Gabor filter:
<ul>
<li>Divide the half-circle from 0 to 180&deg; by the number of desired
angles. For instance, if the user chooses two angles, <b>MeasureTexture</b>
uses 0 and 90 &deg; (horizontal and vertical) for the filter
orientations. This is the &theta; value from the reference paper.</li>
<li>For each angle, compute the Gabor filter for each object in the image
at two phases separated by 90&deg; in order to account for texture
features whose peaks fall on even or odd quarter-wavelengths.</li>
<li>Multiply the image times each Gabor filter and sum over the pixels
in each object.</li>
<li>Take the square root of the sum of the squares of the two filter scores.
This results in one score per &theta;.</li>
<li>Save the maximum score over all &theta; as the score at the desired scale.</li>
</ul>
</p>

<h4>References</h4>
<ul>
<li>Haralick RM, Shanmugam K, Dinstein I. (1973), "Textural Features for Image
Classification" <i>IEEE Transaction on Systems Man, Cybernetics</i>,
SMC-3(6):610-621. 
<a href="http://dx.doi.org/10.1109/TSMC.1973.4309314">(link)</a></li>
<li>Gabor D. (1946). "Theory of communication" 
<i>Journal of the Institute of Electrical Engineers</i> 93:429-441.
<a href="http://dx.doi.org/10.1049/ji-3-2.1946.0074">(link)</a></li>
</ul>
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
import scipy.ndimage as scind

import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
from cellprofiler.settings import YES, NO
import cellprofiler.measurements as cpmeas
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.cpmath.haralick import Haralick, normalized_per_object
from cellprofiler.cpmath.filter import gabor, stretch

"""The category of the per-object measurements made by this module"""
TEXTURE = 'Texture'

"""The "name" slot in the object group dictionary entry"""
OG_NAME = 'name'
"""The "remove"slot in the object group dictionary entry"""
OG_REMOVE = 'remove'

F_HARALICK = """AngularSecondMoment Contrast Correlation Variance 
InverseDifferenceMoment SumAverage SumVariance SumEntropy Entropy
DifferenceVariance DifferenceEntropy InfoMeas1 InfoMeas2""".split()

F_GABOR = "Gabor"

H_HORIZONTAL = "Horizontal"
A_HORIZONTAL = "0"
H_VERTICAL = "Vertical"
A_VERTICAL = "90"
H_DIAGONAL = "Diagonal"
A_DIAGONAL = "45"
H_ANTIDIAGONAL = "Anti-diagonal"
A_ANTIDIAGONAL = "135"
H_ALL = [H_HORIZONTAL, H_VERTICAL, H_DIAGONAL, H_ANTIDIAGONAL]

H_TO_A = { H_HORIZONTAL: A_HORIZONTAL,
           H_VERTICAL: A_VERTICAL,
           H_DIAGONAL: A_DIAGONAL,
           H_ANTIDIAGONAL: A_ANTIDIAGONAL }
class MeasureTexture(cpm.CPModule):

    module_name = "MeasureTexture"
    variable_revision_number = 3
    category = 'Measurement'

    def create_settings(self):
        """Create the settings for the module at startup.
        
        The module allows for an unlimited number of measured objects, each
        of which has an entry in self.object_groups.
        """ 
        self.image_groups = []
        self.object_groups = []
        self.scale_groups = []
        self.image_count = cps.HiddenCount(self.image_groups)
        self.object_count = cps.HiddenCount(self.object_groups)
        self.scale_count = cps.HiddenCount(self.scale_groups)
        self.add_image_cb(can_remove = False)
        self.add_images = cps.DoSomething("", "Add another image",
                                          self.add_image_cb)
        self.image_divider = cps.Divider()
        self.add_object_cb(can_remove = True)
        self.add_objects = cps.DoSomething("", "Add another object",
                                           self.add_object_cb)
        self.object_divider = cps.Divider()
        self.add_scale_cb(can_remove = False)
        self.add_scales = cps.DoSomething("", "Add another scale",
                                          self.add_scale_cb)
        self.scale_divider = cps.Divider()
        
        self.wants_gabor = cps.Binary(
            "Measure Gabor features?", True, doc =
            """The Gabor features measure striped texture in an object, and can 
            take a substantial time to calculate. 
            <p>Select <i>%(YES)s</i> to measure the Gabor features. Select 
            <i>%(NO)s</i> to skip the Gabor feature calculation if it is not 
            informative for your images.</p>"""%globals())
        
        self.gabor_angles = cps.Integer("Number of angles to compute for Gabor",4,2, doc="""
            <i>(Used only if Gabor features are measured)</i><br>
            Enter the number of angles to use for each Gabor texture measurement.
            The default value is 4 which detects bands in the horizontal, vertical and diagonal
            orientations.""")

    def settings(self):
        """The settings as they appear in the save file."""
        result = [self.image_count, self.object_count, self.scale_count]
        for groups, elements in [(self.image_groups, ['image_name']),
                                (self.object_groups, ['object_name']),
                                (self.scale_groups, ['scale', 'angles'])]:
            for group in groups:
                for element in elements:
                    result += [getattr(group, element)]
        result += [self.wants_gabor, self.gabor_angles]
        return result

    def prepare_settings(self,setting_values):
        """Adjust the number of object groups based on the number of
        setting_values"""
        for count, sequence, fn in\
            ((int(setting_values[0]), self.image_groups, self.add_image_cb),
             (int(setting_values[1]), self.object_groups, self.add_object_cb),
             (int(setting_values[2]), self.scale_groups, self.add_scale_cb)):
            del sequence[count:]
            while len(sequence) < count:
                fn()
        
    def visible_settings(self):
        """The settings as they appear in the module viewer"""
        result = []
        for groups, add_button, div in [(self.image_groups, self.add_images, self.image_divider),
                                        (self.object_groups, self.add_objects, self.object_divider),
                                        (self.scale_groups, self.add_scales, self.scale_divider)]:
            for group in groups:
                result += group.visible_settings()
            result += [add_button, div]
        result += [self.wants_gabor]
        if self.wants_gabor:
            result += [self.gabor_angles]
        return result

    def add_image_cb(self, can_remove = True):
        '''Add an image to the image_groups collection
        
        can_delete - set this to False to keep from showing the "remove"
                     button for images that must be present.
        '''
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))
        group.append('image_name', 
                     cps.ImageNameSubscriber(
                         "Select an image to measure",cps.NONE, doc="""
                         Select the grayscale images whose texture you want to measure."""))
        
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove this image", self.image_groups, group))
        self.image_groups.append(group)

    def add_object_cb(self, can_remove = True):
        '''Add an object to the object_groups collection
        
        can_delete - set this to False to keep from showing the "remove"
                     button for objects that must be present.
        '''
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))
        group.append('object_name', 
                     cps.ObjectNameSubscriber("Select objects to measure",cps.NONE, doc="""
                        Select the objects whose texture you want to measure. 
                        If you only want to measure the texture 
                        for the image overall, you can remove all objects using the "Remove this object" button. 
                        <p>Objects specified here will have their
                        texture measured against <i>all</i> images specfied above, which
                        may lead to image-object combinations that are unneccesary. If you
                        do not want this behavior, use multiple <b>MeasureTexture</b>
                        modules to specify the particular image-object measures that you want.</p>"""))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove this object", self.object_groups, group))
        self.object_groups.append(group)

    def add_scale_cb(self, can_remove = True):
        '''Add a scale to the scale_groups collection
        
        can_delete - set this to False to keep from showing the "remove"
                     button for scales that must be present.
        '''
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))
        group.append('scale', 
                     cps.Integer("Texture scale to measure",
                                 len(self.scale_groups)+3,
                                 doc="""You can specify the scale of texture to be measured, in pixel units; 
                                 the texture scale is the distance between correlated intensities in the image. A 
                                 higher number for the scale of texture measures larger patterns of 
                                 texture whereas smaller numbers measure more localized patterns of 
                                 texture. It is best to measure texture on a scale smaller than your 
                                 objects' sizes, so be sure that the value entered for scale of texture is 
                                 smaller than most of your objects. For very small objects (smaller than 
                                 the scale of texture you are measuring), the texture cannot be measured 
                                 and will result in a undefined value in the output file."""))
        group.append('angles', cps.MultiChoice(
            "Angles to measure", H_ALL, H_ALL,
        doc = """The Haralick texture measurements are based on the correlation
        between pixels offset by the scale in one of four directions:
        <p><ul>
        <li><i>%(H_HORIZONTAL)s</i> - the correlated pixel is "scale" pixels
        to the right of the pixel of interest.</li>
        <li><i>%(H_VERTICAL)s</i> - the correlated pixel is "scale" pixels
        below the pixel of interest.</li>
        <li><i>%(H_DIAGONAL)s</i> - the correlated pixel is "scale" pixels
        to the right and "scale" pixels below the pixel of interest.</li>
        <li><i>%(H_ANTIDIAGONAL)s</i> - the correlated pixel is "scale"
        pixels to the left and "scale" pixels below the pixel of interest.</li>
        </ul><p>
        Choose one or more directions to measure.""" % globals()))
                                
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove this scale", self.scale_groups, group))
        self.scale_groups.append(group)

        
    def validate_module(self, pipeline):
        """Make sure chosen objects, images and scales are selected only once"""
        images = set()
        for group in self.image_groups:
            if group.image_name.value in images:
                raise cps.ValidationError(
                    "%s has already been selected" %group.image_name.value,
                    group.image_name)
            images.add(group.image_name.value)
            
        objects = set()
        for group in self.object_groups:
            if group.object_name.value in objects:
                raise cps.ValidationError(
                    "%s has already been selected" %group.object_name.value,
                    group.object_name)
            objects.add(group.object_name.value)
            
        scales = set()
        for group in self.scale_groups:
            if group.scale.value in scales:
                raise cps.ValidationError(
                    "%s has already been selected" %group.scale.value,
                    group.scale)
            scales.add(group.scale.value)
            
    def get_categories(self,pipeline, object_name):
        """Get the measurement categories supplied for the given object name.
        
        pipeline - pipeline being run
        object_name - name of labels in question (or 'Images')
        returns a list of category names
        """
        if any([object_name == og.object_name for og in self.object_groups]):
            return [TEXTURE]
        elif object_name == cpmeas.IMAGE:
            return [TEXTURE]
        else:
            return []

    def get_features(self):
        '''Return the feature names for this pipeline's configuration'''
        return F_HARALICK+([F_GABOR] if self.wants_gabor else [])
    
    def get_measurements(self, pipeline, object_name, category):
        '''Get the measurements made on the given object in the given category
        
        pipeline - pipeline being run
        object_name - name of objects being measured
        category - measurement category
        '''
        if category in self.get_categories(pipeline, object_name):
            return self.get_features()
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        '''Get the list of images measured
        
        pipeline - pipeline being run
        object_name - name of objects being measured
        category - measurement category
        measurement - measurement made on images
        '''
        measurements = self.get_measurements(pipeline, object_name, category)
        if measurement in measurements:
            return [x.image_name.value for x in self.image_groups]
        return []

    def get_measurement_scales(self, pipeline, object_name, category, 
                               measurement, image_name):
        '''Get the list of scales at which the measurement was taken

        pipeline - pipeline being run
        object_name - name of objects being measured
        category - measurement category
        measurement - name of measurement made
        image_name - name of image that was measured
        '''
        if len(self.get_measurement_images(pipeline, object_name, category,
                                           measurement)) > 0:
            if measurement == F_GABOR:
                return [x.scale.value for x in self.scale_groups]
            
            return sum([["%d_%s" % (x.scale.value, H_TO_A[h])
                         for h in x.angles.get_selections()]
                        for x in self.scale_groups], [])
        return []
    
    def get_measurement_columns(self, pipeline):
        '''Get column names output for each measurement.'''
        cols = []
        for feature in self.get_features():
            for im in self.image_groups:
                for sg in self.scale_groups:
                    if feature == F_GABOR:
                        cols += [(cpmeas.IMAGE,
                                  '%s_%s_%s_%d' % (TEXTURE, feature, 
                                                   im.image_name.value, 
                                                   sg.scale.value),
                                  cpmeas.COLTYPE_FLOAT)]
                    else:
                        for angle in sg.angles.get_selections():
                            cols += [(cpmeas.IMAGE,
                                      '%s_%s_%s_%d_%s' % (
                                          TEXTURE, feature, im.image_name.value, 
                                          sg.scale.value, H_TO_A[angle]),
                                      cpmeas.COLTYPE_FLOAT)]
                            
        for ob in self.object_groups:
            for feature in self.get_features():
                for im in self.image_groups:
                    for sg in self.scale_groups:
                        if feature == F_GABOR:
                            cols += [(ob.object_name.value,
                                      "%s_%s_%s_%d" % (
                                          TEXTURE, feature, im.image_name.value, 
                                          sg.scale.value),
                                      cpmeas.COLTYPE_FLOAT)]
                        else:
                            for angle in sg.angles.get_selections():
                                cols += [(ob.object_name.value,
                                          "%s_%s_%s_%d_%s" % (
                                              TEXTURE, feature, 
                                              im.image_name.value, 
                                              sg.scale.value, H_TO_A[angle]),
                                          cpmeas.COLTYPE_FLOAT)]
                                
        return cols

    def run(self, workspace):
        """Run, computing the area measurements for the objects"""
        
        workspace.display_data.col_labels = [
            "Image", "Object", "Measurement", "Scale", "Value"]
        statistics = []
        for image_group in self.image_groups:
            image_name = image_group.image_name.value
            for scale_group in self.scale_groups:
                scale = scale_group.scale.value
                if self.wants_gabor:
                    statistics += self.run_image_gabor(image_name, scale, workspace)
                for angle in scale_group.angles.get_selections():
                    statistics += self.run_image(image_name, scale, angle, 
                                                 workspace)
                for object_group in self.object_groups:
                    object_name = object_group.object_name.value
                    for angle in scale_group.angles.get_selections():
                        statistics += self.run_one(
                            image_name, object_name, scale, angle, workspace)
                    if self.wants_gabor:
                        statistics += self.run_one_gabor(image_name, 
                                                         object_name, 
                                                         scale,
                                                         workspace)
        if self.show_window:
            workspace.display_data.statistics = statistics
    
    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        figure.subplot_table(0, 0, 
                             workspace.display_data.statistics,
                             col_labels = workspace.display_data.col_labels)
    
    def run_one(self, image_name, object_name, scale, angle, workspace):
        """Run, computing the area measurements for a single map of objects"""
        statistics = []
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale=True)
        objects = workspace.get_objects(object_name)
        pixel_data = image.pixel_data
        if image.has_mask:
            mask = image.mask
        else:
            mask = None
        labels = objects.segmented
        try:
            pixel_data = objects.crop_image_similarly(pixel_data)
        except ValueError:
            #
            # Recover by cropping the image to the labels
            #
            pixel_data, m1 = cpo.size_similarly(labels, pixel_data)
            if np.any(~m1):
                if mask is None:
                    mask = m1
                else:
                    mask, m2 = cpo.size_similarly(labels, mask)
                    mask[~m2] = False
            
        if np.all(labels == 0):
            for name in F_HARALICK:
                statistics += self.record_measurement(
                    workspace, image_name, object_name, 
                    str(scale) + "_" + H_TO_A[angle], name, np.zeros((0,)))
        else:
            scale_i, scale_j = self.get_angle_ij(angle, scale)
                
            for name, value in zip(F_HARALICK, Haralick(pixel_data,
                                                        labels,
                                                        scale_i,
                                                        scale_j,
                                                        mask=mask).all()):
                statistics += self.record_measurement(
                    workspace, image_name, object_name, 
                    str(scale) + "_" + H_TO_A[angle], name, value)
        return statistics

    def get_angle_ij(self, angle, scale):
        if angle == H_VERTICAL:
            return scale, 0
        elif angle == H_HORIZONTAL:
            return 0, scale
        elif angle == H_DIAGONAL:
            return scale, scale
        elif angle == H_ANTIDIAGONAL:
            return scale, -scale
        
    def run_image(self, image_name, scale, angle, workspace):
        '''Run measurements on image'''
        statistics = []
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale=True)
        pixel_data = image.pixel_data
        image_labels = np.ones(pixel_data.shape, int)
        if image.has_mask:
            image_labels[~ image.mask] = 0
        scale_i, scale_j = self.get_angle_ij(angle, scale)
        for name, value in zip(F_HARALICK, Haralick(pixel_data,
                                                    image_labels,
                                                    scale_i,
                                                    scale_j).all()):
            statistics += self.record_image_measurement(
                workspace, image_name, str(scale) + "_" + H_TO_A[angle],
                name, value)
        return statistics
        
    def run_one_gabor(self, image_name, object_name, scale, workspace):
        objects = workspace.get_objects(object_name)
        labels = objects.segmented
        object_count = np.max(labels)
        if object_count > 0:
            image = workspace.image_set.get_image(image_name,
                                                  must_be_grayscale=True)
            pixel_data = image.pixel_data
            labels = objects.segmented
            if image.has_mask:
                mask = image.mask
            else:
                mask = None
            try:
                pixel_data = objects.crop_image_similarly(pixel_data)
                if mask is not None:
                    mask = objects.crop_image_similarly(mask)
                    labels[~mask] = 0
            except ValueError:
                pixel_data, m1 = cpo.size_similarly(labels, pixel_data)
                labels[~m1] = 0
                if mask is not None:
                    mask, m2 = cpo.size_similarly(labels, mask)
                    labels[~m2] = 0
                    labels[~mask] = 0
            pixel_data = normalized_per_object(pixel_data, labels)
            best_score = np.zeros((object_count,))
            for angle in range(self.gabor_angles.value):
                theta = np.pi * angle / self.gabor_angles.value
                g = gabor(pixel_data, labels, scale, theta)
                score_r = fix(scind.sum(g.real, labels,
                                         np.arange(object_count, dtype=np.int32)+ 1))
                score_i = fix(scind.sum(g.imag, labels,
                                         np.arange(object_count, dtype=np.int32)+ 1))
                score = np.sqrt(score_r**2+score_i**2)
                best_score = np.maximum(best_score, score)
        else:
            best_score = np.zeros((0,))
        statistics = self.record_measurement(workspace, 
                                             image_name, 
                                             object_name, 
                                             scale,
                                             F_GABOR, 
                                             best_score)
        return statistics
            
    def run_image_gabor(self, image_name, scale, workspace):
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale=True)
        pixel_data = image.pixel_data
        labels = np.ones(pixel_data.shape, int)
        if image.has_mask:
            labels[~image.mask] = 0
        pixel_data = stretch(pixel_data, labels > 0)
        best_score = 0
        for angle in range(self.gabor_angles.value):
            theta = np.pi * angle / self.gabor_angles.value
            g = gabor(pixel_data, labels, scale, theta)
            score_r = np.sum(g.real)
            score_i = np.sum(g.imag)
            score = np.sqrt(score_r**2+score_i**2)
            best_score = max(best_score, score)
        statistics = self.record_image_measurement(workspace, 
                                                   image_name, 
                                                   scale,
                                                   F_GABOR, 
                                                   best_score)
        return statistics

    def record_measurement(self, workspace,  
                           image_name, object_name, scale,
                           feature_name, result):
        """Record the result of a measurement in the workspace's
        measurements"""
        data = fix(result)
        data[~np.isfinite(data)] = 0
        workspace.add_measurement(
            object_name, 
            "%s_%s_%s_%s" % (TEXTURE, feature_name,image_name, str(scale)), 
            data)
        statistics = [[image_name, object_name, 
                       "%s %s"%(aggregate_name, feature_name), scale, 
                       "%.2f"%fn(data) if len(data) else "-"]
                       for aggregate_name, fn in (("min",np.min),
                                                  ("max",np.max),
                                                  ("mean",np.mean),
                                                  ("median",np.median),
                                                  ("std dev",np.std))]
        return statistics

    def record_image_measurement(self, workspace,  
                                 image_name, scale,
                                 feature_name, result):
        """Record the result of a measurement in the workspace's
        measurements"""
        if not np.isfinite(result):
            result = 0
        workspace.measurements.add_image_measurement("%s_%s_%s_%s"%
                                                     (TEXTURE, feature_name,
                                                      image_name, str(scale)), 
                                                     result)
        statistics = [[image_name, "-", 
                       feature_name, scale, 
                       "%.2f"%(result)]]
        return statistics
    
    def upgrade_settings(self,setting_values,variable_revision_number,
                         module_name,from_matlab):
        """Adjust the setting_values for older save file versions
        
        setting_values - a list of strings representing the settings for
                         this module.
        variable_revision_number - the variable revision number of the module
                                   that saved the settings
        module_name - the name of the module that saved the settings
        from_matlab - true if it was a Matlab module that saved the settings
        
        returns the modified settings, revision number and "from_matlab" flag
        """
        if from_matlab and variable_revision_number == 2:
            #
            # The first 3 settings are:
            # image count (1 for legacy)
            # object count (calculated)
            # scale_count (calculated)
            #
            object_names = [name for name in setting_values[1:7]
                            if name.upper() != cps.DO_NOT_USE.upper()] 
            scales = setting_values[7].split(',')
            setting_values = ([ "1", str(len(object_names)), str(len(scales)),
                               setting_values[0]] + object_names + scales +
                              ["4"])
            variable_revision_number = 1
            from_matlab = False
        if not from_matlab and variable_revision_number == 1:
            #
            # Added "wants_gabor"
            #
            setting_values = setting_values[:-1] + [cps.YES] + setting_values[-1:]
            variable_revision_number = 2
        if not from_matlab and variable_revision_number == 2:
            #
            # Added angles
            #
            image_count = int(setting_values[0])
            object_count = int(setting_values[1])
            scale_count = int(setting_values[2])
            scale_offset = 3 + image_count + object_count
            new_setting_values = setting_values[:scale_offset]
            for scale in setting_values[scale_offset:(scale_offset+scale_count)]:
                new_setting_values += [scale, H_HORIZONTAL] 
            new_setting_values += setting_values[(scale_offset+scale_count):]
            setting_values = new_setting_values
            variable_revision_number = 3
                
        return setting_values, variable_revision_number, from_matlab
