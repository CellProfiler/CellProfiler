'''<b>Measure Granularity</b> outputs spectra of size measurements
of the textures in the image.
<hr>
Image granularity is a texture measurement that tries a series of structure elements
of increasing size and outputs a spectrum of measures of how well these structure
elements fit in the texture of the image. Granularity is measured as described by
Ilya Ravkin (references below). The size of the starting structure element as well
as the range of the spectrum is given as input.

<h4>Available measurements</h4>
<ul>
<li><i>Granularity:</i> The module returns one measurement for each instance of the granularity spectrum.</li>
</ul>

<h4>References</h4>
<ul>
<li>Serra J. (1989) <i>Image Analysis and Mathematical Morphology</i>, Vol. 1. Academic
Press, London </li>
<li>Maragos P. "Pattern spectrum and multiscale shape
representation", <i>IEEE Transactions on Pattern Analysis and Machine
Intelligence</i>, 11, N 7, pp. 701-716, 1989</li>
<li>Vincent L. (2000) "Granulometries and Opening Trees", <i>Fundamenta Informaticae</i>,
41, No. 1-2, pp. 57-90, IOS Press, 2000.</li>
<li>Vincent L. (1992) "Morphological Area Opening and Closing for Grayscale Images",
<i>Proc. NATO Shape in Picture Workshop</i>, Driebergen, The Netherlands, pp.
197-208.</li>
<li>Ravkin I, Temov V. (1988) "Bit representation techniques and image processing",
<i>Applied Informatics</i>, v.14, pp. 41-90, Finances and Statistics, Moskow,
(in Russian)</li>
</ul>
'''

import centrosome.cpmorphology as morph
import numpy as np
import scipy.ndimage as scind
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.setting as cps
import cellprofiler.workspace as cpw
import skimage.morphology

'Granularity category'
C_GRANULARITY = "Granularity_%s_%s"

IMAGE_SETTING_COUNT_V2 = 5
IMAGE_SETTING_COUNT_V3 = 6
IMAGE_SETTING_COUNT = IMAGE_SETTING_COUNT_V3

OBJECTS_SETTING_COUNT_V3 = 1
OBJECTS_SETTING_COUNT = OBJECTS_SETTING_COUNT_V3


class MeasureGranularity(cpm.Module):
    module_name = 'MeasureGranularity'
    category = "Measurement"
    variable_revision_number = 3

    def create_settings(self):
        self.divider_top = cps.Divider(line=False)
        self.images = []
        self.image_count = cps.HiddenCount(self.images, "Image count")
        self.add_image(can_remove=False)
        self.add_button = cps.DoSomething("", "Add another image", self.add_image)
        self.divider_bottom = cps.Divider(line=False)

    def add_image(self, can_remove=True):
        group = GranularitySettingsGroup()
        group.can_remove = can_remove
        if can_remove:
            group.append("divider", cps.Divider(line=True))

        group.append("image_name", cps.ImageNameSubscriber(
                "Select an image to measure", cps.NONE, doc="""
            Select the grayscale images whose granularity you want to measure."""))

        group.append("subsample_size", cps.Float(
                "Subsampling factor for granularity measurements",
                0.25, minval=np.finfo(float).eps, maxval=1, doc='''
            If the textures of
            interest are larger than a few pixels, we recommend you subsample the image with a factor
            &lt;1 to speed up the processing. Down sampling the image will let you detect larger
            structures with a smaller sized structure element. A factor &gt;1 might increase the accuracy
            but also require more processing time. Images are typically of higher resolution than is
            required for granularity measurements, so the default value is 0.25. For low-resolution images,
            increase the subsampling fraction; for high-resolution images, decrease the subsampling
            fraction. Subsampling by 1/4 reduces computation time by (1/4)<sup>3</sup> because the size
            of the image is (1/4)<sup>2</sup> of original and the range of granular spectrum can
            be 1/4 of original. Moreover, the results are sometimes actually a little better
            with subsampling, which is probably because with subsampling the
            individual granular spectrum components can be used as features, whereas
            without subsampling a feature should be a sum of several adjacent
            granular spectrum components. The recommendation on the numerical value
            cannot be determined in advance; an analysis as in this reference may be
            required before running the whole set.
            See this <a href="http://www.ravkin.net/presentations/Statistical%20properties%20of%20algorithms%20for%20analysis%20of%20cell%20images.pdf">
            pdf</a>, slides 27-31, 49-50.'''))

        group.append("image_sample_size", cps.Float(
                "Subsampling factor for background reduction",
                .25, minval=np.finfo(float).eps, maxval=1, doc='''
            It is important to
            remove low frequency image background variations as they will affect the final granularity
            measurement. Any method can be used as a pre-processing step prior to this module;
            we have chosen to simply subtract a highly open image. To do it quickly, we subsample the image
            first. The subsampling factor for background reduction is usually [0.125 &ndash; 0.25].  This is
            highly empirical, but a small factor should be used if the structures of interest are large. The
            significance of background removal in the context of granulometry is that image
            volume at certain granular size is normalized by total image volume, which depends on
            how the background was removed.'''))

        group.append("element_size", cps.Integer(
                "Radius of structuring element",
                10, minval=1, doc='''
            This radius should correspond to the radius of the textures of interest <i>after</i>
            subsampling; i.e., if textures in the original image scale have a radius of 40
            pixels, and a subsampling factor of 0.25 is used, the structuring element size should be
            10 or slightly smaller, and the range of the spectrum defined below will cover more sizes.'''))

        group.append("granular_spectrum_length", cps.Integer(
                "Range of the granular spectrum",
                16, minval=1, doc='''
            You may need a trial run to see which granular
            spectrum range yields informative measurements. Start by using a wide spectrum and
            narrow it down to the informative range to save time.'''))

        group.append("add_objects_button", cps.DoSomething(
                "", "Add another object", group.add_objects, doc="""
            Press this button to add granularity measurements for
            objects, such as those identified by a prior
            <b>IdentifyPrimaryObjects</b> module. <b>MeasureGranularity</b>
            will measure the image's granularity within each object at the
            requested scales."""))

        group.objects = []

        group.object_count = cps.HiddenCount(group.objects, "Object count")

        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove this image", self.images, group))
        self.images.append(group)
        return group

    def validate_module(self, pipeline):
        '''Make sure settings are compatible. In particular, we make sure that no measurements are duplicated'''
        measurements, sources = self.get_measurement_columns(pipeline, return_sources=True)
        d = {}
        for m, s in zip(measurements, sources):
            if m in d:
                raise cps.ValidationError("Measurement %s made twice." % (m[1]), s[0])
            d[m] = True

    def settings(self):
        result = [self.image_count]
        for image in self.images:
            result += [
                image.object_count, image.image_name, image.subsample_size,
                image.image_sample_size, image.element_size,
                image.granular_spectrum_length]
            result += [ob.objects_name for ob in image.objects]
        return result

    def prepare_settings(self, setting_values):
        '''Adjust self.images to account for the expected # of images'''
        image_count = int(setting_values[0])
        idx = 1
        del self.images[:]
        while len(self.images) < image_count:
            image = self.add_image(len(self.images) > 0)
            object_count = int(setting_values[idx])
            idx += IMAGE_SETTING_COUNT
            for i in range(object_count):
                image.add_objects()
                idx += OBJECTS_SETTING_COUNT

    def visible_settings(self):
        result = []
        for index, image in enumerate(self.images):
            result += image.visible_settings()
        result += [self.add_button]
        return result

    def run(self, workspace):
        max_scale = np.max([image.granular_spectrum_length.value
                            for image in self.images])
        col_labels = (["Image name"] +
                      ["GS%d" % n for n in range(1, max_scale + 1)])
        statistics = []
        for image in self.images:
            statistic = self.run_on_image_setting(workspace, image)
            statistic += ["-"] * (max_scale - image.granular_spectrum_length.value)
            statistics.append(statistic)
        if self.show_window:
            workspace.display_data.statistics = statistics
            workspace.display_data.col_labels = col_labels

    def display(self, workspace, figure):
        statistics = workspace.display_data.statistics
        col_labels = workspace.display_data.col_labels
        figure.set_subplots((1, 1))
        figure.subplot_table(0, 0, statistics, col_labels=col_labels)

    def run_on_image_setting(self, workspace, image):
        assert isinstance(workspace, cpw.Workspace)
        image_set = workspace.image_set
        measurements = workspace.measurements
        im = image_set.get_image(image.image_name.value,
                                 must_be_grayscale=True)
        #
        # Downsample the image and mask
        #
        new_shape = np.array(im.pixel_data.shape)
        if image.subsample_size.value < 1:
            new_shape = new_shape * image.subsample_size.value
            if im.dimensions is 2:
                i, j = (np.mgrid[0:new_shape[0], 0:new_shape[1]].astype(float) / image.subsample_size.value)
                pixels = scind.map_coordinates(im.pixel_data, (i, j), order=1)
                mask = scind.map_coordinates(im.mask.astype(float), (i, j)) > .9
            else:
                k, i, j = (np.mgrid[0:new_shape[0], 0:new_shape[1], 0:new_shape[2]].astype(float) / image.subsample_size.value)
                pixels = scind.map_coordinates(im.pixel_data, (k, i, j), order=1)
                mask = scind.map_coordinates(im.mask.astype(float), (k, i, j)) > .9
        else:
            pixels = im.pixel_data
            mask = im.mask
        #
        # Remove background pixels using a greyscale tophat filter
        #
        if image.image_sample_size.value < 1:
            back_shape = new_shape * image.image_sample_size.value
            if im.dimensions is 2:
                i, j = (np.mgrid[0:back_shape[0], 0:back_shape[1]].astype(float) / image.image_sample_size.value)
                back_pixels = scind.map_coordinates(pixels, (i, j), order=1)
                back_mask = scind.map_coordinates(mask.astype(float), (i, j)) > .9
            else:
                k, i, j = (np.mgrid[0:new_shape[0], 0:new_shape[1], 0:new_shape[2]].astype(float) / image.subsample_size.value)
                back_pixels = scind.map_coordinates(pixels, (k, i, j), order=1)
                back_mask = scind.map_coordinates(mask.astype(float), (k, i, j)) > .9
        else:
            back_pixels = pixels
            back_mask = mask
            back_shape = new_shape
        radius = image.element_size.value
        if im.dimensions is 2:
            selem = skimage.morphology.disk(radius, dtype=bool)
        else:
            selem = skimage.morphology.ball(radius, dtype=bool)
        back_pixels_mask = np.zeros_like(back_pixels)
        back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
        back_pixels = skimage.morphology.erosion(back_pixels_mask, selem=selem)
        back_pixels_mask = np.zeros_like(back_pixels)
        back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
        back_pixels = skimage.morphology.dilation(back_pixels_mask, selem=selem)
        if image.image_sample_size.value < 1:
            if im.dimensions is 2:
                i, j = np.mgrid[0:new_shape[0], 0:new_shape[1]].astype(float)
                #
                # Make sure the mapping only references the index range of
                # back_pixels.
                #
                i *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
                j *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
                back_pixels = scind.map_coordinates(back_pixels, (i, j), order=1)
            else:
                k, i, j = np.mgrid[0:new_shape[0], 0:new_shape[1], 0:new_shape[2]].astype(float)
                k *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
                i *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
                j *= float(back_shape[2] - 1) / float(new_shape[2] - 1)
                back_pixels = scind.map_coordinates(back_pixels, (k, i, j), order=1)
        pixels -= back_pixels
        pixels[pixels < 0] = 0

        #
        # For each object, build a little record
        #
        class ObjectRecord(object):
            def __init__(self, name):
                self.name = name
                self.labels = workspace.object_set.get_objects(name).segmented
                self.nobjects = np.max(self.labels)
                if self.nobjects != 0:
                    self.range = np.arange(1, np.max(self.labels) + 1)
                    self.labels = self.labels.copy()
                    self.labels[~ im.mask] = 0
                    self.current_mean = fix(
                            scind.mean(im.pixel_data,
                                       self.labels,
                                       self.range))
                    self.start_mean = np.maximum(
                            self.current_mean, np.finfo(float).eps)

        object_records = [ObjectRecord(ob.objects_name.value)
                          for ob in image.objects]
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
        ng = image.granular_spectrum_length.value
        startmean = np.mean(pixels[mask])
        ero = pixels.copy()
        # Mask the test image so that masked pixels will have no effect
        # during reconstruction
        #
        ero[~mask] = 0
        currentmean = startmean
        startmean = max(startmean, np.finfo(float).eps)

        if im.dimensions is 2:
            footprint = skimage.morphology.disk(1, dtype=bool)
        else:
            footprint = skimage.morphology.ball(1, dtype=bool)
        statistics = [image.image_name.value]
        for i in range(1, ng + 1):
            prevmean = currentmean
            ero_mask = np.zeros_like(ero)
            ero_mask[mask == True] = ero[mask == True]
            ero = skimage.morphology.erosion(ero_mask, selem=footprint)
            rec = skimage.morphology.reconstruction(ero, pixels, selem=footprint)
            currentmean = np.mean(rec[mask])
            gs = (prevmean - currentmean) * 100 / startmean
            statistics += ["%.2f" % gs]
            feature = image.granularity_feature(i)
            measurements.add_image_measurement(feature, gs)
            #
            # Restore the reconstructed image to the shape of the
            # original image so we can match against object labels
            #
            orig_shape = im.pixel_data.shape
            if im.dimensions is 2:
                i, j = np.mgrid[0:orig_shape[0], 0:orig_shape[1]].astype(float)
                #
                # Make sure the mapping only references the index range of
                # back_pixels.
                #
                i *= float(new_shape[0] - 1) / float(orig_shape[0] - 1)
                j *= float(new_shape[1] - 1) / float(orig_shape[1] - 1)
                rec = scind.map_coordinates(rec, (i, j), order=1)
            else:
                k, i, j = np.mgrid[0:orig_shape[0], 0:orig_shape[1], 0:orig_shape[2]].astype(float)
                k *= float(new_shape[0] - 1) / float(orig_shape[0] - 1)
                i *= float(new_shape[1] - 1) / float(orig_shape[1] - 1)
                j *= float(new_shape[2] - 1) / float(orig_shape[2] - 1)
                rec = scind.map_coordinates(rec, (k, i, j), order=1)
            #
            # Calculate the means for the objects
            #
            for object_record in object_records:
                assert isinstance(object_record, ObjectRecord)
                if object_record.nobjects > 0:
                    new_mean = fix(scind.mean(rec, object_record.labels,
                                              object_record.range))
                    gss = ((object_record.current_mean - new_mean) * 100 /
                           object_record.start_mean)
                    object_record.current_mean = new_mean
                else:
                    gss = np.zeros((0,))
                measurements.add_measurement(object_record.name, feature, gss)
        return statistics

    def get_measurement_columns(self, pipeline, return_sources=False):
        result = []
        sources = []
        for image in self.images:
            gslength = image.granular_spectrum_length.value
            for i in range(1, gslength + 1):
                result += [(cpmeas.IMAGE,
                            image.granularity_feature(i),
                            cpmeas.COLTYPE_FLOAT)]
                sources += [(image.image_name, image.granularity_feature(i))]
            for ob in image.objects:
                for i in range(1, gslength + 1):
                    result += [(ob.objects_name.value,
                                image.granularity_feature(i),
                                cpmeas.COLTYPE_FLOAT)]
                    sources += [(ob.objects_name.value, image.granularity_feature(i))]

        if return_sources:
            return result, sources
        else:
            return result

    def get_matching_images(self, object_name):
        """Return all image records that match the given object name

        object_name - name of an object or IMAGE to match all
        """
        if object_name == cpmeas.IMAGE:
            return self.images
        return [image for image in self.images
                if object_name in [ob.objects_name.value
                                   for ob in image.objects]]

    def get_categories(self, pipeline, object_name):
        if len(self.get_matching_images(object_name)) > 0:
            return ['Granularity']
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        max_length = 0
        if category == 'Granularity':
            for image in self.get_matching_images(object_name):
                max_length = max(max_length, image.granular_spectrum_length.value)
        return [str(i) for i in range(1, max_length + 1)]

    def get_measurement_images(self, pipeline, object_name, category,
                               measurement):
        result = []
        if category == 'Granularity':
            try:
                length = int(measurement)
                if length <= 0:
                    return []
            except ValueError:
                return []
            for image in self.get_matching_images(object_name):
                if image.granular_spectrum_length.value >= length:
                    result.append(image.image_name.value)
        return result

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 1:
            # Matlab and pyCP v1 are identical
            from_matlab = False
            variable_revision_number = 1
        if variable_revision_number == 1:
            # changed to use cps.SettingsGroup() but did not change the
            # ordering of any of the settings
            variable_revision_number = 2
        if variable_revision_number == 2:
            # Changed to add objects and explicit image numbers
            image_count = int(len(setting_values) / IMAGE_SETTING_COUNT_V2)
            new_setting_values = [str(image_count)]
            for i in range(image_count):
                # Object setting count = 0
                new_setting_values += ["0"]
                new_setting_values += setting_values[:IMAGE_SETTING_COUNT_V2]
                setting_values = setting_values[IMAGE_SETTING_COUNT_V2:]
            setting_values = new_setting_values
            variable_revision_number = 3
        return setting_values, variable_revision_number, from_matlab

    def volumetric(self):
        return True


class GranularitySettingsGroup(cps.SettingsGroup):
    def granularity_feature(self, length):
        return C_GRANULARITY % (length, self.image_name.value)

    def add_objects(self):
        og = cps.SettingsGroup()
        og.append("objects_name", cps.ObjectNameSubscriber(
                "Select objects to measure", cps.NONE,
                doc="""Select the objects whose granualarity
            will be measured. You can select objects from prior modules
            that identify objects, such as <b>IdentifyPrimaryObjects</b>. If you only want to measure the granularity
            for the image overall, you can remove all objects using the "Remove this object" button."""))
        og.append("remover", cps.RemoveSettingButton(
                "", "Remove this object", self.objects, og))
        self.objects.append(og)

    def visible_settings(self):
        result = []
        if self.can_remove:
            result += [self.divider]
        result += [self.image_name, self.subsample_size, self.image_sample_size,
                   self.element_size, self.granular_spectrum_length]
        for ob in self.objects:
            result += [ob.objects_name, ob.remover]
        result += [self.add_objects_button]
        if self.can_remove:
            result += [self.remover]
        return result
