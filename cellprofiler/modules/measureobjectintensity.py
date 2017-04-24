"""
<b>Measure Object Intensity</b> measures several intensity features for identified objects.
<hr>
Given an image with objects identified (e.g. nuclei or cells), this module extracts intensity features for each object
based on one or more corresponding grayscale images. Measurements are recorded for each object.
<p>Intensity measurements are made for all combinations of the images and objects entered. If you want only specific
image/object measurements, you can use multiple MeasureObjectIntensity modules for each group of measurements
desired.</p>
<p>Note that for publication purposes, the units of intensity from microscopy images are usually described as
"Intensity units" or "Arbitrary intensity units" since microscopes are not calibrated to an absolute scale. Also, it is
important to note whether you are reporting either the mean or the integrated intensity, so specify "Mean intensity
units" or "Integrated intensity units" accordingly.</p>
<p>Keep in mind that the default behavior in CellProfiler is to rescale the image intensity from 0 to 1 by dividing all
pixels in the image by the maximum possible intensity value. This "maximum possible" value is defined by the "Set
intensity range from" setting in <b>NamesAndTypes</b>; see the help for that setting for more details.</p>
<h4>Available measurements</h4>
<ul>
    <li><i>IntegratedIntensity:</i> The sum of the pixel intensities within an object.</li>
    <li><i>MeanIntensity:</i> The average pixel intensity within an object.</li>
    <li><i>StdIntensity:</i> The standard deviation of the pixel intensities within an object.</li>
    <li><i>MaxIntensity:</i> The maximal pixel intensity within an object.</li>
    <li><i>MinIntensity:</i> The minimal pixel intensity within an object.</li>
    <li><i>IntegratedIntensityEdge:</i> The sum of the edge pixel intensities of an object.</li>
    <li><i>MeanIntensityEdge:</i> The average edge pixel intensity of an object.</li>
    <li><i>StdIntensityEdge:</i> The standard deviation of the edge pixel intensities of an object.</li>
    <li><i>MaxIntensityEdge:</i> The maximal edge pixel intensity of an object.</li>
    <li><i>MinIntensityEdge:</i> The minimal edge pixel intensity of an object.</li>
    <li><i>MassDisplacement:</i> The distance between the centers of gravity in the gray-level representation of the
    object and the binary representation of the object.</li>
    <li><i>LowerQuartileIntensity:</i> The intensity value of the pixel for which 25% of the pixels in the object have
    lower values.</li>
    <li><i>MedianIntensity:</i> The median intensity value within the object</li>
    <li><i>MADIntensity:</i> The median absolute deviation (MAD) value of the intensities within the object. The MAD is
    defined as the median(|x<sub>i</sub> - median(x)|).</li>
    <li><i>UpperQuartileIntensity:</i> The intensity value of the pixel for which 75% of the pixels in the object have
    lower values.</li>
    <li><i>Location_CenterMassIntensity_X, Location_CenterMassIntensity_Y:</i> The pixel (X,Y) coordinates of the
    intensity weighted centroid (= center of mass = first moment) of all pixels within the object.</li>
    <li><i>Location_MaxIntensity_X, Location_MaxIntensity_Y:</i> The pixel (X,Y) coordinates of the pixel with the
    maximum intensity within the object.</li>
</ul>See also <b>NamesAndTypes</b>, <b>MeasureImageIntensity</b>.
"""

import centrosome.cpmorphology
import centrosome.filter
import centrosome.outline
import numpy
import scipy.ndimage
import skimage.segmentation

import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting
import identify

INTENSITY = 'Intensity'
INTEGRATED_INTENSITY = 'IntegratedIntensity'
MEAN_INTENSITY = 'MeanIntensity'
STD_INTENSITY = 'StdIntensity'
MIN_INTENSITY = 'MinIntensity'
MAX_INTENSITY = 'MaxIntensity'
INTEGRATED_INTENSITY_EDGE = 'IntegratedIntensityEdge'
MEAN_INTENSITY_EDGE = 'MeanIntensityEdge'
STD_INTENSITY_EDGE = 'StdIntensityEdge'
MIN_INTENSITY_EDGE = 'MinIntensityEdge'
MAX_INTENSITY_EDGE = 'MaxIntensityEdge'
MASS_DISPLACEMENT = 'MassDisplacement'
LOWER_QUARTILE_INTENSITY = 'LowerQuartileIntensity'
MEDIAN_INTENSITY = 'MedianIntensity'
MAD_INTENSITY = 'MADIntensity'
UPPER_QUARTILE_INTENSITY = 'UpperQuartileIntensity'
LOC_CMI_X = 'CenterMassIntensity_X'
LOC_CMI_Y = 'CenterMassIntensity_Y'
LOC_CMI_Z = 'CenterMassIntensity_Z'
LOC_MAX_X = 'MaxIntensity_X'
LOC_MAX_Y = 'MaxIntensity_Y'
LOC_MAX_Z = 'MaxIntensity_Z'

ALL_MEASUREMENTS = [INTEGRATED_INTENSITY, MEAN_INTENSITY, STD_INTENSITY,
                    MIN_INTENSITY, MAX_INTENSITY, INTEGRATED_INTENSITY_EDGE,
                    MEAN_INTENSITY_EDGE, STD_INTENSITY_EDGE,
                    MIN_INTENSITY_EDGE, MAX_INTENSITY_EDGE,
                    MASS_DISPLACEMENT, LOWER_QUARTILE_INTENSITY,
                    MEDIAN_INTENSITY, MAD_INTENSITY, UPPER_QUARTILE_INTENSITY]
ALL_LOCATION_MEASUREMENTS = [LOC_CMI_X, LOC_CMI_Y, LOC_CMI_Z, LOC_MAX_X, LOC_MAX_Y, LOC_MAX_Z]


class MeasureObjectIntensity(cellprofiler.module.Module):
    module_name = "MeasureObjectIntensity"
    variable_revision_number = 3
    category = "Measurement"

    def create_settings(self):
        self.images = []
        self.add_image(can_remove=False)
        self.image_count = cellprofiler.setting.HiddenCount(self.images)
        self.add_image_button = cellprofiler.setting.DoSomething("", "Add another image", self.add_image)
        self.divider = cellprofiler.setting.Divider()
        self.objects = []
        self.add_object(can_remove=False)
        self.add_object_button = cellprofiler.setting.DoSomething("", "Add another object", self.add_object)

    def add_image(self, can_remove=True):
        '''Add an image to the image_groups collection

        can_delete - set this to False to keep from showing the "remove"
                     button for images that must be present.
        '''
        group = cellprofiler.setting.SettingsGroup()
        if can_remove:
            group.append("divider", cellprofiler.setting.Divider(line=False))
        group.append("name", cellprofiler.setting.ImageNameSubscriber(
                "Select an image to measure", cellprofiler.setting.NONE, doc="""
            Select the grayscale images whose intensity you want to measure."""))

        if can_remove:
            group.append("remover", cellprofiler.setting.RemoveSettingButton("", "Remove this image", self.images, group))
        self.images.append(group)

    def add_object(self, can_remove=True):
        '''Add an object to the object_groups collection

        can_delete - set this to False to keep from showing the "remove"
                     button for images that must be present.
        '''
        group = cellprofiler.setting.SettingsGroup()
        if can_remove:
            group.append("divider", cellprofiler.setting.Divider(line=False))
        group.append("name", cellprofiler.setting.ObjectNameSubscriber(
                "Select objects to measure", cellprofiler.setting.NONE, doc="""
            Select the objects whose intensities you want to measure."""))

        if can_remove:
            group.append("remover", cellprofiler.setting.RemoveSettingButton("", "Remove this object", self.objects, group))
        self.objects.append(group)

    def settings(self):
        result = [self.image_count]
        result += [im.name for im in self.images]
        result += [obj.name for obj in self.objects]
        return result

    def visible_settings(self):
        result = []
        for im in self.images:
            result += im.visible_settings()
        result += [self.add_image_button, self.divider]
        for im in self.objects:
            result += im.visible_settings()
        result += [self.add_object_button]
        return result

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        '''Adjust setting values if they came from a previous revision

        setting_values - a sequence of strings representing the settings
                         for the module as stored in the pipeline
        variable_revision_number - the variable revision number of the
                         module at the time the pipeline was saved. Use this
                         to determine how the incoming setting values map
                         to those of the current module version.
        module_name - the name of the module that did the saving. This can be
                      used to import the settings from another module if
                      that module was merged into the current module
        from_matlab - True if the settings came from a Matlab pipeline, False
                      if the settings are from a CellProfiler 2.0 pipeline.

        Overriding modules should return a tuple of setting_values,
        variable_revision_number and True if upgraded to CP 2.0, otherwise
        they should leave things as-is so that the caller can report
        an error.
        '''
        if from_matlab and variable_revision_number == 2:
            # Old matlab-style. Erase any setting values that are
            # "Do not use"
            new_setting_values = [setting_values[0], cellprofiler.setting.DO_NOT_USE]
            for setting_value in setting_values[1:]:
                if setting_value != cellprofiler.setting.DO_NOT_USE:
                    new_setting_values.append(setting_value)
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 2
        if variable_revision_number == 2:
            assert not from_matlab
            num_imgs = setting_values.index(cellprofiler.setting.DO_NOT_USE)
            setting_values = [str(num_imgs)] + setting_values[:num_imgs] + setting_values[num_imgs + 1:]
            variable_revision_number = 3
        return setting_values, variable_revision_number, from_matlab

    def prepare_settings(self, setting_values):
        """Do any sort of adjustment to the settings required for the given values

        setting_values - the values for the settings

        This method allows a module to specialize itself according to
        the number of settings and their value. For instance, a module that
        takes a variable number of images or objects can increase or decrease
        the number of relevant settings so they map correctly to the values.

        See cellprofiler.modules.measureobjectsizeshape for an example.
        """
        #
        # The settings have two parts - images, then objects
        # The parts are divided by the string, cps.DO_NOT_USE
        #
        image_count = int(setting_values[0])
        object_count = len(setting_values) - image_count - 1
        del self.images[image_count:]
        while len(self.images) < image_count:
            self.add_image()
        del self.objects[object_count:]
        while len(self.objects) < object_count:
            self.add_object()

    def validate_module(self, pipeline):
        """Make sure chosen objects and images are selected only once"""
        images = set()
        for group in self.images:
            if group.name.value in images:
                raise cellprofiler.setting.ValidationError(
                        "%s has already been selected" % group.name.value,
                        group.name)
            images.add(group.name.value)

        objects = set()
        for group in self.objects:
            if group.name.value in objects:
                raise cellprofiler.setting.ValidationError(
                        "%s has already been selected" % group.name.value,
                        group.name)
            objects.add(group.name.value)

    def get_measurement_columns(self, pipeline):
        '''Return the column definitions for measurements made by this module'''
        columns = []
        for image_name in [im.name for im in self.images]:
            for object_name in [obj.name for obj in self.objects]:
                for category, features in (
                        (INTENSITY, ALL_MEASUREMENTS),
                        (cellprofiler.measurement.C_LOCATION, ALL_LOCATION_MEASUREMENTS)):
                    for feature in features:
                        columns.append((object_name.value,
                                        "%s_%s_%s" % (category, feature,
                                                      image_name.value),
                                        cellprofiler.measurement.COLTYPE_FLOAT))

        return columns

    def get_categories(self, pipeline, object_name):
        """Get the categories of measurements supplied for the given object name

        pipeline - pipeline being run
        object_name - name of labels in question (or 'Images')
        returns a list of category names
        """
        for object_name_variable in [obj.name for obj in self.objects]:
            if object_name_variable.value == object_name:
                return [INTENSITY, cellprofiler.measurement.C_LOCATION]
        return []

    def get_measurements(self, pipeline, object_name, category):
        """Get the measurements made on the given object in the given category"""
        if category == cellprofiler.measurement.C_LOCATION:
            all_measurements = ALL_LOCATION_MEASUREMENTS
        elif category == INTENSITY:
            all_measurements = ALL_MEASUREMENTS
        else:
            return []
        for object_name_variable in [obj.name for obj in self.objects]:
            if object_name_variable.value == object_name:
                return all_measurements
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        """Get the images used to make the given measurement in the given category on the given object"""
        if category == INTENSITY:
            if measurement not in ALL_MEASUREMENTS:
                return []
        elif category == cellprofiler.measurement.C_LOCATION:
            if measurement not in ALL_LOCATION_MEASUREMENTS:
                return []
        else:
            return []
        for object_name_variable in [obj.name for obj in self.objects]:
            if object_name_variable == object_name:
                return [image.name.value for image in self.images]
        return []

    def run(self, workspace):
        if self.show_window:
            workspace.display_data.col_labels = (
                "Image", "Object", "Feature", "Mean", "Median", "STD")
            workspace.display_data.statistics = statistics = []
        for image_name in [img.name for img in self.images]:
            image = workspace.image_set.get_image(image_name.value,
                                                  must_be_grayscale=True)
            for object_name in [obj.name for obj in self.objects]:
                # Need to refresh image after each iteration...
                img = image.pixel_data
                if image.has_mask:
                    masked_image = img.copy()
                    masked_image[~image.mask] = 0
                    image_mask = image.mask
                else:
                    masked_image = img
                    image_mask = numpy.ones_like(img, dtype=numpy.bool)

                if image.dimensions == 2:
                    img = img.reshape(1, *img.shape)
                    masked_image = masked_image.reshape(1, *masked_image.shape)
                    image_mask = image_mask.reshape(1, *image_mask.shape)

                objects = workspace.object_set.get_objects(object_name.value)
                nobjects = objects.count
                integrated_intensity = numpy.zeros((nobjects,))
                integrated_intensity_edge = numpy.zeros((nobjects,))
                mean_intensity = numpy.zeros((nobjects,))
                mean_intensity_edge = numpy.zeros((nobjects,))
                std_intensity = numpy.zeros((nobjects,))
                std_intensity_edge = numpy.zeros((nobjects,))
                min_intensity = numpy.zeros((nobjects,))
                min_intensity_edge = numpy.zeros((nobjects,))
                max_intensity = numpy.zeros((nobjects,))
                max_intensity_edge = numpy.zeros((nobjects,))
                mass_displacement = numpy.zeros((nobjects,))
                lower_quartile_intensity = numpy.zeros((nobjects,))
                median_intensity = numpy.zeros((nobjects,))
                mad_intensity = numpy.zeros((nobjects,))
                upper_quartile_intensity = numpy.zeros((nobjects,))
                cmi_x = numpy.zeros((nobjects,))
                cmi_y = numpy.zeros((nobjects,))
                cmi_z = numpy.zeros((nobjects,))
                max_x = numpy.zeros((nobjects,))
                max_y = numpy.zeros((nobjects,))
                max_z = numpy.zeros((nobjects,))
                for labels, lindexes in objects.get_labels():
                    lindexes = lindexes[lindexes != 0]

                    if image.dimensions == 2:
                        labels = labels.reshape(1, *labels.shape)

                    labels, img = cellprofiler.object.crop_labels_and_image(labels, img)
                    _, masked_image = cellprofiler.object.crop_labels_and_image(labels, masked_image)
                    outlines = skimage.segmentation.find_boundaries(labels, mode='inner')

                    if image.has_mask:
                        _, mask = cellprofiler.object.crop_labels_and_image(labels, image_mask)
                        masked_labels = labels.copy()
                        masked_labels[~mask] = 0
                        masked_outlines = outlines.copy()
                        masked_outlines[~mask] = 0
                    else:
                        masked_labels = labels
                        masked_outlines = outlines

                    lmask = masked_labels > 0 & numpy.isfinite(img)  # Ignore NaNs, Infs
                    has_objects = numpy.any(lmask)
                    if has_objects:
                        limg = img[lmask]

                        llabels = labels[lmask]

                        mesh_z,\
                            mesh_y, \
                            mesh_x = numpy.mgrid[0:masked_image.shape[0], 0:masked_image.shape[1], 0:masked_image.shape[2]]

                        mesh_x = mesh_x[lmask]
                        mesh_y = mesh_y[lmask]
                        mesh_z = mesh_z[lmask]

                        lcount = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.sum(numpy.ones(len(limg)), llabels, lindexes))

                        integrated_intensity[lindexes - 1] = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.sum(limg, llabels, lindexes))

                        mean_intensity[lindexes - 1] = integrated_intensity[lindexes - 1] / lcount

                        std_intensity[lindexes - 1] = numpy.sqrt(
                            centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.mean((limg - mean_intensity[llabels - 1]) ** 2, llabels, lindexes))
                        )

                        min_intensity[lindexes - 1] = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.minimum(limg, llabels, lindexes))

                        max_intensity[lindexes - 1] = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.maximum(limg, llabels, lindexes))

                        # Compute the position of the intensity maximum
                        max_position = numpy.array(centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.maximum_position(limg, llabels, lindexes)), dtype=int)
                        max_position = numpy.reshape(max_position, (max_position.shape[0],))

                        max_x[lindexes - 1] = mesh_x[max_position]
                        max_y[lindexes - 1] = mesh_y[max_position]
                        max_z[lindexes - 1] = mesh_z[max_position]

                        # The mass displacement is the distance between the center
                        # of mass of the binary image and of the intensity image. The
                        # center of mass is the average X or Y for the binary image
                        # and the sum of X or Y * intensity / integrated intensity
                        cm_x = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.mean(mesh_x, llabels, lindexes))
                        cm_y = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.mean(mesh_y, llabels, lindexes))
                        cm_z = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.mean(mesh_z, llabels, lindexes))

                        i_x = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.sum(mesh_x * limg, llabels, lindexes))
                        i_y = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.sum(mesh_y * limg, llabels, lindexes))
                        i_z = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.sum(mesh_z * limg, llabels, lindexes))

                        cmi_x[lindexes - 1] = i_x / integrated_intensity[lindexes - 1]
                        cmi_y[lindexes - 1] = i_y / integrated_intensity[lindexes - 1]
                        cmi_z[lindexes - 1] = i_z / integrated_intensity[lindexes - 1]

                        diff_x = cm_x - cmi_x[lindexes - 1]
                        diff_y = cm_y - cmi_y[lindexes - 1]
                        diff_z = cm_z - cmi_z[lindexes - 1]

                        mass_displacement[lindexes - 1] = numpy.sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)

                        #
                        # Sort the intensities by label, then intensity.
                        # For each label, find the index above and below
                        # the 25%, 50% and 75% mark and take the weighted
                        # average.
                        #
                        order = numpy.lexsort((limg, llabels))
                        areas = lcount.astype(int)
                        indices = numpy.cumsum(areas) - areas
                        for dest, fraction in (
                                (lower_quartile_intensity, 1.0 / 4.0),
                                (median_intensity, 1.0 / 2.0),
                                (upper_quartile_intensity, 3.0 / 4.0)
                        ):
                            qindex = indices.astype(float) + areas * fraction
                            qfraction = qindex - numpy.floor(qindex)
                            qindex = qindex.astype(int)
                            qmask = qindex < indices + areas - 1
                            qi = qindex[qmask]
                            qf = qfraction[qmask]
                            dest[lindexes[qmask] - 1] = (limg[order[qi]] * (1 - qf) + limg[order[qi + 1]] * qf)

                            #
                            # In some situations (e.g. only 3 points), there may
                            # not be an upper bound.
                            #
                            qmask = (~qmask) & (areas > 0)
                            dest[lindexes[qmask] - 1] = limg[order[qindex[qmask]]]

                        #
                        # Once again, for the MAD
                        #
                        madimg = numpy.abs(limg - median_intensity[llabels - 1])
                        order = numpy.lexsort((madimg, llabels))
                        qindex = indices.astype(float) + areas / image.dimensions
                        qfraction = qindex - numpy.floor(qindex)
                        qindex = qindex.astype(int)
                        qmask = qindex < indices + areas - 1
                        qi = qindex[qmask]
                        qf = qfraction[qmask]
                        mad_intensity[lindexes[qmask] - 1] = (madimg[order[qi]] * (1 - qf) + madimg[order[qi + 1]] * qf)
                        qmask = (~qmask) & (areas > 0)
                        mad_intensity[lindexes[qmask] - 1] = madimg[order[qindex[qmask]]]

                    emask = masked_outlines > 0
                    eimg = img[emask]
                    elabels = labels[emask]
                    has_edge = len(eimg) > 0

                    if has_edge:
                        ecount = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.sum(numpy.ones(len(eimg)), elabels, lindexes))

                        integrated_intensity_edge[lindexes - 1] = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.sum(eimg, elabels, lindexes))

                        mean_intensity_edge[lindexes - 1] = integrated_intensity_edge[lindexes - 1] / ecount

                        std_intensity_edge[lindexes - 1] = numpy.sqrt(
                            centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.mean((eimg - mean_intensity_edge[elabels - 1]) ** 2, elabels, lindexes))
                        )

                        min_intensity_edge[lindexes - 1] = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.minimum(eimg, elabels, lindexes))

                        max_intensity_edge[lindexes - 1] = centrosome.cpmorphology.fixup_scipy_ndimage_result(scipy.ndimage.maximum(eimg, elabels, lindexes))

                m = workspace.measurements

                for category, feature_name, measurement in (
                        (INTENSITY, INTEGRATED_INTENSITY, integrated_intensity),
                        (INTENSITY, MEAN_INTENSITY, mean_intensity),
                        (INTENSITY, STD_INTENSITY, std_intensity),
                        (INTENSITY, MIN_INTENSITY, min_intensity),
                        (INTENSITY, MAX_INTENSITY, max_intensity),
                        (INTENSITY, INTEGRATED_INTENSITY_EDGE, integrated_intensity_edge),
                        (INTENSITY, MEAN_INTENSITY_EDGE, mean_intensity_edge),
                        (INTENSITY, STD_INTENSITY_EDGE, std_intensity_edge),
                        (INTENSITY, MIN_INTENSITY_EDGE, min_intensity_edge),
                        (INTENSITY, MAX_INTENSITY_EDGE, max_intensity_edge),
                        (INTENSITY, MASS_DISPLACEMENT, mass_displacement),
                        (INTENSITY, LOWER_QUARTILE_INTENSITY, lower_quartile_intensity),
                        (INTENSITY, MEDIAN_INTENSITY, median_intensity),
                        (INTENSITY, MAD_INTENSITY, mad_intensity),
                        (INTENSITY, UPPER_QUARTILE_INTENSITY, upper_quartile_intensity),
                        (cellprofiler.measurement.C_LOCATION, LOC_CMI_X, cmi_x),
                        (cellprofiler.measurement.C_LOCATION, LOC_CMI_Y, cmi_y),
                        (cellprofiler.measurement.C_LOCATION, LOC_CMI_Z, cmi_z),
                        (cellprofiler.measurement.C_LOCATION, LOC_MAX_X, max_x),
                        (cellprofiler.measurement.C_LOCATION, LOC_MAX_Y, max_y),
                        (cellprofiler.measurement.C_LOCATION, LOC_MAX_Z, max_z)
                ):
                    measurement_name = "{}_{}_{}".format(category, feature_name, image_name.value)
                    m.add_measurement(object_name.value, measurement_name, measurement)
                    if self.show_window and len(measurement) > 0:
                        statistics.append(
                            (
                                image_name.value,
                                object_name.value,
                                feature_name,
                                numpy.round(numpy.mean(measurement), 3),
                                numpy.round(numpy.median(measurement), 3),
                                numpy.round(numpy.std(measurement), 3)
                            )
                        )

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        figure.subplot_table(0, 0,
                             workspace.display_data.statistics,
                             col_labels=workspace.display_data.col_labels)

    def volumetric(self):
        return True
