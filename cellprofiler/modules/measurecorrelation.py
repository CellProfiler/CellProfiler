'''<b>Measure Correlation</b> measures the colocalization and correlation between intensities in different images (e.g., different color channels) on a pixel-by-pixel basis, within identified
objects or across an entire image.
<hr>
Given two or more images, this module calculates the correlation & colocalization (Overlap, Manders, Costes' Automated Threshold & Rank Weighted Colocalization) between the
pixel intensities. The correlation / colocalization can be measured for entire
images, or a correlation measurement can be made within each individual object.

Correlations / Colocalizations will be calculated between all pairs of images that are selected in
the module, as well as between selected objects. For example, if correlations
are to be measured for a set of red, green, and blue images containing identified nuclei,
measurements will be made between the following:
<ul>
<li>The blue and green, red and green, and red and blue images. </li>
<li>The nuclei in each of the above image pairs.</li>
</ul>

<h4>Available measurements</h4>
<ul>
<li><i>Correlation:</i> The correlation between a pair of images <i>I</i> and <i>J</i>,
calculated as Pearson's correlation coefficient. The formula is
covariance(<i>I</i> ,<i>J</i>)/[std(<i>I</i> ) &times; std(<i>J</i>)].</li>
<li><i>Slope:</i> The slope of the least-squares regression between a pair of images
I and J. Calculated using the model <i>A</i> &times; <i>I</i> + <i>B</i> = <i>J</i>, w
here <i>A</i> is the slope.</li>
<li><i>Overlap coefficient:</i> The overlap coefficient is a modification of Pearson's correlation
where average intesity values of the pixels are not subtracted from the original intesity values. For
a pair of images R and G, the overlap coefficient is measured as
r = sum(Ri * Gi) / sqrt (sum(Ri*Ri)*sum(Gi*Gi)).</li>
<li><i>Manders coefficient:</i> The Manders coefficient for a pair of images R and G is measured as
M1 = sum(Ri_coloc)/sum(Ri) and M2 = sum(Gi_coloc)/sum(Gi), where Ri_coloc = Ri when Gi > 0, 0 otherwise and
Gi_coloc =  Gi when Ri >0, 0 otherwise.</li>
<li><i>Manders coefficient (Costes Automated Threshold):</i> Costes' automated threshold estimates
maximum threshold of intensity for each image based on correlation. Manders coefficient is applied
on thresholded images as Ri_coloc = Ri when Gi > Gthr and Gi_coloc = Gi when Ri > Rthr where
Gthr and Rthr are thresholds calculed using Costes' authomated threshold method.</li>
<li><i>Rank Weighted Colocalization coefficient:</i> The RWC coefficient for a pair of images R and G is measured as
RWC1 = sum(Ri_coloc*Wi)/sum(Ri) and RWC2 = sum(Gi_coloc*Wi)/sum(Gi), where Wi is Weight defined as
Wi = (Rmax - Di)/Rmax where Rmax is the maximum of Ranks among R and G based on the max intensity, and
Di = abs(Rank(Ri) - Rank(Gi)) (absolute difference in ranks between R and G) and Ri_coloc = Ri when Gi > 0, 0 otherwise
and Gi_coloc =  Gi when Ri >0, 0 otherwise. (Singan et al. 2011, BMC Bioinformatics 12:407).
</li></ul>

'''

import numpy as np
import scipy.ndimage as scind
import scipy.stats as scistat
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from scipy.linalg import lstsq

import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.setting as cps

M_IMAGES = "Across entire image"
M_OBJECTS = "Within objects"
M_IMAGES_AND_OBJECTS = "Both"

'''Feature name format for the correlation measurement'''
F_CORRELATION_FORMAT = "Correlation_Correlation_%s_%s"

'''Feature name format for the slope measurement'''
F_SLOPE_FORMAT = "Correlation_Slope_%s_%s"

'''Feature name format for the overlap coefficient measurement'''
F_OVERLAP_FORMAT = "Correlation_Overlap_%s_%s"

'''Feature name format for the Manders Coefficient measurement'''
F_K_FORMAT = "Correlation_K_%s_%s"

'''Feature name format for the Manders Coefficient measurement'''
F_KS_FORMAT = "Correlation_KS_%s_%s"

'''Feature name format for the Manders Coefficient measurement'''
F_MANDERS_FORMAT = "Correlation_Manders_%s_%s"

'''Feature name format for the RWC Coefficient measurement'''
F_RWC_FORMAT = "Correlation_RWC_%s_%s"

'''Feature name format for the Costes Coefficient measurement'''
F_COSTES_FORMAT = "Correlation_Costes_%s_%s"


class MeasureCorrelation(cpm.Module):
    module_name = 'MeasureCorrelation'
    category = 'Measurement'
    variable_revision_number = 3

    def create_settings(self):
        '''Create the initial settings for the module'''
        self.image_groups = []
        self.add_image(can_delete=False)
        self.spacer_1 = cps.Divider()
        self.add_image(can_delete=False)
        self.image_count = cps.HiddenCount(self.image_groups)

        self.add_image_button = cps.DoSomething("", 'Add another image', self.add_image)
        self.spacer_2 = cps.Divider()
        self.thr = cps.Float(
                "Set threshold as percentage of maximum intensity for the images",
                15, minval=0, maxval=99, doc='''\
            Select the threshold as a percentage of the maximum intensity of the above image [0-99].''')

        self.images_or_objects = cps.Choice(
                'Select where to measure correlation',
                [M_IMAGES, M_OBJECTS, M_IMAGES_AND_OBJECTS], doc='''
            You can measure the correlation in several ways:
            <ul>
            <li><i>%(M_OBJECTS)s:</i> Measure correlation only in those pixels previously
            identified as an object. You will be asked to specify which object to measure from.</li>
            <li><i>%(M_IMAGES)s:</i> Measure the correlation across all pixels in the images.</li>
            <li><i>%(M_IMAGES_AND_OBJECTS)s:</i> Calculate both measurements above.</li>
            </ul>
            All methods measure correlation on a pixel by pixel basis.''' % globals())

        self.object_groups = []
        self.add_object(can_delete=False)
        self.object_count = cps.HiddenCount(self.object_groups)

        self.spacer_2 = cps.Divider(line=True)

        self.add_object_button = cps.DoSomething("", 'Add another object', self.add_object)

    def add_image(self, can_delete=True):
        '''Add an image to the image_groups collection

        can_delete - set this to False to keep from showing the "remove"
                     button for images that must be present.
        '''
        group = cps.SettingsGroup()
        if can_delete:
            group.append("divider", cps.Divider(line=False))
        group.append("image_name", cps.ImageNameSubscriber(
                'Select an image to measure', cps.NONE, doc='''
            Select an image to measure the correlation from.'''))

        if len(self.image_groups) == 0:  # Insert space between 1st two images for aesthetics
            group.append("extra_divider", cps.Divider(line=False))

        if can_delete:
            group.append("remover", cps.RemoveSettingButton("", "Remove this image", self.image_groups, group))

        self.image_groups.append(group)

    def add_object(self, can_delete=True):
        '''Add an object to the object_groups collection'''
        group = cps.SettingsGroup()
        if can_delete:
            group.append("divider", cps.Divider(line=False))

        group.append("object_name", cps.ObjectNameSubscriber(
                'Select an object to measure', cps.NONE, doc='''
            Select the objects to be measured.'''))

        if can_delete:
            group.append("remover", cps.RemoveSettingButton('', 'Remove this object', self.object_groups, group))
        self.object_groups.append(group)

    def settings(self):
        '''Return the settings to be saved in the pipeline'''
        result = [self.image_count, self.object_count]
        result += [image_group.image_name for image_group in self.image_groups]
        result += [self.thr]
        result += [self.images_or_objects]
        result += [object_group.object_name for object_group in self.object_groups]
        return result

    def prepare_settings(self, setting_values):
        '''Make sure there are the right number of image and object slots for the incoming settings'''
        image_count = int(setting_values[0])
        object_count = int(setting_values[1])
        if image_count < 2:
            raise ValueError(
                    "The MeasureCorrelate module must have at least two input images. %d found in pipeline file" % image_count)

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
        result += [self.add_image_button, self.spacer_2, self.thr, self.images_or_objects]
        if self.wants_objects():
            for object_group in self.object_groups:
                result += object_group.visible_settings()
            result += [self.add_object_button]
        return result

    def get_image_pairs(self):
        '''Yield all permutations of pairs of images to correlate

        Yields the pairs of images in a canonical order.
        '''
        for i in range(self.image_count.value - 1):
            for j in range(i + 1, self.image_count.value):
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
        col_labels = ["First image", "Second image", "Objects", "Measurement", "Value"]
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
        result = []
        if np.any(mask):
            #
            # Perform the correlation, which returns:
            # [ [ii, ij],
            #   [ji, jj] ]
            #
            fi = first_pixel_data[mask]
            si = second_pixel_data[mask]
            corr = np.corrcoef((fi, si))[1, 0]
            #
            # Find the slope as a linear regression to
            # A * i1 + B = i2
            #
            coeffs = lstsq(np.array((fi, np.ones_like(fi))).transpose(), si)[0]
            slope = coeffs[0]
            result += [[first_image_name, second_image_name, "-", "Correlation", "%.3f" % corr],
                       [first_image_name, second_image_name, "-", "Slope", "%.3f" % slope]]
            # Orthogonal Regression for Costes' automated threshold
            nonZero = (fi > 0) | (si > 0)

            xvar = np.var(fi[nonZero], axis=0, ddof=1)
            yvar = np.var(si[nonZero], axis=0, ddof=1)

            xmean = np.mean(fi[nonZero], axis=0)
            ymean = np.mean(si[nonZero], axis=0)

            z = fi[nonZero] + si[nonZero]
            zvar = np.var(z, axis=0, ddof=1)

            covar = 0.5 * (zvar - (xvar + yvar))

            denom = 2 * covar
            num = (yvar - xvar) + np.sqrt((yvar - xvar) * (yvar - xvar) + 4 * (covar * covar))
            a = (num / denom)
            b = (ymean - a * xmean)

            i = 1
            while i > 0.003921568627:
                Thr_fi_c = i
                Thr_si_c = (a * i) + b
                combt = (fi < Thr_fi_c) | (si < Thr_si_c)
                costReg = scistat.pearsonr(fi[combt], si[combt])
                if costReg[0] <= 0:
                    break
                i = i - 0.003921568627

            # Costes' thershold calculation
            combined_thresh_c = (fi > Thr_fi_c) & (si > Thr_si_c)
            fi_thresh_c = fi[combined_thresh_c]
            si_thresh_c = si[combined_thresh_c]
            tot_fi_thr_c = fi[(fi > Thr_fi_c)].sum()
            tot_si_thr_c = si[(si > Thr_si_c)].sum()

            # Threshold as percentage of maximum intensity in each channel
            thr_fi = self.thr.value * np.max(fi) / 100
            thr_si = self.thr.value * np.max(si) / 100
            combined_thresh = (fi > thr_fi) & (si > thr_si)
            fi_thresh = fi[combined_thresh]
            si_thresh = si[combined_thresh]
            tot_fi_thr = fi[(fi > thr_fi)].sum()
            tot_si_thr = si[(si > thr_si)].sum()

            # Manders Coefficient
            M1 = 0
            M2 = 0
            M1 = fi_thresh.sum() / tot_fi_thr
            M2 = si_thresh.sum() / tot_si_thr

            result += [[first_image_name, second_image_name, "-", "Manders Coefficient","%.3f"%M1],
                       [second_image_name, first_image_name, "-", "Manders Coefficient","%.3f"%M2]]

            # RWC Coefficient
            RWC1 = 0
            RWC2 = 0
            Rank1 = np.lexsort([fi])
            Rank2 = np.lexsort([si])
            Rank1_U = np.hstack([[False], fi[Rank1[:-1]] != fi[Rank1[1:]]])
            Rank2_U = np.hstack([[False], si[Rank2[:-1]] != si[Rank2[1:]]])
            Rank1_S = np.cumsum(Rank1_U)
            Rank2_S = np.cumsum(Rank2_U)
            Rank_im1 = np.zeros(fi.shape, dtype=int)
            Rank_im2 = np.zeros(si.shape, dtype=int)
            Rank_im1[Rank1] = Rank1_S
            Rank_im2[Rank2] = Rank2_S

            R = max(Rank_im1.max(), Rank_im2.max()) + 1
            Di = abs(Rank_im1 - Rank_im2)
            weight = ((R - Di) * 1.0) / R
            weight_thresh = weight[combined_thresh]
            RWC1 = (fi_thresh * weight_thresh).sum() / tot_fi_thr
            RWC2 = (si_thresh * weight_thresh).sum() / tot_si_thr
            result += [[first_image_name, second_image_name, "-", "RWC Coefficient", "%.3f" % RWC1],
                       [second_image_name, first_image_name, "-", "RWC Coefficient", "%.3f" % RWC2]]

            # Costes' Automated Threshold
            C1 = 0
            C2 = 0
            C1 = fi_thresh_c.sum() / tot_fi_thr_c
            C2 = si_thresh_c.sum() / tot_si_thr_c

            result += [[first_image_name, second_image_name, "-", "Manders Coefficient (Costes)","%.3f"%C1],
                       [second_image_name, first_image_name, "-", "Manders Coefficient (Costes)","%.3f"%C2]]

            # Overlap Coefficient
            overlap = 0
            overlap = (fi_thresh * si_thresh).sum() / np.sqrt((fi_thresh ** 2).sum() * (si_thresh ** 2).sum())
            K1 = (fi_thresh * si_thresh).sum() / (fi_thresh ** 2).sum()
            K2 = (fi_thresh * si_thresh).sum() / (si_thresh ** 2).sum()
            result += [[first_image_name, second_image_name, "-", "Overlap Coefficient", "%.3f" % overlap]]

        else:
            corr = np.NaN
            slope = np.NaN
            C1 = np.NaN
            C2 = np.NaN
            M1 = np.NaN
            M2 = np.NaN
            RWC1 = np.NaN
            RWC2 = np.NaN
            overlap = np.NaN
            K1 = np.NaN
            K2 = np.NaN

        #
        # Add the measurements
        #
        corr_measurement = F_CORRELATION_FORMAT % (first_image_name,
                                                   second_image_name)
        slope_measurement = F_SLOPE_FORMAT % (first_image_name,
                                              second_image_name)
        overlap_measurement = F_OVERLAP_FORMAT % (first_image_name,
                                                  second_image_name)
        k_measurement_1 = F_K_FORMAT % (first_image_name,
                                        second_image_name)
        k_measurement_2 = F_K_FORMAT % (second_image_name,
                                        first_image_name)
        manders_measurement_1 = F_MANDERS_FORMAT % (first_image_name,
                                                    second_image_name)
        manders_measurement_2 = F_MANDERS_FORMAT % (second_image_name,
                                                    first_image_name)
        rwc_measurement_1 = F_RWC_FORMAT % (first_image_name,
                                            second_image_name)
        rwc_measurement_2 = F_RWC_FORMAT % (second_image_name,
                                            first_image_name)
        costes_measurement_1 = F_COSTES_FORMAT % (first_image_name,
                                                  second_image_name)
        costes_measurement_2 = F_COSTES_FORMAT % (second_image_name,
                                                  first_image_name)

        workspace.measurements.add_image_measurement(corr_measurement, corr)
        workspace.measurements.add_image_measurement(slope_measurement, slope)
        workspace.measurements.add_image_measurement(overlap_measurement, overlap)
        workspace.measurements.add_image_measurement(k_measurement_1, K1)
        workspace.measurements.add_image_measurement(k_measurement_2, K2)
        workspace.measurements.add_image_measurement(manders_measurement_1, M1)
        workspace.measurements.add_image_measurement(manders_measurement_2, M2)
        workspace.measurements.add_image_measurement(rwc_measurement_1, RWC1)
        workspace.measurements.add_image_measurement(rwc_measurement_2, RWC2)
        workspace.measurements.add_image_measurement(costes_measurement_1, C1)
        workspace.measurements.add_image_measurement(costes_measurement_2, C2)

        return result

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
            first_pixels = objects.crop_image_similarly(first_image.pixel_data)
            first_mask = objects.crop_image_similarly(first_image.mask)
        except ValueError:
            first_pixels, m1 = cpo.size_similarly(labels, first_image.pixel_data)
            first_mask, m1 = cpo.size_similarly(labels, first_image.mask)
            first_mask[~m1] = False
        try:
            second_pixels = objects.crop_image_similarly(second_image.pixel_data)
            second_mask = objects.crop_image_similarly(second_image.mask)
        except ValueError:
            second_pixels, m1 = cpo.size_similarly(labels, second_image.pixel_data)
            second_mask, m1 = cpo.size_similarly(labels, second_image.mask)
            second_mask[~m1] = False
        mask = ((labels > 0) & first_mask & second_mask)
        first_pixels = first_pixels[mask]
        second_pixels = second_pixels[mask]
        labels = labels[mask]
        result = []
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

        n_objects = objects.count
        # Handle case when both images for the correlation are completely masked out

        if n_objects == 0:
            corr = np.zeros((0,))
            overlap = np.zeros((0,))
            K1 = np.zeros((0,))
            K2 = np.zeros((0,))
            M1 = np.zeros((0,))
            M2 = np.zeros((0,))
            RWC1 = np.zeros((0,))
            RWC2 = np.zeros((0,))
            C1 = np.zeros((0,))
            C2 = np.zeros((0,))
        elif np.where(mask)[0].__len__() == 0:
            corr = np.zeros((n_objects,))
            corr[:] = np.NaN
            overlap = K1 = K2 = M1 = M2 = RWC1 = RWC2 = C1 = C2 = corr
        else:
            #
            # The correlation is sum((x-mean(x))(y-mean(y)) /
            #                         ((n-1) * std(x) *std(y)))
            #
            lrange = np.arange(n_objects, dtype=np.int32) + 1
            area = fix(scind.sum(np.ones_like(labels), labels, lrange))
            mean1 = fix(scind.mean(first_pixels, labels, lrange))
            mean2 = fix(scind.mean(second_pixels, labels, lrange))
            #
            # Calculate the standard deviation times the population.
            #
            std1 = np.sqrt(fix(scind.sum((first_pixels - mean1[labels - 1]) ** 2,
                                         labels, lrange)))
            std2 = np.sqrt(fix(scind.sum((second_pixels - mean2[labels - 1]) ** 2,
                                         labels, lrange)))
            x = first_pixels - mean1[labels - 1]  # x - mean(x)
            y = second_pixels - mean2[labels - 1]  # y - mean(y)
            corr = fix(scind.sum(x * y / (std1[labels - 1] * std2[labels - 1]),
                                 labels, lrange))
            # Explicitly set the correlation to NaN for masked objects
            corr[scind.sum(1, labels, lrange) == 0] = np.NaN
            result += [
                [first_image_name, second_image_name, object_name, "Mean Correlation coeff", "%.3f" % np.mean(corr)],
                [first_image_name, second_image_name, object_name, "Median Correlation coeff",
                 "%.3f" % np.median(corr)],
                [first_image_name, second_image_name, object_name, "Min Correlation coeff", "%.3f" % np.min(corr)],
                [first_image_name, second_image_name, object_name, "Max Correlation coeff", "%.3f" % np.max(corr)]]

            # Threshold as percentage of maximum intensity of objects in each channel
            tff = (self.thr.value / 100) * fix(scind.maximum(first_pixels, labels, lrange))
            tss = (self.thr.value / 100) * fix(scind.maximum(second_pixels, labels, lrange))

            combined_thresh = (first_pixels > tff[labels - 1]) & (second_pixels > tss[labels - 1])
            fi_thresh = first_pixels[combined_thresh]
            si_thresh = second_pixels[combined_thresh]
            tot_fi_thr = scind.sum(first_pixels[first_pixels > tff[labels - 1]], labels[first_pixels > tff[labels - 1]],
                                   lrange)
            tot_si_thr = scind.sum(second_pixels[second_pixels > tss[labels - 1]],
                                   labels[second_pixels > tss[labels - 1]], lrange)

            nonZero = (fi > 0) | (si > 0)
            xvar = np.var(fi[nonZero], axis=0, ddof=1)
            yvar = np.var(si[nonZero], axis=0, ddof=1)

            xmean = np.mean(fi[nonZero], axis=0)
            ymean = np.mean(si[nonZero], axis=0)

            z = fi[nonZero] + si[nonZero]
            zvar = np.var(z, axis=0, ddof=1)

            covar = 0.5 * (zvar - (xvar + yvar))

            denom = 2 * covar
            num = (yvar - xvar) + np.sqrt((yvar - xvar) * (yvar - xvar) + 4 * (covar * covar))
            a = (num / denom)
            b = (ymean - a * xmean)

            i = 1
            while i > 0.003921568627:
                thr_fi_c = i
                thr_si_c = (a * i) + b
                combt = (fi < thr_fi_c) | (si < thr_si_c)
                costReg = scistat.pearsonr(fi[combt], si[combt])
                if costReg[0] <= 0:
                    break
                i = i - 0.003921568627

            # Costes' thershold for entire image is applied to each object
            fi_above_thr = first_pixels > thr_fi_c
            si_above_thr = second_pixels > thr_si_c
            combined_thresh_c = fi_above_thr & si_above_thr
            fi_thresh_c = first_pixels[combined_thresh_c]
            si_thresh_c = second_pixels[combined_thresh_c]
            if np.any(fi_above_thr):
                tot_fi_thr_c = scind.sum(first_pixels[first_pixels > thr_fi_c], labels[first_pixels > thr_fi_c], lrange)
            else:
                tot_fi_thr_c = np.zeros(len(lrange))
            if np.any(si_above_thr):
                tot_si_thr_c = scind.sum(second_pixels[second_pixels > thr_si_c], labels[second_pixels > thr_si_c],
                                         lrange)
            else:
                tot_si_thr_c = np.zeros(len(lrange))

            # Manders Coefficient
            M1 = np.zeros(len(lrange))
            M2 = np.zeros(len(lrange))

            if np.any(combined_thresh):
                M1 = np.array(scind.sum(fi_thresh,labels[combined_thresh],lrange)) / np.array(tot_fi_thr)
                M2 = np.array(scind.sum(si_thresh,labels[combined_thresh],lrange)) / np.array(tot_si_thr)
            result += [[first_image_name, second_image_name, object_name,"Mean Manders coeff","%.3f"%np.mean(M1)],
                       [first_image_name, second_image_name, object_name,"Median Manders coeff","%.3f"%np.median(M1)],
                       [first_image_name, second_image_name, object_name,"Min Manders coeff","%.3f"%np.min(M1)],
                       [first_image_name, second_image_name, object_name,"Max Manders coeff","%.3f"%np.max(M1)]]
            result += [[second_image_name, first_image_name, object_name,"Mean Manders coeff","%.3f"%np.mean(M2)],
                       [second_image_name, first_image_name, object_name,"Median Manders coeff","%.3f"%np.median(M2)],
                       [second_image_name, first_image_name, object_name,"Min Manders coeff","%.3f"%np.min(M2)],
                       [second_image_name, first_image_name, object_name,"Max Manders coeff","%.3f"%np.max(M2)]]

            # RWC Coefficient
            RWC1 = np.zeros(len(lrange))
            RWC2 = np.zeros(len(lrange))
            [Rank1] = np.lexsort(([labels], [first_pixels]))
            [Rank2] = np.lexsort(([labels], [second_pixels]))
            Rank1_U = np.hstack([[False], first_pixels[Rank1[:-1]] != first_pixels[Rank1[1:]]])
            Rank2_U = np.hstack([[False], second_pixels[Rank2[:-1]] != second_pixels[Rank2[1:]]])
            Rank1_S = np.cumsum(Rank1_U)
            Rank2_S = np.cumsum(Rank2_U)
            Rank_im1 = np.zeros(first_pixels.shape, dtype=int)
            Rank_im2 = np.zeros(second_pixels.shape, dtype=int)
            Rank_im1[Rank1] = Rank1_S
            Rank_im2[Rank2] = Rank2_S

            R = max(Rank_im1.max(), Rank_im2.max()) + 1
            Di = abs(Rank_im1 - Rank_im2)
            weight = (R - Di) * 1.0 / R
            weight_thresh = weight[combined_thresh]
            if np.any(combined_thresh_c):
                RWC1 = np.array(scind.sum(fi_thresh * weight_thresh, labels[combined_thresh], lrange)) / np.array(
                        tot_fi_thr)
                RWC2 = np.array(scind.sum(si_thresh * weight_thresh, labels[combined_thresh], lrange)) / np.array(
                        tot_si_thr)

            result += [[first_image_name, second_image_name, object_name, "Mean RWC coeff", "%.3f" % np.mean(RWC1)],
                       [first_image_name, second_image_name, object_name, "Median RWC coeff", "%.3f" % np.median(RWC1)],
                       [first_image_name, second_image_name, object_name, "Min RWC coeff", "%.3f" % np.min(RWC1)],
                       [first_image_name, second_image_name, object_name, "Max RWC coeff", "%.3f" % np.max(RWC1)]]
            result += [[second_image_name, first_image_name, object_name, "Mean RWC coeff", "%.3f" % np.mean(RWC2)],
                       [second_image_name, first_image_name, object_name, "Median RWC coeff", "%.3f" % np.median(RWC2)],
                       [second_image_name, first_image_name, object_name, "Min RWC coeff", "%.3f" % np.min(RWC2)],
                       [second_image_name, first_image_name, object_name, "Max RWC coeff", "%.3f" % np.max(RWC2)]]

            # Costes Automated Threshold
            C1 = np.zeros(len(lrange))
            C2 = np.zeros(len(lrange))
            if np.any(combined_thresh_c):
                C1 = np.array(scind.sum(fi_thresh_c,labels[combined_thresh_c],lrange)) / np.array(tot_fi_thr_c)
                C2 = np.array(scind.sum(si_thresh_c,labels[combined_thresh_c],lrange)) / np.array(tot_si_thr_c)
            result += [[first_image_name, second_image_name, object_name,"Mean Manders coeff (Costes)","%.3f"%np.mean(C1)],
                       [first_image_name, second_image_name, object_name,"Median Manders coeff (Costes)","%.3f"%np.median(C1)],
                       [first_image_name, second_image_name, object_name,"Min Manders coeff (Costes)","%.3f"%np.min(C1)],
                       [first_image_name, second_image_name, object_name,"Max Manders coeff (Costes)","%.3f"%np.max(C1)]
                       ]
            result += [[second_image_name, first_image_name, object_name,"Mean Manders coeff (Costes)","%.3f"%np.mean(C2)],
                       [second_image_name, first_image_name, object_name,"Median Manders coeff (Costes)","%.3f"%np.median(C2)],
                       [second_image_name, first_image_name, object_name,"Min Manders coeff (Costes)","%.3f"%np.min(C2)],
                       [second_image_name, first_image_name, object_name,"Max Manders coeff (Costes)","%.3f"%np.max(C2)]
                       ]

            # Overlap Coefficient
            fpsq = scind.sum(first_pixels[combined_thresh] ** 2, labels[combined_thresh], lrange)
            spsq = scind.sum(second_pixels[combined_thresh] ** 2, labels[combined_thresh], lrange)
            pdt = np.sqrt(np.array(fpsq) * np.array(spsq))

            if np.any(combined_thresh):
                overlap = fix(
                        scind.sum(first_pixels[combined_thresh] * second_pixels[combined_thresh],
                                  labels[combined_thresh],
                                  lrange) / pdt)
                K1 = fix((scind.sum(first_pixels[combined_thresh] * second_pixels[combined_thresh],
                                    labels[combined_thresh], lrange)) / (np.array(fpsq)))
                K2 = fix(
                        scind.sum(first_pixels[combined_thresh] * second_pixels[combined_thresh],
                                  labels[combined_thresh],
                                  lrange) / np.array(spsq))
            else:
                overlap = K1 = K2 = np.zeros(len(lrange))
            result += [
                [first_image_name, second_image_name, object_name, "Mean Overlap coeff", "%.3f" % np.mean(overlap)],
                [first_image_name, second_image_name, object_name, "Median Overlap coeff", "%.3f" % np.median(overlap)],
                [first_image_name, second_image_name, object_name, "Min Overlap coeff", "%.3f" % np.min(overlap)],
                [first_image_name, second_image_name, object_name, "Max Overlap coeff", "%.3f" % np.max(overlap)]]

        measurement = ("Correlation_Correlation_%s_%s" %
                       (first_image_name, second_image_name))
        overlap_measurement = (F_OVERLAP_FORMAT % (first_image_name,
                                                   second_image_name))
        k_measurement_1 = (F_K_FORMAT % (first_image_name,
                                         second_image_name))
        k_measurement_2 = (F_K_FORMAT % (second_image_name,
                                         first_image_name))
        manders_measurement_1 = (F_MANDERS_FORMAT % (first_image_name,
                                                     second_image_name))
        manders_measurement_2 = (F_MANDERS_FORMAT % (second_image_name,
                                                     first_image_name))
        rwc_measurement_1 = (F_RWC_FORMAT % (first_image_name,
                                             second_image_name))
        rwc_measurement_2 = (F_RWC_FORMAT % (second_image_name,
                                             first_image_name))
        costes_measurement_1 = (F_COSTES_FORMAT % (first_image_name,
                                                   second_image_name))
        costes_measurement_2 = (F_COSTES_FORMAT % (second_image_name,
                                                   first_image_name))

        workspace.measurements.add_measurement(object_name, measurement, corr)
        workspace.measurements.add_measurement(object_name, overlap_measurement, overlap)
        workspace.measurements.add_measurement(object_name, k_measurement_1, K1)
        workspace.measurements.add_measurement(object_name, k_measurement_2, K2)
        workspace.measurements.add_measurement(object_name, manders_measurement_1, M1)
        workspace.measurements.add_measurement(object_name, manders_measurement_2, M2)
        workspace.measurements.add_measurement(object_name, rwc_measurement_1, RWC1)
        workspace.measurements.add_measurement(object_name, rwc_measurement_2, RWC2)
        workspace.measurements.add_measurement(object_name, costes_measurement_1, C1)
        workspace.measurements.add_measurement(object_name, costes_measurement_2, C2)

        if n_objects == 0:
            return [[first_image_name, second_image_name, object_name,
                     "Mean correlation", "-"],
                    [first_image_name, second_image_name, object_name,
                     "Median correlation", "-"],
                    [first_image_name, second_image_name, object_name,
                     "Min correlation", "-"],
                    [first_image_name, second_image_name, object_name,
                     "Max correlation", "-"]]
        else:
            return result

    def get_measurement_columns(self, pipeline):
        '''Return column definitions for all measurements made by this module'''
        columns = []
        for first_image, second_image in self.get_image_pairs():
            if self.wants_images():
                columns += [(cpmeas.IMAGE,
                             F_CORRELATION_FORMAT % (first_image, second_image),
                             cpmeas.COLTYPE_FLOAT),
                            (cpmeas.IMAGE,
                             F_SLOPE_FORMAT % (first_image, second_image),
                             cpmeas.COLTYPE_FLOAT),
                            (cpmeas.IMAGE,
                             F_OVERLAP_FORMAT % (first_image, second_image),
                             cpmeas.COLTYPE_FLOAT),
                            (cpmeas.IMAGE,
                             F_K_FORMAT % (first_image, second_image),
                             cpmeas.COLTYPE_FLOAT),
                            (cpmeas.IMAGE,
                             F_K_FORMAT % (second_image, first_image),
                             cpmeas.COLTYPE_FLOAT),
                            (cpmeas.IMAGE,
                             F_MANDERS_FORMAT % (first_image, second_image),
                             cpmeas.COLTYPE_FLOAT),
                            (cpmeas.IMAGE,
                             F_MANDERS_FORMAT % (second_image, first_image),
                             cpmeas.COLTYPE_FLOAT),
                            (cpmeas.IMAGE,
                             F_RWC_FORMAT % (first_image, second_image),
                             cpmeas.COLTYPE_FLOAT),
                            (cpmeas.IMAGE,
                             F_RWC_FORMAT % (second_image, first_image),
                             cpmeas.COLTYPE_FLOAT),
                            (cpmeas.IMAGE,
                             F_COSTES_FORMAT % (first_image, second_image),
                             cpmeas.COLTYPE_FLOAT),
                            (cpmeas.IMAGE,
                             F_COSTES_FORMAT % (second_image, first_image),
                             cpmeas.COLTYPE_FLOAT)]

            if self.wants_objects():
                for i in range(self.object_count.value):
                    object_name = self.object_groups[i].object_name.value
                    columns += [(object_name,
                                 F_CORRELATION_FORMAT %
                                 (first_image, second_image),
                                 cpmeas.COLTYPE_FLOAT),
                                (object_name,
                                 F_OVERLAP_FORMAT %
                                 (first_image, second_image),
                                 cpmeas.COLTYPE_FLOAT),
                                (object_name,
                                 F_K_FORMAT %
                                 (first_image, second_image),
                                 cpmeas.COLTYPE_FLOAT),
                                (object_name,
                                 F_K_FORMAT %
                                 (second_image, first_image),
                                 cpmeas.COLTYPE_FLOAT),
                                (object_name,
                                 F_MANDERS_FORMAT %
                                 (first_image, second_image),
                                 cpmeas.COLTYPE_FLOAT),
                                (object_name,
                                 F_MANDERS_FORMAT %
                                 (second_image, first_image),
                                 cpmeas.COLTYPE_FLOAT),
                                (object_name,
                                 F_RWC_FORMAT %
                                 (first_image, second_image),
                                 cpmeas.COLTYPE_FLOAT),
                                (object_name,
                                 F_RWC_FORMAT %
                                 (second_image, first_image),
                                 cpmeas.COLTYPE_FLOAT),
                                (object_name,
                                 F_COSTES_FORMAT %
                                 (first_image, second_image),
                                 cpmeas.COLTYPE_FLOAT),
                                (object_name,
                                 F_COSTES_FORMAT %
                                 (second_image, first_image),
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
                return ["Correlation", "Slope", "Overlap",
                        "K", "Manders", "RWC", "Costes"]
            else:
                return ["Correlation", "Overlap", "K", "Manders", "RWC", "Costes"]
        return []

    def get_measurement_images(self, pipeline, object_name, category,
                               measurement):
        '''Return the joined pairs of images measured'''
        result = []
        if measurement in self.get_measurements(pipeline, object_name, category):
            for i1, i2 in self.get_image_pairs():
                result.append("%s_%s" % (i1, i2))
                # For asymmetric, return both orderings
                if measurement in ("K", "Manders", "RWC", "Costes"):
                    result.append("%s_%s" % (i2, i1))
        return result

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        '''Adjust the setting values for pipelines saved under old revisions'''
        if from_matlab and variable_revision_number == 3:
            image_names = [x for x in setting_values[:4]
                           if x.upper() != cps.DO_NOT_USE.upper()]
            wants_image_measured = np.any([x == cpmeas.IMAGE
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
            image_names = setting_values[2:(image_count + 2)]
            m = setting_values[image_count + 2]
            object_names = setting_values[(image_count + 3):]
            if m == "Images":
                m = M_IMAGES
            elif m == "Objects":
                m = M_OBJECTS
            elif m == "Images and objects":
                m = M_IMAGES_AND_OBJECTS
            setting_values = ([str(image_count), str(object_count)] +
                              image_names + [m] + object_names)
            variable_revision_number = 2
        if variable_revision_number == 2:
            image_count = int(setting_values[0])
            idx_thr = image_count + 2
            setting_values = \
                setting_values[:idx_thr] + ["15.0"] + setting_values[idx_thr:]
            variable_revision_number = 3

        return setting_values, variable_revision_number, from_matlab

    def volumetric(self):
        return True
