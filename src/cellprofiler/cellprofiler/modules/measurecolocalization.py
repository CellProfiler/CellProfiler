"""
MeasureColocalization
=====================

**MeasureColocalization** measures the colocalization and correlation
between intensities in different images (e.g., different color channels)
on a pixel-by-pixel basis, within identified objects or across an entire
image.

Given two or more images, this module calculates the correlation &
colocalization (Overlap, Manders, Costes’ Automated Threshold & Rank
Weighted Colocalization) between the pixel intensities. The correlation
/ colocalization can be measured for entire images, or a correlation
measurement can be made within each individual object. Correlations /
Colocalizations will be calculated between all pairs of images that are
selected in the module, as well as between selected objects. For
example, if correlations are to be measured for a set of red, green, and
blue images containing identified nuclei, measurements will be made
between the following:

-  The blue and green, red and green, and red and blue images.
-  The nuclei in each of the above image pairs.

A good primer on colocalization theory can be found on the `SVI website`_.

You can find a helpful review on colocalization from Aaron *et al*. `here`_.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Correlation:* The correlation between a pair of images *I* and *J*,
   calculated as Pearson’s correlation coefficient. The formula is
   covariance(\ *I* ,\ *J*)/[std(\ *I* ) × std(\ *J*)].
-  *Slope:* The slope of the least-squares regression between a pair of
   images I and J. Calculated using the model *A* × *I* + *B* = *J*, where *A* is the slope.
-  *Overlap coefficient:* The overlap coefficient is a modification of
   Pearson’s correlation where average intensity values of the pixels are
   not subtracted from the original intensity values. For a pair of
   images R and G, the overlap coefficient is measured as r = sum(Ri \*
   Gi) / sqrt (sum(Ri\*Ri)\*sum(Gi\*Gi)).
-  *Manders coefficient:* The Manders coefficient for a pair of images R
   and G is measured as M1 = sum(Ri_coloc)/sum(Ri) and M2 =
   sum(Gi_coloc)/sum(Gi), where Ri_coloc = Ri when Gi > 0, 0 otherwise
   and Gi_coloc = Gi when Ri >0, 0 otherwise.
-  *Manders coefficient (Costes Automated Threshold):* Costes’ automated
   threshold estimates maximum threshold of intensity for each image
   based on correlation. Manders coefficient is applied on thresholded
   images as Ri_coloc = Ri when Gi > Gthr and Gi_coloc = Gi when Ri >
   Rthr where Gthr and Rthr are thresholds calculated using Costes’
   automated threshold method.
-  *Rank Weighted Colocalization coefficient:* The RWC coefficient for a
   pair of images R and G is measured as RWC1 =
   sum(Ri_coloc\*Wi)/sum(Ri) and RWC2 = sum(Gi_coloc\*Wi)/sum(Gi),
   where Wi is Weight defined as Wi = (Rmax - Di)/Rmax where Rmax is the
   maximum of Ranks among R and G based on the max intensity, and Di =
   abs(Rank(Ri) - Rank(Gi)) (absolute difference in ranks between R and
   G) and Ri_coloc = Ri when Gi > 0, 0 otherwise and Gi_coloc = Gi
   when Ri >0, 0 otherwise. (Singan et al. 2011, BMC Bioinformatics
   12:407).

References
^^^^^^^^^^

-  Aaron JS, Taylor AB, Chew TL. Image co-localization - co-occurrence versus correlation.
   J Cell Sci. 2018;131(3):jcs211847. Published 2018 Feb 8. doi:10.1242/jcs.211847


   
.. _SVI website: http://svi.nl/ColocalizationTheory
.. _here: https://jcs.biologists.org/content/joces/131/3/jcs211847.full.pdf
"""

import numpy
import scipy.ndimage
import scipy.stats
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Divider, Binary, ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import (
    LabelListSubscriber,
    ImageListSubscriber,
)
from cellprofiler_core.setting.text import Float
from cellprofiler_core.utilities.core.object import size_similarly
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from scipy.linalg import lstsq

M_IMAGES = "Across entire image"
M_OBJECTS = "Within objects"
M_IMAGES_AND_OBJECTS = "Both"

M_FAST = "Fast"
M_FASTER = "Faster"
M_ACCURATE = "Accurate"

"""Feature name format for the correlation measurement"""
F_CORRELATION_FORMAT = "Correlation_Correlation_%s_%s"

"""Feature name format for the slope measurement"""
F_SLOPE_FORMAT = "Correlation_Slope_%s_%s"

"""Feature name format for the overlap coefficient measurement"""
F_OVERLAP_FORMAT = "Correlation_Overlap_%s_%s"

"""Feature name format for the Manders Coefficient measurement"""
F_K_FORMAT = "Correlation_K_%s_%s"

"""Feature name format for the Manders Coefficient measurement"""
F_KS_FORMAT = "Correlation_KS_%s_%s"

"""Feature name format for the Manders Coefficient measurement"""
F_MANDERS_FORMAT = "Correlation_Manders_%s_%s"

"""Feature name format for the RWC Coefficient measurement"""
F_RWC_FORMAT = "Correlation_RWC_%s_%s"

"""Feature name format for the Costes Coefficient measurement"""
F_COSTES_FORMAT = "Correlation_Costes_%s_%s"


class MeasureColocalization(Module):
    module_name = "MeasureColocalization"
    category = "Measurement"
    variable_revision_number = 5

    def create_settings(self):
        """Create the initial settings for the module"""

        self.images_list = ImageListSubscriber(
            "Select images to measure",
            [],
            doc="""Select images to measure the correlation/colocalization in.""",
        )

        self.objects_list = LabelListSubscriber(
            "Select objects to measure",
            [],
            doc="""\
*(Used only when "Within objects" or "Both" are selected)*

Select the objects to be measured.""",
        )

        self.thr = Float(
            "Set threshold as percentage of maximum intensity for the images",
            15,
            minval=0,
            maxval=99,
            doc="""\
You may choose to measure colocalization metrics only for those pixels above 
a certain threshold. Select the threshold as a percentage of the maximum intensity 
of the above image [0-99].

This value is used by the Overlap, Manders, and Rank Weighted Colocalization 
measurements.
""",
        )

        self.images_or_objects = Choice(
            "Select where to measure correlation",
            [M_IMAGES, M_OBJECTS, M_IMAGES_AND_OBJECTS],
            doc="""\
You can measure the correlation in several ways:

-  *%(M_OBJECTS)s:* Measure correlation only in those pixels previously
   identified as within an object. You will be asked to choose which object
   type to measure within.
-  *%(M_IMAGES)s:* Measure the correlation across all pixels in the
   images.
-  *%(M_IMAGES_AND_OBJECTS)s:* Calculate both measurements above.

All methods measure correlation on a pixel by pixel basis.
"""
            % globals(),
        )

        self.spacer = Divider(line=True)

        self.do_all = Binary(
            "Run all metrics?",
            True,
            doc="""\
Select *{YES}* to run all of CellProfiler's correlation 
and colocalization algorithms on your images and/or objects; 
otherwise select *{NO}* to pick which correlation and 
colocalization algorithms to run.
""".format(
                **{"YES": "Yes", "NO": "No"}
            ),
        )

        self.do_corr_and_slope = Binary(
            "Calculate correlation and slope metrics?",
            True,
            doc="""\
Select *{YES}* to run the Pearson correlation and slope metrics.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.do_manders = Binary(
            "Calculate the Manders coefficients?",
            True,
            doc="""\
Select *{YES}* to run the Manders coefficients.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.do_rwc = Binary(
            "Calculate the Rank Weighted Colocalization coefficients?",
            True,
            doc="""\
Select *{YES}* to run the Rank Weighted Colocalization coefficients.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.do_overlap = Binary(
            "Calculate the Overlap coefficients?",
            True,
            doc="""\
Select *{YES}* to run the Overlap coefficients.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.do_costes = Binary(
            "Calculate the Manders coefficients using Costes auto threshold?",
            True,
            doc="""\
Select *{YES}* to run the Manders coefficients using Costes auto threshold.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.fast_costes = Choice(
            "Method for Costes thresholding",
            [M_FASTER, M_FAST, M_ACCURATE],
            doc=f"""\
This setting determines the method used to calculate the threshold for use within the
Costes calculations. The *{M_FAST}* and *{M_ACCURATE}* modes will test candidate thresholds
in descending order until the optimal threshold is reached. Selecting *{M_FAST}* will attempt 
to skip candidates when results are far from the optimal value being sought. Selecting *{M_ACCURATE}* 
will test every possible threshold value. When working with 16-bit images these methods can be extremely 
time-consuming. Selecting *{M_FASTER}* will use a modified bisection algorithm to find the threshold 
using a shrinking window of candidates. This is substantially faster but may produce slightly lower 
thresholds in exceptional circumstances.

In the vast majority of instances the results of all strategies should be identical. We recommend using 
*{M_FAST}* mode when working with 8-bit images and *{M_FASTER}* mode when using 16-bit images.

Alternatively, you may want to disable these specific measurements entirely 
(available when "*Run All Metrics?*" is set to "*No*").
"""
        )

    def settings(self):
        """Return the settings to be saved in the pipeline"""
        result = [
            self.images_list,
            self.thr,
            self.images_or_objects,
            self.objects_list,
            self.do_all,
            self.do_corr_and_slope,
            self.do_manders,
            self.do_rwc,
            self.do_overlap,
            self.do_costes,
            self.fast_costes,
        ]
        return result

    def visible_settings(self):
        result = [
            self.images_list,
            self.spacer,
            self.thr,
            self.images_or_objects,
        ]
        if self.wants_objects():
            result += [self.objects_list]
        result += [self.do_all]
        if not self.do_all:
            result += [
                self.do_corr_and_slope,
                self.do_manders,
                self.do_rwc,
                self.do_overlap,
                self.do_costes,
            ]
        if self.do_all or self.do_costes:
            result += [self.fast_costes]
        return result

    def help_settings(self):
        """Return the settings to be displayed in the help menu"""
        help_settings = [
            self.images_or_objects,
            self.thr,
            self.images_list,
            self.objects_list,
            self.do_all,
            self.fast_costes,
        ]
        return help_settings

    def get_image_pairs(self):
        """Yield all permutations of pairs of images to correlate

        Yields the pairs of images in a canonical order.
        """
        for i in range(len(self.images_list.value) - 1):
            for j in range(i + 1, len(self.images_list.value)):
                yield (
                    self.images_list.value[i],
                    self.images_list.value[j],
                )

    def wants_images(self):
        """True if the user wants to measure correlation on whole images"""
        return self.images_or_objects in (M_IMAGES, M_IMAGES_AND_OBJECTS)

    def wants_objects(self):
        """True if the user wants to measure per-object correlations"""
        return self.images_or_objects in (M_OBJECTS, M_IMAGES_AND_OBJECTS)

    def run(self, workspace):
        """Calculate measurements on an image set"""
        col_labels = ["First image", "Second image", "Objects", "Measurement", "Value"]
        statistics = []
        if len(self.images_list.value) < 2:
            raise ValueError("At least 2 images must be selected for analysis.")
        for first_image_name, second_image_name in self.get_image_pairs():
            if self.wants_images():
                statistics += self.run_image_pair_images(
                    workspace, first_image_name, second_image_name
                )
            if self.wants_objects():
                for object_name in self.objects_list.value:
                    statistics += self.run_image_pair_objects(
                        workspace, first_image_name, second_image_name, object_name
                    )
        if self.show_window:
            workspace.display_data.statistics = statistics
            workspace.display_data.col_labels = col_labels

    def display(self, workspace, figure):
        statistics = workspace.display_data.statistics
        if self.wants_objects():
            helptext = "default"
        else:
            helptext = None
        figure.set_subplots((1, 1))
        figure.subplot_table(
            0, 0, statistics, workspace.display_data.col_labels, title=helptext
        )

    def run_image_pair_images(self, workspace, first_image_name, second_image_name):
        """Calculate the correlation between the pixels of two images"""
        first_image = workspace.image_set.get_image(
            first_image_name, must_be_grayscale=True
        )
        second_image = workspace.image_set.get_image(
            second_image_name, must_be_grayscale=True
        )
        first_pixel_data = first_image.pixel_data
        first_mask = first_image.mask
        first_pixel_count = numpy.product(first_pixel_data.shape)
        second_pixel_data = second_image.pixel_data
        second_mask = second_image.mask
        second_pixel_count = numpy.product(second_pixel_data.shape)
        #
        # Crop the larger image similarly to the smaller one
        #
        if first_pixel_count < second_pixel_count:
            second_pixel_data = first_image.crop_image_similarly(second_pixel_data)
            second_mask = first_image.crop_image_similarly(second_mask)
        elif second_pixel_count < first_pixel_count:
            first_pixel_data = second_image.crop_image_similarly(first_pixel_data)
            first_mask = second_image.crop_image_similarly(first_mask)
        mask = (
            first_mask
            & second_mask
            & (~numpy.isnan(first_pixel_data))
            & (~numpy.isnan(second_pixel_data))
        )
        result = []
        if numpy.any(mask):
            fi = first_pixel_data[mask]
            si = second_pixel_data[mask]

            if self.do_corr_and_slope:
                #
                # Perform the correlation, which returns:
                # [ [ii, ij],
                #   [ji, jj] ]
                #
                corr = numpy.corrcoef((fi, si))[1, 0]
                #
                # Find the slope as a linear regression to
                # A * i1 + B = i2
                #
                coeffs = lstsq(numpy.array((fi, numpy.ones_like(fi))).transpose(), si)[
                    0
                ]
                slope = coeffs[0]
                result += [
                    [
                        first_image_name,
                        second_image_name,
                        "-",
                        "Correlation",
                        "%.3f" % corr,
                    ],
                    [first_image_name, second_image_name, "-", "Slope", "%.3f" % slope],
                ]

            if any((self.do_manders, self.do_rwc, self.do_overlap)):
                # Threshold as percentage of maximum intensity in each channel
                thr_fi = self.thr.value * numpy.max(fi) / 100
                thr_si = self.thr.value * numpy.max(si) / 100
                combined_thresh = (fi > thr_fi) & (si > thr_si)
                fi_thresh = fi[combined_thresh]
                si_thresh = si[combined_thresh]
                tot_fi_thr = fi[(fi > thr_fi)].sum()
                tot_si_thr = si[(si > thr_si)].sum()

            if self.do_manders:
                # Manders Coefficient
                M1 = 0
                M2 = 0
                M1 = fi_thresh.sum() / tot_fi_thr
                M2 = si_thresh.sum() / tot_si_thr

                result += [
                    [
                        first_image_name,
                        second_image_name,
                        "-",
                        "Manders Coefficient",
                        "%.3f" % M1,
                    ],
                    [
                        second_image_name,
                        first_image_name,
                        "-",
                        "Manders Coefficient",
                        "%.3f" % M2,
                    ],
                ]

            if self.do_rwc:
                # RWC Coefficient
                RWC1 = 0
                RWC2 = 0
                Rank1 = numpy.lexsort([fi])
                Rank2 = numpy.lexsort([si])
                Rank1_U = numpy.hstack([[False], fi[Rank1[:-1]] != fi[Rank1[1:]]])
                Rank2_U = numpy.hstack([[False], si[Rank2[:-1]] != si[Rank2[1:]]])
                Rank1_S = numpy.cumsum(Rank1_U)
                Rank2_S = numpy.cumsum(Rank2_U)
                Rank_im1 = numpy.zeros(fi.shape, dtype=int)
                Rank_im2 = numpy.zeros(si.shape, dtype=int)
                Rank_im1[Rank1] = Rank1_S
                Rank_im2[Rank2] = Rank2_S

                R = max(Rank_im1.max(), Rank_im2.max()) + 1
                Di = abs(Rank_im1 - Rank_im2)
                weight = ((R - Di) * 1.0) / R
                weight_thresh = weight[combined_thresh]
                RWC1 = (fi_thresh * weight_thresh).sum() / tot_fi_thr
                RWC2 = (si_thresh * weight_thresh).sum() / tot_si_thr
                result += [
                    [
                        first_image_name,
                        second_image_name,
                        "-",
                        "RWC Coefficient",
                        "%.3f" % RWC1,
                    ],
                    [
                        second_image_name,
                        first_image_name,
                        "-",
                        "RWC Coefficient",
                        "%.3f" % RWC2,
                    ],
                ]

            if self.do_overlap:
                # Overlap Coefficient
                overlap = 0
                overlap = (fi_thresh * si_thresh).sum() / numpy.sqrt(
                    (fi_thresh ** 2).sum() * (si_thresh ** 2).sum()
                )
                K1 = (fi_thresh * si_thresh).sum() / (fi_thresh ** 2).sum()
                K2 = (fi_thresh * si_thresh).sum() / (si_thresh ** 2).sum()
                result += [
                    [
                        first_image_name,
                        second_image_name,
                        "-",
                        "Overlap Coefficient",
                        "%.3f" % overlap,
                    ]
                ]

            if self.do_costes:
                # Orthogonal Regression for Costes' automated threshold
                scale = get_scale(first_image.scale, second_image.scale)
                if self.fast_costes == M_FASTER:
                    thr_fi_c, thr_si_c = self.bisection_costes(fi, si, scale)
                else:
                    thr_fi_c, thr_si_c = self.linear_costes(fi, si, scale)

                # Costes' thershold calculation
                combined_thresh_c = (fi > thr_fi_c) & (si > thr_si_c)
                fi_thresh_c = fi[combined_thresh_c]
                si_thresh_c = si[combined_thresh_c]
                tot_fi_thr_c = fi[(fi > thr_fi_c)].sum()
                tot_si_thr_c = si[(si > thr_si_c)].sum()

                # Costes' Automated Threshold
                C1 = 0
                C2 = 0
                C1 = fi_thresh_c.sum() / tot_fi_thr_c
                C2 = si_thresh_c.sum() / tot_si_thr_c

                result += [
                    [
                        first_image_name,
                        second_image_name,
                        "-",
                        "Manders Coefficient (Costes)",
                        "%.3f" % C1,
                    ],
                    [
                        second_image_name,
                        first_image_name,
                        "-",
                        "Manders Coefficient (Costes)",
                        "%.3f" % C2,
                    ],
                ]

        else:
            corr = numpy.NaN
            slope = numpy.NaN
            C1 = numpy.NaN
            C2 = numpy.NaN
            M1 = numpy.NaN
            M2 = numpy.NaN
            RWC1 = numpy.NaN
            RWC2 = numpy.NaN
            overlap = numpy.NaN
            K1 = numpy.NaN
            K2 = numpy.NaN

        #
        # Add the measurements
        #
        if self.do_corr_and_slope:
            corr_measurement = F_CORRELATION_FORMAT % (
                first_image_name,
                second_image_name,
            )
            slope_measurement = F_SLOPE_FORMAT % (first_image_name, second_image_name)
            workspace.measurements.add_image_measurement(corr_measurement, corr)
            workspace.measurements.add_image_measurement(slope_measurement, slope)
        if self.do_overlap:
            overlap_measurement = F_OVERLAP_FORMAT % (
                first_image_name,
                second_image_name,
            )
            k_measurement_1 = F_K_FORMAT % (first_image_name, second_image_name)
            k_measurement_2 = F_K_FORMAT % (second_image_name, first_image_name)
            workspace.measurements.add_image_measurement(overlap_measurement, overlap)
            workspace.measurements.add_image_measurement(k_measurement_1, K1)
            workspace.measurements.add_image_measurement(k_measurement_2, K2)
        if self.do_manders:
            manders_measurement_1 = F_MANDERS_FORMAT % (
                first_image_name,
                second_image_name,
            )
            manders_measurement_2 = F_MANDERS_FORMAT % (
                second_image_name,
                first_image_name,
            )
            workspace.measurements.add_image_measurement(manders_measurement_1, M1)
            workspace.measurements.add_image_measurement(manders_measurement_2, M2)
        if self.do_rwc:
            rwc_measurement_1 = F_RWC_FORMAT % (first_image_name, second_image_name)
            rwc_measurement_2 = F_RWC_FORMAT % (second_image_name, first_image_name)
            workspace.measurements.add_image_measurement(rwc_measurement_1, RWC1)
            workspace.measurements.add_image_measurement(rwc_measurement_2, RWC2)
        if self.do_costes:
            costes_measurement_1 = F_COSTES_FORMAT % (
                first_image_name,
                second_image_name,
            )
            costes_measurement_2 = F_COSTES_FORMAT % (
                second_image_name,
                first_image_name,
            )
            workspace.measurements.add_image_measurement(costes_measurement_1, C1)
            workspace.measurements.add_image_measurement(costes_measurement_2, C2)

        return result

    def run_image_pair_objects(
        self, workspace, first_image_name, second_image_name, object_name
    ):
        """Calculate per-object correlations between intensities in two images"""
        first_image = workspace.image_set.get_image(
            first_image_name, must_be_grayscale=True
        )
        second_image = workspace.image_set.get_image(
            second_image_name, must_be_grayscale=True
        )
        objects = workspace.object_set.get_objects(object_name)
        #
        # Crop both images to the size of the labels matrix
        #
        labels = objects.segmented
        try:
            first_pixels = objects.crop_image_similarly(first_image.pixel_data)
            first_mask = objects.crop_image_similarly(first_image.mask)
        except ValueError:
            first_pixels, m1 = size_similarly(labels, first_image.pixel_data)
            first_mask, m1 = size_similarly(labels, first_image.mask)
            first_mask[~m1] = False
        try:
            second_pixels = objects.crop_image_similarly(second_image.pixel_data)
            second_mask = objects.crop_image_similarly(second_image.mask)
        except ValueError:
            second_pixels, m1 = size_similarly(labels, second_image.pixel_data)
            second_mask, m1 = size_similarly(labels, second_image.mask)
            second_mask[~m1] = False
        mask = (labels > 0) & first_mask & second_mask
        first_pixels = first_pixels[mask]
        second_pixels = second_pixels[mask]
        labels = labels[mask]
        result = []
        first_pixel_data = first_image.pixel_data
        first_mask = first_image.mask
        first_pixel_count = numpy.product(first_pixel_data.shape)
        second_pixel_data = second_image.pixel_data
        second_mask = second_image.mask
        second_pixel_count = numpy.product(second_pixel_data.shape)
        #
        # Crop the larger image similarly to the smaller one
        #
        if first_pixel_count < second_pixel_count:
            second_pixel_data = first_image.crop_image_similarly(second_pixel_data)
            second_mask = first_image.crop_image_similarly(second_mask)
        elif second_pixel_count < first_pixel_count:
            first_pixel_data = second_image.crop_image_similarly(first_pixel_data)
            first_mask = second_image.crop_image_similarly(first_mask)
        mask = (
            first_mask
            & second_mask
            & (~numpy.isnan(first_pixel_data))
            & (~numpy.isnan(second_pixel_data))
        )
        if numpy.any(mask):
            fi = first_pixel_data[mask]
            si = second_pixel_data[mask]

        n_objects = objects.count
        # Handle case when both images for the correlation are completely masked out

        if n_objects == 0:
            corr = numpy.zeros((0,))
            overlap = numpy.zeros((0,))
            K1 = numpy.zeros((0,))
            K2 = numpy.zeros((0,))
            M1 = numpy.zeros((0,))
            M2 = numpy.zeros((0,))
            RWC1 = numpy.zeros((0,))
            RWC2 = numpy.zeros((0,))
            C1 = numpy.zeros((0,))
            C2 = numpy.zeros((0,))
        elif numpy.where(mask)[0].__len__() == 0:
            corr = numpy.zeros((n_objects,))
            corr[:] = numpy.NaN
            overlap = K1 = K2 = M1 = M2 = RWC1 = RWC2 = C1 = C2 = corr
        else:
            lrange = numpy.arange(n_objects, dtype=numpy.int32) + 1

            if self.do_corr_and_slope:
                #
                # The correlation is sum((x-mean(x))(y-mean(y)) /
                #                         ((n-1) * std(x) *std(y)))
                #

                mean1 = fix(scipy.ndimage.mean(first_pixels, labels, lrange))
                mean2 = fix(scipy.ndimage.mean(second_pixels, labels, lrange))
                #
                # Calculate the standard deviation times the population.
                #
                std1 = numpy.sqrt(
                    fix(
                        scipy.ndimage.sum(
                            (first_pixels - mean1[labels - 1]) ** 2, labels, lrange
                        )
                    )
                )
                std2 = numpy.sqrt(
                    fix(
                        scipy.ndimage.sum(
                            (second_pixels - mean2[labels - 1]) ** 2, labels, lrange
                        )
                    )
                )
                x = first_pixels - mean1[labels - 1]  # x - mean(x)
                y = second_pixels - mean2[labels - 1]  # y - mean(y)
                corr = fix(
                    scipy.ndimage.sum(
                        x * y / (std1[labels - 1] * std2[labels - 1]), labels, lrange
                    )
                )
                # Explicitly set the correlation to NaN for masked objects
                corr[scipy.ndimage.sum(1, labels, lrange) == 0] = numpy.NaN
                result += [
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Mean Correlation coeff",
                        "%.3f" % numpy.mean(corr),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Median Correlation coeff",
                        "%.3f" % numpy.median(corr),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Min Correlation coeff",
                        "%.3f" % numpy.min(corr),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Max Correlation coeff",
                        "%.3f" % numpy.max(corr),
                    ],
                ]

            if any((self.do_manders, self.do_rwc, self.do_overlap)):
                # Threshold as percentage of maximum intensity of objects in each channel
                tff = (self.thr.value / 100) * fix(
                    scipy.ndimage.maximum(first_pixels, labels, lrange)
                )
                tss = (self.thr.value / 100) * fix(
                    scipy.ndimage.maximum(second_pixels, labels, lrange)
                )

                combined_thresh = (first_pixels >= tff[labels - 1]) & (
                    second_pixels >= tss[labels - 1]
                )
                fi_thresh = first_pixels[combined_thresh]
                si_thresh = second_pixels[combined_thresh]
                tot_fi_thr = scipy.ndimage.sum(
                    first_pixels[first_pixels >= tff[labels - 1]],
                    labels[first_pixels >= tff[labels - 1]],
                    lrange,
                )
                tot_si_thr = scipy.ndimage.sum(
                    second_pixels[second_pixels >= tss[labels - 1]],
                    labels[second_pixels >= tss[labels - 1]],
                    lrange,
                )

            if self.do_manders:
                # Manders Coefficient
                M1 = numpy.zeros(len(lrange))
                M2 = numpy.zeros(len(lrange))

                if numpy.any(combined_thresh):
                    M1 = numpy.array(
                        scipy.ndimage.sum(fi_thresh, labels[combined_thresh], lrange)
                    ) / numpy.array(tot_fi_thr)
                    M2 = numpy.array(
                        scipy.ndimage.sum(si_thresh, labels[combined_thresh], lrange)
                    ) / numpy.array(tot_si_thr)
                result += [
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Mean Manders coeff",
                        "%.3f" % numpy.mean(M1),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Median Manders coeff",
                        "%.3f" % numpy.median(M1),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Min Manders coeff",
                        "%.3f" % numpy.min(M1),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Max Manders coeff",
                        "%.3f" % numpy.max(M1),
                    ],
                ]
                result += [
                    [
                        second_image_name,
                        first_image_name,
                        object_name,
                        "Mean Manders coeff",
                        "%.3f" % numpy.mean(M2),
                    ],
                    [
                        second_image_name,
                        first_image_name,
                        object_name,
                        "Median Manders coeff",
                        "%.3f" % numpy.median(M2),
                    ],
                    [
                        second_image_name,
                        first_image_name,
                        object_name,
                        "Min Manders coeff",
                        "%.3f" % numpy.min(M2),
                    ],
                    [
                        second_image_name,
                        first_image_name,
                        object_name,
                        "Max Manders coeff",
                        "%.3f" % numpy.max(M2),
                    ],
                ]

            if self.do_rwc:
                # RWC Coefficient
                RWC1 = numpy.zeros(len(lrange))
                RWC2 = numpy.zeros(len(lrange))
                [Rank1] = numpy.lexsort(([labels], [first_pixels]))
                [Rank2] = numpy.lexsort(([labels], [second_pixels]))
                Rank1_U = numpy.hstack(
                    [[False], first_pixels[Rank1[:-1]] != first_pixels[Rank1[1:]]]
                )
                Rank2_U = numpy.hstack(
                    [[False], second_pixels[Rank2[:-1]] != second_pixels[Rank2[1:]]]
                )
                Rank1_S = numpy.cumsum(Rank1_U)
                Rank2_S = numpy.cumsum(Rank2_U)
                Rank_im1 = numpy.zeros(first_pixels.shape, dtype=int)
                Rank_im2 = numpy.zeros(second_pixels.shape, dtype=int)
                Rank_im1[Rank1] = Rank1_S
                Rank_im2[Rank2] = Rank2_S

                R = max(Rank_im1.max(), Rank_im2.max()) + 1
                Di = abs(Rank_im1 - Rank_im2)
                weight = (R - Di) * 1.0 / R
                weight_thresh = weight[combined_thresh]

                if numpy.any(combined_thresh):
                    RWC1 = numpy.array(
                        scipy.ndimage.sum(
                            fi_thresh * weight_thresh, labels[combined_thresh], lrange
                        )
                    ) / numpy.array(tot_fi_thr)
                    RWC2 = numpy.array(
                        scipy.ndimage.sum(
                            si_thresh * weight_thresh, labels[combined_thresh], lrange
                        )
                    ) / numpy.array(tot_si_thr)

                result += [
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Mean RWC coeff",
                        "%.3f" % numpy.mean(RWC1),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Median RWC coeff",
                        "%.3f" % numpy.median(RWC1),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Min RWC coeff",
                        "%.3f" % numpy.min(RWC1),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Max RWC coeff",
                        "%.3f" % numpy.max(RWC1),
                    ],
                ]
                result += [
                    [
                        second_image_name,
                        first_image_name,
                        object_name,
                        "Mean RWC coeff",
                        "%.3f" % numpy.mean(RWC2),
                    ],
                    [
                        second_image_name,
                        first_image_name,
                        object_name,
                        "Median RWC coeff",
                        "%.3f" % numpy.median(RWC2),
                    ],
                    [
                        second_image_name,
                        first_image_name,
                        object_name,
                        "Min RWC coeff",
                        "%.3f" % numpy.min(RWC2),
                    ],
                    [
                        second_image_name,
                        first_image_name,
                        object_name,
                        "Max RWC coeff",
                        "%.3f" % numpy.max(RWC2),
                    ],
                ]

            if self.do_overlap:
                # Overlap Coefficient
                if numpy.any(combined_thresh):
                    fpsq = scipy.ndimage.sum(
                        first_pixels[combined_thresh] ** 2,
                        labels[combined_thresh],
                        lrange,
                    )
                    spsq = scipy.ndimage.sum(
                        second_pixels[combined_thresh] ** 2,
                        labels[combined_thresh],
                        lrange,
                    )
                    pdt = numpy.sqrt(numpy.array(fpsq) * numpy.array(spsq))

                    overlap = fix(
                        scipy.ndimage.sum(
                            first_pixels[combined_thresh]
                            * second_pixels[combined_thresh],
                            labels[combined_thresh],
                            lrange,
                        )
                        / pdt
                    )
                    K1 = fix(
                        (
                            scipy.ndimage.sum(
                                first_pixels[combined_thresh]
                                * second_pixels[combined_thresh],
                                labels[combined_thresh],
                                lrange,
                            )
                        )
                        / (numpy.array(fpsq))
                    )
                    K2 = fix(
                        scipy.ndimage.sum(
                            first_pixels[combined_thresh]
                            * second_pixels[combined_thresh],
                            labels[combined_thresh],
                            lrange,
                        )
                        / numpy.array(spsq)
                    )
                else:
                    overlap = K1 = K2 = numpy.zeros(len(lrange))
                result += [
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Mean Overlap coeff",
                        "%.3f" % numpy.mean(overlap),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Median Overlap coeff",
                        "%.3f" % numpy.median(overlap),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Min Overlap coeff",
                        "%.3f" % numpy.min(overlap),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Max Overlap coeff",
                        "%.3f" % numpy.max(overlap),
                    ],
                ]

            if self.do_costes:
                # Orthogonal Regression for Costes' automated threshold
                scale = get_scale(first_image.scale, second_image.scale)

                if self.fast_costes == M_FASTER:
                    thr_fi_c, thr_si_c = self.bisection_costes(fi, si, scale)
                else:
                    thr_fi_c, thr_si_c = self.linear_costes(fi, si, scale)

                # Costes' thershold for entire image is applied to each object
                fi_above_thr = first_pixels > thr_fi_c
                si_above_thr = second_pixels > thr_si_c
                combined_thresh_c = fi_above_thr & si_above_thr
                fi_thresh_c = first_pixels[combined_thresh_c]
                si_thresh_c = second_pixels[combined_thresh_c]
                if numpy.any(fi_above_thr):
                    tot_fi_thr_c = scipy.ndimage.sum(
                        first_pixels[first_pixels >= thr_fi_c],
                        labels[first_pixels >= thr_fi_c],
                        lrange,
                    )
                else:
                    tot_fi_thr_c = numpy.zeros(len(lrange))
                if numpy.any(si_above_thr):
                    tot_si_thr_c = scipy.ndimage.sum(
                        second_pixels[second_pixels >= thr_si_c],
                        labels[second_pixels >= thr_si_c],
                        lrange,
                    )
                else:
                    tot_si_thr_c = numpy.zeros(len(lrange))

                # Costes Automated Threshold
                C1 = numpy.zeros(len(lrange))
                C2 = numpy.zeros(len(lrange))
                if numpy.any(combined_thresh_c):
                    C1 = numpy.array(
                        scipy.ndimage.sum(
                            fi_thresh_c, labels[combined_thresh_c], lrange
                        )
                    ) / numpy.array(tot_fi_thr_c)
                    C2 = numpy.array(
                        scipy.ndimage.sum(
                            si_thresh_c, labels[combined_thresh_c], lrange
                        )
                    ) / numpy.array(tot_si_thr_c)
                result += [
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Mean Manders coeff (Costes)",
                        "%.3f" % numpy.mean(C1),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Median Manders coeff (Costes)",
                        "%.3f" % numpy.median(C1),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Min Manders coeff (Costes)",
                        "%.3f" % numpy.min(C1),
                    ],
                    [
                        first_image_name,
                        second_image_name,
                        object_name,
                        "Max Manders coeff (Costes)",
                        "%.3f" % numpy.max(C1),
                    ],
                ]
                result += [
                    [
                        second_image_name,
                        first_image_name,
                        object_name,
                        "Mean Manders coeff (Costes)",
                        "%.3f" % numpy.mean(C2),
                    ],
                    [
                        second_image_name,
                        first_image_name,
                        object_name,
                        "Median Manders coeff (Costes)",
                        "%.3f" % numpy.median(C2),
                    ],
                    [
                        second_image_name,
                        first_image_name,
                        object_name,
                        "Min Manders coeff (Costes)",
                        "%.3f" % numpy.min(C2),
                    ],
                    [
                        second_image_name,
                        first_image_name,
                        object_name,
                        "Max Manders coeff (Costes)",
                        "%.3f" % numpy.max(C2),
                    ],
                ]

        if self.do_corr_and_slope:
            measurement = "Correlation_Correlation_%s_%s" % (
                first_image_name,
                second_image_name,
            )
            workspace.measurements.add_measurement(object_name, measurement, corr)
        if self.do_manders:
            manders_measurement_1 = F_MANDERS_FORMAT % (
                first_image_name,
                second_image_name,
            )
            manders_measurement_2 = F_MANDERS_FORMAT % (
                second_image_name,
                first_image_name,
            )
            workspace.measurements.add_measurement(
                object_name, manders_measurement_1, M1
            )
            workspace.measurements.add_measurement(
                object_name, manders_measurement_2, M2
            )
        if self.do_rwc:
            rwc_measurement_1 = F_RWC_FORMAT % (first_image_name, second_image_name)
            rwc_measurement_2 = F_RWC_FORMAT % (second_image_name, first_image_name)
            workspace.measurements.add_measurement(object_name, rwc_measurement_1, RWC1)
            workspace.measurements.add_measurement(object_name, rwc_measurement_2, RWC2)
        if self.do_overlap:
            overlap_measurement = F_OVERLAP_FORMAT % (
                first_image_name,
                second_image_name,
            )
            k_measurement_1 = F_K_FORMAT % (first_image_name, second_image_name)
            k_measurement_2 = F_K_FORMAT % (second_image_name, first_image_name)
            workspace.measurements.add_measurement(
                object_name, overlap_measurement, overlap
            )
            workspace.measurements.add_measurement(object_name, k_measurement_1, K1)
            workspace.measurements.add_measurement(object_name, k_measurement_2, K2)
        if self.do_costes:
            costes_measurement_1 = F_COSTES_FORMAT % (
                first_image_name,
                second_image_name,
            )
            costes_measurement_2 = F_COSTES_FORMAT % (
                second_image_name,
                first_image_name,
            )
            workspace.measurements.add_measurement(
                object_name, costes_measurement_1, C1
            )
            workspace.measurements.add_measurement(
                object_name, costes_measurement_2, C2
            )

        if n_objects == 0:
            return [
                [
                    first_image_name,
                    second_image_name,
                    object_name,
                    "Mean correlation",
                    "-",
                ],
                [
                    first_image_name,
                    second_image_name,
                    object_name,
                    "Median correlation",
                    "-",
                ],
                [
                    first_image_name,
                    second_image_name,
                    object_name,
                    "Min correlation",
                    "-",
                ],
                [
                    first_image_name,
                    second_image_name,
                    object_name,
                    "Max correlation",
                    "-",
                ],
            ]
        else:
            return result

    def linear_costes(self, fi, si, scale_max=255):
        """
        Finds the Costes Automatic Threshold for colocalization using a linear algorithm.
        Candiate thresholds are gradually decreased until Pearson R falls below 0.
        If "Fast" mode is enabled the "steps" between tested thresholds will be increased
        when Pearson R is much greater than 0.
        """
        i_step = 1 / scale_max
        non_zero = (fi > 0) | (si > 0)
        xvar = numpy.var(fi[non_zero], axis=0, ddof=1)
        yvar = numpy.var(si[non_zero], axis=0, ddof=1)

        xmean = numpy.mean(fi[non_zero], axis=0)
        ymean = numpy.mean(si[non_zero], axis=0)

        z = fi[non_zero] + si[non_zero]
        zvar = numpy.var(z, axis=0, ddof=1)

        covar = 0.5 * (zvar - (xvar + yvar))

        denom = 2 * covar
        num = (yvar - xvar) + numpy.sqrt(
            (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
        )
        a = num / denom
        b = ymean - a * xmean

        # Start at 1 step above the maximum value
        img_max = max(fi.max(), si.max())
        i = i_step * ((img_max // i_step) + 1)

        num_true = None
        fi_max = fi.max()
        si_max = si.max()

        # Initialise without a threshold
        costReg, _ = scipy.stats.pearsonr(fi, si)
        thr_fi_c = i
        thr_si_c = (a * i) + b
        while i > fi_max and (a * i) + b > si_max:
            i -= i_step
        while i > i_step:
            thr_fi_c = i
            thr_si_c = (a * i) + b
            combt = (fi < thr_fi_c) | (si < thr_si_c)
            try:
                # Only run pearsonr if the input has changed.
                if (positives := numpy.count_nonzero(combt)) != num_true:
                    costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
                    num_true = positives

                if costReg <= 0:
                    break
                elif self.fast_costes.value == M_ACCURATE or i < i_step * 10:
                    i -= i_step
                elif costReg > 0.45:
                    # We're way off, step down 10x
                    i -= i_step * 10
                elif costReg > 0.35:
                    # Still far from 0, step 5x
                    i -= i_step * 5
                elif costReg > 0.25:
                    # Step 2x
                    i -= i_step * 2
                else:
                    i -= i_step
            except ValueError:
                break
        return thr_fi_c, thr_si_c

    def bisection_costes(self, fi, si, scale_max=255):
        """
        Finds the Costes Automatic Threshold for colocalization using a bisection algorithm.
        Candidate thresholds are selected from within a window of possible intensities,
        this window is narrowed based on the R value of each tested candidate.
        We're looking for the first point below 0, and R value can become highly variable
        at lower thresholds in some samples. Therefore the candidate tested in each
        loop is 1/6th of the window size below the maximum value (as opposed to the midpoint).
        """

        non_zero = (fi > 0) | (si > 0)
        xvar = numpy.var(fi[non_zero], axis=0, ddof=1)
        yvar = numpy.var(si[non_zero], axis=0, ddof=1)

        xmean = numpy.mean(fi[non_zero], axis=0)
        ymean = numpy.mean(si[non_zero], axis=0)

        z = fi[non_zero] + si[non_zero]
        zvar = numpy.var(z, axis=0, ddof=1)

        covar = 0.5 * (zvar - (xvar + yvar))

        denom = 2 * covar
        num = (yvar - xvar) + numpy.sqrt(
            (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
        )
        a = num / denom
        b = ymean - a * xmean

        # Initialise variables
        left = 1
        right = scale_max
        mid = ((right - left) // (6/5)) + left
        lastmid = 0
        # Marks the value with the last positive R value.
        valid = 1

        while lastmid != mid:
            thr_fi_c = mid / scale_max
            thr_si_c = (a * thr_fi_c) + b
            combt = (fi < thr_fi_c) | (si < thr_si_c)
            if numpy.count_nonzero(combt) <= 2:
                # Can't run pearson with only 2 values.
                left = mid - 1
            else:
                try:
                    costReg, _ = scipy.stats.pearsonr(fi[combt], si[combt])
                    if costReg < 0:
                        left = mid - 1
                    elif costReg >= 0:
                        right = mid + 1
                        valid = mid
                except ValueError:
                    # Catch misc Pearson errors with low sample numbers
                    left = mid - 1
            lastmid = mid
            if right - left > 6:
                mid = ((right - left) // (6 / 5)) + left
            else:
                mid = ((right - left) // 2) + left

        thr_fi_c = (valid - 1) / scale_max
        thr_si_c = (a * thr_fi_c) + b

        return thr_fi_c, thr_si_c

    def get_measurement_columns(self, pipeline):
        """Return column definitions for all measurements made by this module"""
        columns = []
        for first_image, second_image in self.get_image_pairs():
            if self.wants_images():
                if self.do_corr_and_slope:
                    columns += [
                        (
                            "Image",
                            F_CORRELATION_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                        (
                            "Image",
                            F_SLOPE_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                    ]
                if self.do_overlap:
                    columns += [
                        (
                            "Image",
                            F_OVERLAP_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                        (
                            "Image",
                            F_K_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                        (
                            "Image",
                            F_K_FORMAT % (second_image, first_image),
                            COLTYPE_FLOAT,
                        ),
                    ]
                if self.do_manders:
                    columns += [
                        (
                            "Image",
                            F_MANDERS_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                        (
                            "Image",
                            F_MANDERS_FORMAT % (second_image, first_image),
                            COLTYPE_FLOAT,
                        ),
                    ]

                if self.do_rwc:
                    columns += [
                        (
                            "Image",
                            F_RWC_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                        (
                            "Image",
                            F_RWC_FORMAT % (second_image, first_image),
                            COLTYPE_FLOAT,
                        ),
                    ]
                if self.do_costes:
                    columns += [
                        (
                            "Image",
                            F_COSTES_FORMAT % (first_image, second_image),
                            COLTYPE_FLOAT,
                        ),
                        (
                            "Image",
                            F_COSTES_FORMAT % (second_image, first_image),
                            COLTYPE_FLOAT,
                        ),
                    ]

            if self.wants_objects():
                for i in range(len(self.objects_list.value)):
                    object_name = self.objects_list.value[i]
                    if self.do_corr_and_slope:
                        columns += [
                            (
                                object_name,
                                F_CORRELATION_FORMAT % (first_image, second_image),
                                COLTYPE_FLOAT,
                            )
                        ]
                    if self.do_overlap:
                        columns += [
                            (
                                object_name,
                                F_OVERLAP_FORMAT % (first_image, second_image),
                                COLTYPE_FLOAT,
                            ),
                            (
                                object_name,
                                F_K_FORMAT % (first_image, second_image),
                                COLTYPE_FLOAT,
                            ),
                            (
                                object_name,
                                F_K_FORMAT % (second_image, first_image),
                                COLTYPE_FLOAT,
                            ),
                        ]
                    if self.do_manders:
                        columns += [
                            (
                                object_name,
                                F_MANDERS_FORMAT % (first_image, second_image),
                                COLTYPE_FLOAT,
                            ),
                            (
                                object_name,
                                F_MANDERS_FORMAT % (second_image, first_image),
                                COLTYPE_FLOAT,
                            ),
                        ]
                    if self.do_rwc:
                        columns += [
                            (
                                object_name,
                                F_RWC_FORMAT % (first_image, second_image),
                                COLTYPE_FLOAT,
                            ),
                            (
                                object_name,
                                F_RWC_FORMAT % (second_image, first_image),
                                COLTYPE_FLOAT,
                            ),
                        ]
                    if self.do_costes:
                        columns += [
                            (
                                object_name,
                                F_COSTES_FORMAT % (first_image, second_image),
                                COLTYPE_FLOAT,
                            ),
                            (
                                object_name,
                                F_COSTES_FORMAT % (second_image, first_image),
                                COLTYPE_FLOAT,
                            ),
                        ]
        return columns

    def get_categories(self, pipeline, object_name):
        """Return the categories supported by this module for the given object

        object_name - name of the measured object or IMAGE
        """
        if (object_name == "Image" and self.wants_images()) or (
            (object_name != "Image")
            and self.wants_objects()
            and (object_name in self.objects_list.value)
        ):
            return ["Correlation"]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if self.get_categories(pipeline, object_name) == [category]:
            results = []
            if self.do_corr_and_slope:
                if object_name == "Image":
                    results += ["Correlation", "Slope"]
                else:
                    results += ["Correlation"]
            if self.do_overlap:
                results += ["Overlap", "K"]
            if self.do_manders:
                results += ["Manders"]
            if self.do_rwc:
                results += ["RWC"]
            if self.do_costes:
                results += ["Costes"]
            return results
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        """Return the joined pairs of images measured"""
        result = []
        if measurement in self.get_measurements(pipeline, object_name, category):
            for i1, i2 in self.get_image_pairs():
                result.append("%s_%s" % (i1, i2))
                # For asymmetric, return both orderings
                if measurement in ("K", "Manders", "RWC", "Costes"):
                    result.append("%s_%s" % (i2, i1))
        return result

    def validate_module(self, pipeline):
        """Make sure chosen objects are selected only once"""
        if len(self.images_list.value) < 2:
            raise ValidationError("This module needs at least 2 images to be selected", self.images_list)

        if self.wants_objects():
            if len(self.objects_list.value) == 0:
                raise ValidationError("No object sets selected", self.objects_list)

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust the setting values for pipelines saved under old revisions"""
        if variable_revision_number < 2:
            raise NotImplementedError(
                "Automatic upgrade for this module is not supported in CellProfiler 3."
            )

        if variable_revision_number == 2:
            image_count = int(setting_values[0])
            idx_thr = image_count + 2
            setting_values = (
                setting_values[:idx_thr] + ["15.0"] + setting_values[idx_thr:]
            )
            variable_revision_number = 3

        if variable_revision_number == 3:
            num_images = int(setting_values[0])
            num_objects = int(setting_values[1])
            div_img = 2 + num_images
            div_obj = div_img + 2 + num_objects
            images_set = set(setting_values[2:div_img])
            thr_mode = setting_values[div_img : div_img + 2]
            objects_set = set(setting_values[div_img + 2 : div_obj])
            other_settings = setting_values[div_obj:]
            if "None" in images_set:
                images_set.remove("None")
            if "None" in objects_set:
                objects_set.remove("None")
            images_string = ", ".join(map(str, images_set))
            objects_string = ", ".join(map(str, objects_set))
            setting_values = (
                [images_string] + thr_mode + [objects_string] + other_settings
            )
            variable_revision_number = 4
        if variable_revision_number == 4:
            # Add costes mode switch
            setting_values += [M_FASTER]
            variable_revision_number = 5
        return setting_values, variable_revision_number

    def volumetric(self):
        return True


def get_scale(scale_1, scale_2):
    if scale_1 is not None and scale_2 is not None:
        return max(scale_1, scale_2)
    elif scale_1 is not None:
        return scale_1
    elif scale_2 is not None:
        return scale_2
    else:
        return 255
