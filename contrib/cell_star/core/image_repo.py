# -*- coding: utf-8 -*-
"""
Images repository calculates and stores intermediate images used in segmentation process.
Date: 2013-2016
Website: http://cellstar-algorithm.org/
"""

import math

from contrib.cell_star.utils.image_util import *


class ImageRepo(object):
    """
    After initialization with input image it calculates on demand intermediate images such as foreground,
    background mask, and others.
    """

    @property
    def background(self):
        if self._background is None:
            self.calculate_background()

        return self._background

    @property
    def brighter(self):
        if self._brighter is None:
            self.calculate_brighter()

        return self._brighter

    @property
    def brighter_original(self):
        if self._brighter is None:
            self.calculate_brighter_original()

        return self._brighter_original

    @property
    def darker(self):
        if self._darker is None:
            self.calculate_darker()

        return self._darker

    @property
    def darker_original(self):
        if self._darker_original is None:
            self.calculate_darker_original()

        return self._darker_original

    @property
    def image_back_difference(self):
        if self._clean_original is None:
            self.calculate_clean_original()

        return self._clean_original

    @property
    def image_back_difference_blurred(self):
        if self._clean is None:
            self.calculate_clean()

        return self._clean

    @property
    def foreground_mask(self):
        if self._foreground_mask is None:
            self.calculate_forebackground_masks()

        return self._foreground_mask

    @property
    def background_mask(self):
        if self._background_mask is None:
            self.calculate_forebackground_masks()

        return self._background_mask

    @property
    def cell_content_mask(self):
        if self._cell_content_mask is None:
            self.calculate_cell_border_content_mask()

        return self._cell_content_mask

    @property
    def cell_border_mask(self):
        if self._cell_border_mask is None:
            self.calculate_cell_border_content_mask()

        return self._cell_border_mask

    @property
    def segmentation(self):
        if self._segmentation is None:
            self.init_segmentation()

        return self._segmentation

    def __init__(self, image, parameters):
        """
        @param image: 0-1 float array
        @type image: np.ndarray
        @type parameters: dict
        """
        self.parameters = parameters

        # float arrays
        self.image = image
        self._background = None
        self._brighter_original = None
        self._darker_original = None
        self._clean_original = None
        self._brighter = None
        self._darker = None
        self._clean = None

        # binary masks
        self._foreground_mask = None
        self._background_mask = None
        self._cell_border_mask = None
        self._cell_content_mask = None

        # cache
        self._blurred = []  # (image,blur,image_blurred)

        # segmentation labels
        self._segmentation = None

    def init_segmentation(self):
        self._segmentation = np.zeros(self.image.shape[:2], int)

    def calculate_background(self, background_mask=None):
        """
        Fills in background pixels based on foreground pixels using blur.
        @param background_mask: pixels that belong to the background, if not given background is calculated from image
        """

        # No background mask is provided so calculate one based on edges in the image.
        if background_mask is None:
            smoothed = image_smooth(self.image, int(self.parameters["segmentation"]["background"]["computeByBlurring"]
                                                    * self.parameters["segmentation"]["avgCellDiameter"]))
            foreground_mask = image_normalize(abs(self.image - smoothed)) \
                              > self.parameters["segmentation"]["foreground"]["MaskThreshold"]

            foreground_mask = \
                fill_foreground_holes(foreground_mask,
                                      self.parameters["segmentation"]["foreground"]["MaskDilation"]
                                      * self.parameters["segmentation"]["avgCellDiameter"],
                                      self.parameters["segmentation"]["foreground"]["FillHolesWithAreaSmallerThan"]
                                      * self.parameters["segmentation"]["avgCellDiameter"] ** 2 * math.pi / 4,
                                      self.parameters["segmentation"]["foreground"]["MinCellClusterArea"]
                                      * self.parameters["segmentation"]["avgCellDiameter"] ** 2 * math.pi / 4,
                                      self.parameters["segmentation"]["foreground"]["MaskMinRadius"]
                                      * self.parameters["segmentation"]["avgCellDiameter"]
                                      )
            background_mask = np.logical_not(foreground_mask)

        filler_value = np.median(self.image)
        background = self.image.astype(float)
        foreground_mask = np.logical_not(background_mask)
        if background_mask.any():
            filler_value = np.median(background[background_mask])

        background = background * background_mask + filler_value * foreground_mask

        # Spread foreground to background pixels
        smooth_radius = round(self.parameters["segmentation"]["background"]["blur"]
                              * self.parameters["segmentation"]["avgCellDiameter"])
        steps = self.parameters["segmentation"]["background"]["blurSteps"]

        for i in xrange(steps):
            background = image_smooth(background, 1 + round(smooth_radius * ((steps - i) / steps) ** 2))
            background = background * foreground_mask + self.image * background_mask

        self._background = background
        self._foreground_mask = foreground_mask

    def calculate_clean_original(self):
        self._clean_original = image_normalize(self.image - self.background)

    def calculate_brighter_original(self):
        self._brighter_original = self.image - self.background
        self._brighter_original = image_normalize(np.maximum(self._brighter_original, 0))

    def calculate_darker_original(self):
        self._darker_original = self.background - self.image
        self._darker_original = image_normalize(np.maximum(self._darker_original, 0))

    def calculate_forebackground_masks(self):
        """
        Determines foreground which is the part of the image different substantialy from provided background.
        Holes in initial foreground are filled using morphological operations
        """

        # Calculate foreground based on blurred brighter/darker.
        darker_blurred = self.get_blurred(self.darker_original, self.parameters["segmentation"]["foreground"]["blur"])
        brighter_blurred = self.get_blurred(self.brighter_original,
                                            self.parameters["segmentation"]["foreground"]["blur"])
        self._foreground_mask = \
            (darker_blurred + brighter_blurred) > self.parameters["segmentation"]["foreground"]["MaskThreshold"]

        if self.parameters["segmentation"]["foreground"]["pickyDetection"]:
            smooth_coefficient = round(self.parameters["segmentation"]["background"]["computeByBlurring"]
                                       * self.parameters["segmentation"]["avgCellDiameter"])

            temp_blurred = image_smooth(self.image, smooth_coefficient)
            temp_fg_mask = image_normalize(np.abs((self.image - temp_blurred))) > \
                           self.parameters["segmentation"]["foreground"]["MaskThreshold"]

            self._foreground_mask = np.logical_and(self.foreground_mask, temp_fg_mask)

        self._foreground_mask = \
            fill_foreground_holes(self.foreground_mask,
                                  self.parameters["segmentation"]["foreground"]["MaskDilation"]
                                  * self.parameters["segmentation"]["avgCellDiameter"],
                                  self.parameters["segmentation"]["foreground"]["FillHolesWithAreaSmallerThan"]
                                  * self.parameters["segmentation"]["avgCellDiameter"] ** 2 * math.pi / 4,
                                  self.parameters["segmentation"]["foreground"]["MinCellClusterArea"]
                                  * self.parameters["segmentation"]["avgCellDiameter"] ** 2 * math.pi / 4,
                                  self.parameters["segmentation"]["foreground"]["MaskMinRadius"]
                                  * self.parameters["segmentation"]["avgCellDiameter"]
                                  )

        self._background_mask = np.logical_not(self.foreground_mask)

    def calculate_clean(self):
        image_diff_med_filter_size = round(self.parameters["segmentation"]["stars"]["gradientBlur"]
                                           * self.parameters["segmentation"]["avgCellDiameter"])
        self._clean = image_smooth(self.image_back_difference, image_diff_med_filter_size)

        clean_mean = self._clean_original.mean()
        masked = self._clean * self.foreground_mask
        mean_negative_masked = clean_mean * self.background_mask
        self._clean = masked + mean_negative_masked

    def calculate_brighter(self):
        brighter_med_filter_size = np.round(self.parameters["segmentation"]["cellBorder"]["medianFilter"]
                                            * self.parameters["segmentation"]["avgCellDiameter"])
        self._brighter = image_median_filter(self.brighter_original, brighter_med_filter_size)
        self._brighter *= self.foreground_mask

    def calculate_darker(self):
        darker_med_filter_size = round(self.parameters["segmentation"]["cellContent"]["medianFilter"]
                                       * self.parameters["segmentation"]["avgCellDiameter"])
        self._darker = image_median_filter(self.darker_original, darker_med_filter_size)
        self._darker *= self.foreground_mask

    def calculate_cell_border_content_mask(self):
        """
        Splits foreground pixels into cell content and cell borders.
        """
        blur_iterations = int(math.ceil(self.parameters["segmentation"]["cellContent"]["blur"]
                                        * self.parameters["segmentation"]["avgCellDiameter"] - 0.5))
        darker_blurred = self.get_blurred(self.darker, blur_iterations)

        if self.parameters["segmentation"]["cellContent"]["MaskThreshold"] == 0:
            eroded_foreground = image_erode(self.foreground_mask,
                                            self.parameters["segmentation"]["foreground"]["MaskDilation"]
                                            * self.parameters["segmentation"]["avgCellDiameter"])

            if not eroded_foreground.any():
                cell_content_mask_threshold = 0.0
            else:
                cell_content_mask_threshold = np.median(self._darker[eroded_foreground])
                if np.max(self._darker[eroded_foreground]) == cell_content_mask_threshold:  # there is only one value
                    cell_content_mask_threshold = 0.0  # take all
        else:
            cell_content_mask_threshold = self.parameters["segmentation"]["cellContent"]["MaskThreshold"]

        self._cell_content_mask = (self.brighter == 0) & self.foreground_mask & \
                                  (darker_blurred > cell_content_mask_threshold)

        self._cell_border_mask = self.foreground_mask & np.logical_not(self.cell_content_mask)

    def get_blurred(self, image, blur_param, cache_result=False):
        cache = [c for a, b, c in self._blurred if a is image and b == blur_param]
        if cache:
            return cache[0]
        else:
            blurred = image_blur(image, blur_param)
            if cache_result:
                self._blurred.append((image, blur_param, blurred))
            return blurred
