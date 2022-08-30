from re import S
import numpy
import skimage
import scipy
import centrosome
import cellprofiler.library as cp

class Threshold:
    def __init__(
        self,
        threshold_operation,
        two_class_otsu,
        assign_middle_to_foreground,
        log_transform,
        manual_threshold,
        thresholding_measurement,
        threshold_scope,
        threshold_correction_factor,
        threshold_range_min,
        threshold_range_max,
        adaptive_window_size,
        lower_outlier_fraction,
        upper_outlier_fraction,
        averaging_method,
        variance_method,
        number_of_deviations,
        volumetric
        ):
        self.threshold_operation = threshold_operation
        self.two_class_otsu = two_class_otsu
        self.assign_middle_to_foreground = assign_middle_to_foreground
        self.log_transform = log_transform
        self.manual_threshold = manual_threshold
        self.thresholding_measurement = thresholding_measurement
        self.threshold_scope = threshold_scope
        self.threshold_correction_factor = threshold_correction_factor
        self.threshold_range_min = threshold_range_min
        self.threshold_range_max = threshold_range_max
        self.adaptive_window_size = adaptive_window_size
        self.lower_outlier_fraction = lower_outlier_fraction
        self.upper_outlier_fraction = upper_outlier_fraction
        self.averaging_method = averaging_method
        self.variance_method = variance_method
        self.number_of_deviations = number_of_deviations
        self.volumetric = volumetric
    
    def _correct_global_threshold(self, threshold):
        threshold *= self.threshold_correction_factor
        ### Check if min max is always correctly found? Before, this used .min or .max methods
        return min(max(threshold, self.threshold_range_min), self.threshold_range_max)

    def get_global_threshold(self, image, mask, automatic=False):
        image = image[mask]

        # Shortcuts - Check if image array is empty or all pixels are the same value.
        if len(image) == 0:
            threshold = 0.0

        elif numpy.all(image == image[0]):
            threshold = image[0]

        elif automatic or self.threshold_operation.casefold() in ("minimum cross-entropy", "sauvola"):
            tol = max(numpy.min(numpy.diff(numpy.unique(image))) / 2, 0.5 / 65536)
            threshold = skimage.filters.threshold_li(image, tolerance=tol)

        elif self.threshold_operation.casefold() == "robust background":
            threshold = cp.functions.image_processing.get_threshold_robust_background(
                image,
                self.lower_outlier_fraction,
                self.upper_outlier_fraction,
                self.averaging_method,
                self.variance_method,
                self.number_of_deviations,
                )

        elif self.threshold_operation.casefold() == "otsu":
            if self.two_class_otsu.casefold() == "two classes":
                threshold = skimage.filters.threshold_otsu(image)
            elif self.two_class_otsu.casefold() == "three classes":
                bin_wanted = (
                    0 if self.assign_middle_to_foreground.casefold() == "Foreground" else 1
                )
                threshold = skimage.filters.threshold_multiotsu(image, nbins=128)
                threshold = threshold[bin_wanted]
        else:
            raise ValueError("Invalid thresholding settings")
        return threshold

    def _run_local_threshold(self, image, method, volumetric, **kwargs):
        if volumetric:
            t_local = numpy.zeros_like(image)
            for index, plane in enumerate(image):
                t_local[index] = self._get_adaptive_threshold(plane, method, **kwargs)
        else:
            t_local = self._get_adaptive_threshold(image, method, **kwargs)
        return skimage.img_as_float(t_local)

    def _get_adaptive_threshold(self, image, threshold_method, **kwargs):
        """Given a global threshold, compute a threshold per pixel

        Break the image into blocks, computing the threshold per block.
        Afterwards, constrain the block threshold to .7 T < t < 1.5 T.
        """
        # for the X and Y direction, find the # of blocks, given the
        # size constraints
        if self.threshold_operation.casefold() == "otsu":
            bin_wanted = (
                0 if self.assign_middle_to_foreground == "Foreground" else 1
            )
        image_size = numpy.array(image.shape[:2], dtype=int)
        nblocks = image_size // self.adaptive_window_size
        if any(n < 2 for n in nblocks):
            raise ValueError(
                "Adaptive window cannot exceed 50%% of an image dimension.\n"
                "Window of %dpx is too large for a %sx%s image"
                % (self.adaptive_window_size, image_size[1], image_size[0])
            )
        #
        # Use a floating point block size to apportion the roundoff
        # roughly equally to each block
        #
        increment = numpy.array(image_size, dtype=float) / numpy.array(
            nblocks, dtype=float
        )
        #
        # Put the answer here
        #
        thresh_out = numpy.zeros(image_size, image.dtype)
        #
        # Loop once per block, computing the "global" threshold within the
        # block.
        #
        block_threshold = numpy.zeros([nblocks[0], nblocks[1]])
        for i in range(nblocks[0]):
            i0 = int(i * increment[0])
            i1 = int((i + 1) * increment[0])
            for j in range(nblocks[1]):
                j0 = int(j * increment[1])
                j1 = int((j + 1) * increment[1])
                block = image[i0:i1, j0:j1]
                block = block[~numpy.isnan(block)]
                if len(block) == 0:
                    threshold_out = 0.0
                elif numpy.all(block == block[0]):
                    # Don't compute blocks with only 1 value.
                    threshold_out = block[0]
                elif (self.threshold_operation.casefold() == "otsu" and
                      self.two_class_otsu.casefold() == "three class" and
                      len(numpy.unique(block)) < 3):
                    # Can't run 3-class otsu on only 2 values.
                    threshold_out = skimage.filters.threshold_otsu(block)
                else:
                    try: 
                        threshold_out = threshold_method(block, **kwargs)
                    except ValueError:
                        # Drop nbins kwarg when multi-otsu fails. See issue #6324 scikit-image
                        threshold_out = threshold_method(block)
                if isinstance(threshold_out, numpy.ndarray):
                    # Select correct bin if running multiotsu
                    threshold_out = threshold_out[bin_wanted]
                block_threshold[i, j] = threshold_out

        #
        # Use a cubic spline to blend the thresholds across the image to avoid image artifacts
        #
        spline_order = min(3, numpy.min(nblocks) - 1)
        xStart = int(increment[0] / 2)
        xEnd = int((nblocks[0] - 0.5) * increment[0])
        yStart = int(increment[1] / 2)
        yEnd = int((nblocks[1] - 0.5) * increment[1])
        xtStart = 0.5
        xtEnd = image.shape[0] - 0.5
        ytStart = 0.5
        ytEnd = image.shape[1] - 0.5
        block_x_coords = numpy.linspace(xStart, xEnd, nblocks[0])
        block_y_coords = numpy.linspace(yStart, yEnd, nblocks[1])
        adaptive_interpolation = scipy.interpolate.RectBivariateSpline(
            block_x_coords,
            block_y_coords,
            block_threshold,
            bbox=(xtStart, xtEnd, ytStart, ytEnd),
            kx=spline_order,
            ky=spline_order,
        )
        thresh_out_x_coords = numpy.linspace(
            0.5, int(nblocks[0] * increment[0]) - 0.5, thresh_out.shape[0]
        )
        thresh_out_y_coords = numpy.linspace(
            0.5, int(nblocks[1] * increment[1]) - 0.5, thresh_out.shape[1]
        )

        thresh_out = adaptive_interpolation(thresh_out_x_coords, thresh_out_y_coords)

        return thresh_out


    def get_local_threshold(self, image_data, mask):
        image = numpy.where(mask, image_data, np.nan)

        if len(image) == 0 or numpy.all(image == numpy.nan):
            local_threshold = numpy.zeros_like(image)

        elif numpy.all(image == image[0]):
            local_threshold = numpy.full_like(image, image[0])

        elif self.threshold_operation.casefold() == "minimum cross-entropy":
            local_threshold = self._run_local_threshold(
                image,
                method=skimage.filters.threshold_li,
                volumetric=self.volumetric,
                tolerance=max(numpy.min(numpy.diff(numpy.unique(image))) / 2, 0.5 / 65536)
            )
        elif self.threshold_operation.casefold() == "otsu":
            if self.two_class_otsu.casefold() == "two classes":
                local_threshold = self._run_local_threshold(
                    image,
                    method=skimage.filters.threshold_otsu,
                    volumetric=self.volumetric,
                )

            elif self.two_class_otsu.casefold() == "three classes":
                local_threshold = self._run_local_threshold(
                    image,
                    method=skimage.filters.threshold_multiotsu,
                    volumetric=self.volumetric,
                    nbins=128,
                )

        elif self.threshold_operation.casefold() == "robust background":
            local_threshold = self._run_local_threshold(
                image,
                method=cp.functions.image_processing.get_threshold_robust_background,
                lower_outlier_fraction=self.lower_outlier_fraction,
                upper_outlier_fraction=self.upper_outlier_fraction,
                averaging_method=self.averaging_method,
                variance_method=self.variance_method,
                number_of_deviations=self.number_of_deviations,
                volumetric=self.volumetric,
            )

        elif self.threshold_operation.casefold() == "sauvola":
            image_data = numpy.where(mask, image, 0)
            if adaptive_window % 2 == 0:
                adaptive_window += 1
            local_threshold = skimage.filters.threshold_sauvola(
                image, window_size=self.adaptive_window_size
            )

        else:
            raise ValueError("Invalid thresholding settings")
        return local_threshold

    def _correct_local_threshold(self, t_local_orig, t_guide):
        t_local = t_local_orig.copy()
        t_local *= self.threshold_correction_factor

        # Constrain the local threshold to be within [0.7, 1.5] * global_threshold. It's for the pretty common case
        # where you have regions of the image with no cells whatsoever that are as large as whatever window you're
        # using. Without a lower bound, you start having crazy threshold s that detect noise blobs. And same for
        # very crowded areas where there is zero background in the window. You want the foreground to be all
        # detected.
        ##### .min and .max methods removed. Test if this is a good idea
        t_min = max(self.threshold_range_min, t_guide * 0.7)
        t_max = min(self.threshold_range_max, t_guide * 1.5)

        t_local[t_local < t_min] = t_min
        t_local[t_local > t_max] = t_max

        return t_local

    def apply_threshold(self, image, mask, threshold, automatic=False):
        data = image.pixel_data

        if not automatic and self.threshold_smoothing_scale.value == 0:
            return (data >= threshold) & mask, 0

        if automatic:
            sigma = 1
        else:
            # Convert from a scale into a sigma. What I've done here
            # is to structure the Gaussian so that 1/2 of the smoothed
            # intensity is contributed from within the smoothing diameter
            # and 1/2 is contributed from outside.
            sigma = self.threshold_smoothing_scale.value / 0.6744 / 2.0

        blurred_image = centrosome.smooth.smooth_with_function_and_mask(
            data,
            lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0),
            mask,
        )

        return (blurred_image >= threshold) & mask, sigma



    