# coding=utf-8

"""
Align
=====

**Align** aligns images relative to each other, for example, to correct
shifts in the optical path of a microscope in each channel of a
multi-channel set of images.

For two or more input images, this module determines the optimal
alignment among them. Aligning images is useful to obtain proper
measurements of the intensities in one channel based on objects
identified in another channel, for example. Alignment is often needed
when the microscope is not perfectly calibrated. It can also be useful
to align images in a time-lapse series of images. The module stores the
amount of shift between images as a measurement, which can be useful for
quality control purposes.

Note that the second image (and others following) is always aligned with
respect to the first image. That is, the X/Y offsets indicate how much
the second image needs to be shifted by to match the first.

This module does not perform warping or rotation, it simply shifts images
in X and Y. For more complex registration tasks, you might preprocess
images using a plugin for that purpose in FIJI/ImageJ.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============



Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Xshift, Yshift:* The pixel shift in X and Y of the aligned image
   with respect to the original image.

References
^^^^^^^^^^

-  Lewis JP. (1995) “Fast normalized cross-correlation.” *Vision
   Interface*, 1-7.
"""

import numpy as np
import scipy.ndimage as scind
import scipy.sparse
from centrosome.filter import stretch
from scipy.fftpack import fft2, ifft2

from ..constants.measurement import COLTYPE_INTEGER
from ..image import Image
from ..module import Module
from ..setting import Divider, SettingsGroup
from ..setting.choice import Choice
from ..setting.do_something import DoSomething, RemoveSettingButton
from ..setting.subscriber import ImageSubscriber
from ..setting.text import ImageName

M_MUTUAL_INFORMATION = "Mutual Information"
M_CROSS_CORRELATION = "Normalized Cross Correlation"
M_ALL = (M_MUTUAL_INFORMATION, M_CROSS_CORRELATION)

A_SIMILARLY = "Similarly"
A_SEPARATELY = "Separately"

C_SAME_SIZE = "Keep size"
C_CROP = "Crop to aligned region"
C_PAD = "Pad images"

C_ALIGN = "Align"

MEASUREMENT_FORMAT = C_ALIGN + "_%sshift_%s"


class Align(Module):
    module_name = "Align"
    category = "Image Processing"
    variable_revision_number = 3

    def create_settings(self):
        self.first_input_image = ImageSubscriber(
            "Select the first input image",
            "None",
            doc="""\
Specify the name of the first image to align.""",
        )

        self.first_output_image = ImageName(
            "Name the first output image",
            "AlignedRed",
            doc="""\
Enter the name of the first aligned image.""",
        )

        self.separator_1 = Divider(line=False)
        self.second_input_image = ImageSubscriber(
            "Select the second input image",
            "None",
            doc="""\
Specify the name of the second image to align.""",
        )

        self.second_output_image = ImageName(
            "Name the second output image",
            "AlignedGreen",
            doc="""\
Enter the name of the second aligned image.""",
        )

        self.separator_2 = Divider(line=False)
        self.additional_images = []
        self.add_button = DoSomething("", "Add another image", self.add_image)

        self.alignment_method = Choice(
            "Select the alignment method",
            M_ALL,
            doc="""\
Two options for the alignment method are available:

-  *%(M_MUTUAL_INFORMATION)s:* This more general method works well for
   aligning images from different modalities that contain the same
   information, but are expressed differently. However, this method
   performs better than %(M_CROSS_CORRELATION)s, even in the same
   modality, if the images are not highly correlated. It is iterative,
   and thus tends to be slower than other methods, but is more likely to
   be correct. Essentially, alignment is performed by measuring how well
   one image “explains” the other. For example, a fluorescent image can
   be aligned to a brightfield image by this method since the relevant
   features are bright in one modality where they are dim in the other.
-  *%(M_CROSS_CORRELATION)s:* This is a good means of alignment in the
   case of images acquired with the same modality (e.g., all images to
   be aligned are fluorescent). It is fast, however it can be highly
   influenced by a particular, possibly spurious, feature and in turn
   generate anomalously large shifts. It allows for a linear
   relationship between the intensities of the two images, i.e., the
   relevant features in the images to be aligned all have varying
   degrees of brightness.

"""
                % globals(),
        )

        self.crop_mode = Choice(
            "Crop mode",
            [C_CROP, C_PAD, C_SAME_SIZE],
            doc="""\
The crop mode determines how the output images are either cropped or
padded after alignment. The alignment phase calculates the areas in each
image that are found to be overlapping. In almost all cases, there will
be portions of some or all of the images that don’t overlap with any
other aligned image. These portions have no counterpart and will be
excluded from analysis. There are three choices for cropping:

-  *%(C_CROP)s:* Crop every image to the region that overlaps in all
   images. This makes downstream analysis simpler because all of the
   output images have authentic pixel data at all positions, however it
   discards parts of images. Also, the output images may not be the same
   size as the input images which may cause problems if downstream
   modules use aligned and unaligned images (which may be of differing
   sizes) in combination.
-  *%(C_PAD)s:* Align every image and pad with masked black pixels to
   make each image the same size. This results in larger images, but
   preserves all information in each of the images. This may be the best
   choice if images undergo an operation such as smoothing that could
   use the information that would otherwise be cropped.
-  *%(C_SAME_SIZE)s:* Maintain the sizes of the images but align them,
   masking the unaligned portions with black pixels. **Align** aligns
   all images relative to the first. This is a reasonable option for
   alignments with small displacements since it maintains a consistent
   image size which may be useful if output images from different image
   sets will be compared against each other after processing. The
   reference image can also be used across image sets. For example, the
   reference image could be loaded for all image sets in a group to
   align the entire group’s images similarly, then the aligned images
   could be combined in a module such as **MakeProjection**.
   """
                % globals(),
        )



    def add_image(self, can_remove=True):
        """Add an image + associated questions and buttons"""
        group = SettingsGroup()
        if can_remove:
            group.append("divider", Divider(line=False))

        group.append(
            "input_image_name",
            ImageSubscriber(
                "Select the additional image",
                "None",
                doc="""
 Select the additional image to align?""",
            ),
        )

        group.append(
            "output_image_name",
            ImageName(
                "Name the output image",
                "AlignedBlue",
                doc="""
 Enter the name of the aligned image?""",
            ),
        )

        group.append(
            "align_choice",
            Choice(
                "Select how the alignment is to be applied",
                [A_SIMILARLY, A_SEPARATELY],
                doc="""\
An additional image can either be aligned similarly to the second one or
a separate alignment to the first image can be calculated:

-  *%(A_SIMILARLY)s:* The same alignment measurements obtained from the
   first two input images are applied to this additional image.
-  *%(A_SEPARATELY)s:* A new set of alignment measurements are
   calculated for this additional image using the alignment method
   specified with respect to the first input image.
"""
                    % globals(),
            ),
        )

        if can_remove:
            group.append(
                "remover",
                RemoveSettingButton(
                    "", "Remove above image", self.additional_images, group
                ),
            )
        self.additional_images.append(group)

    def settings(self):

        result = [self.alignment_method, self.crop_mode]

        result += [
            self.first_input_image,
            self.first_output_image,
            self.second_input_image,
            self.second_output_image,
        ]
        for additional in self.additional_images:
            result += [
                additional.input_image_name,
                additional.output_image_name,
                additional.align_choice,
            ]
        return result

    def prepare_settings(self, setting_values):
        assert (len(setting_values) - 6) % 3 == 0
        n_additional = (len(setting_values) - 6) / 3
        del self.additional_images[:]
        while len(self.additional_images) < n_additional:
            self.add_image()

    def visible_settings(self):
        result = [self.alignment_method, self.crop_mode]

        result += [
            self.first_input_image,
            self.first_output_image,
            self.separator_1,
            self.second_input_image,
            self.second_output_image,
        ]
        for additional in self.additional_images:
            result += additional.visible_settings()
        result += [self.add_button]
        return result

    def run(self, workspace):
        i_min = np.iinfo(int).max
        j_min = np.iinfo(int).max
        off_x, off_y = self.align(
            workspace, self.first_input_image.value, self.second_input_image.value
        )
        names = [
            (self.first_input_image.value, self.first_output_image.value),
            (self.second_input_image.value, self.second_output_image.value),
        ]
        offsets = [(0, 0), (off_y, off_x)]

        for additional in self.additional_images:
            names.append(
                (additional.input_image_name.value, additional.output_image_name.value)
            )
            if additional.align_choice == A_SIMILARLY:
                a_off_x, a_off_y = off_x, off_y
            else:
                a_off_x, a_off_y = self.align(
                    workspace,
                    self.first_input_image.value,
                    additional.input_image_name.value,
                )
            offsets.append((a_off_y, a_off_x))

        shapes = [
            workspace.image_set.get_image(x).pixel_data.shape[:2] for x, _ in names
        ]
        offsets, shapes = self.adjust_offsets(offsets, shapes)

        #
        # Align and write the measurements
        #
        for (input_name, output_name), (y, x), shape in zip(names, offsets, shapes):
            self.apply_alignment(workspace, input_name, output_name, x, y, shape)
            for axis, value in (("X", -x), ("Y", -y)):
                feature = MEASUREMENT_FORMAT % (axis, output_name)
                workspace.measurements.add_image_measurement(feature, value)

        # save data for display
        workspace.display_data.image_info = [
            (
                input_name,
                workspace.image_set.get_image(input_name).pixel_data,
                output_name,
                workspace.image_set.get_image(output_name).pixel_data,
                x,
                y,
                shape,
            )
            for (input_name, output_name), (y, x), shape in zip(names, offsets, shapes)
        ]

    def display(self, workspace, figure):
        """Display the overlaid images

        workspace - the workspace being run, with display_data holding:
            image_info - a list of lists:
                 input image name of image being aligned
                 input image data
                 output image name of image being aligned
                 output image data
                 x offset
                 y offset
        """
        image_info = workspace.display_data.image_info
        first_input_name = self.first_input_image.value
        first_output_name = self.first_output_image.value
        figure.set_subplots((2, len(image_info) - 1))

        first_input_pixels = image_info[0][1]
        first_output_pixels = image_info[0][3]
        for (
                j,
                (input_name, input_pixels, output_name, output_pixels, off_x, off_y, shape),
        ) in enumerate(image_info[1:]):
            unaligned_title = "Unaligned images: %s and %s" % (
                first_input_name,
                input_name,
            )
            #
            # Make them grayscale if needed
            #
            first_pixels, other_pixels = [
                img if img.ndim == 2 else np.mean(img, 2)
                for img in (first_input_pixels, input_pixels)
            ]
            max_shape = np.maximum(first_pixels.shape, other_pixels.shape)
            img = np.zeros((max_shape[0], max_shape[1], 3))
            img[: first_pixels.shape[0], : first_pixels.shape[1], 0] = first_pixels
            img[: other_pixels.shape[0], : other_pixels.shape[1], 1] = other_pixels
            figure.subplot_imshow(
                0, j, img, unaligned_title, sharexy=figure.subplot(0, 0)
            )

            aligned_title = "Aligned images: %s and %s\nX offset: %d, Y offset: %d" % (
                first_output_name,
                output_name,
                -off_x,
                -off_y,
            )
            first_pixels, other_pixels = [
                img if img.ndim == 2 else np.mean(img, 2)
                for img in (first_output_pixels, output_pixels)
            ]
            max_shape = np.maximum(first_pixels.shape, other_pixels.shape)
            img = np.zeros((max_shape[0], max_shape[1], 3))
            img[: first_pixels.shape[0], : first_pixels.shape[1], 0] = first_pixels
            img[: other_pixels.shape[0], : other_pixels.shape[1], 1] = other_pixels
            figure.subplot_imshow(
                1, j, img, aligned_title, sharexy=figure.subplot(0, 0)
            )

    def align(self, workspace, input1_name, input2_name):
        """Align the second image with the first

        Calculate the alignment offset that must be added to indexes in the
        first image to arrive at indexes in the second image.

        Returns the x,y (not i,j) offsets.
        """
        image1 = workspace.image_set.get_image(input1_name)
        image1_pixels = image1.pixel_data.astype(float)
        image2 = workspace.image_set.get_image(input2_name)
        image2_pixels = image2.pixel_data.astype(float)
        if self.alignment_method == M_CROSS_CORRELATION:
            return self.align_cross_correlation(image1_pixels, image2_pixels)
        else:
            image1_mask = image1.mask
            image2_mask = image2.mask
            return self.align_mutual_information(
                image1_pixels, image2_pixels, image1_mask, image2_mask
            )

    @staticmethod
    def align_cross_correlation(pixels1, pixels2):
        """Align the second image with the first using max cross-correlation

        returns the x,y offsets to add to image1's indexes to align it with
        image2

        Many of the ideas here are based on the paper, "Fast Normalized
        Cross-Correlation" by J.P. Lewis
        (http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html)
        which is frequently cited when addressing this problem.
        """
        #
        # TODO: Possibly use all 3 dimensions for color some day
        #
        if pixels1.ndim == 3:
            pixels1 = np.mean(pixels1, 2)
        if pixels2.ndim == 3:
            pixels2 = np.mean(pixels2, 2)
        #
        # We double the size of the image to get a field of zeros
        # for the parts of one image that don't overlap the displaced
        # second image.
        #
        # Since we're going into the frequency domain, if the images are of
        # different sizes, we can make the FFT shape large enough to capture
        # the period of the largest image - the smaller just will have zero
        # amplitude at that frequency.
        #
        s = np.maximum(pixels1.shape, pixels2.shape)
        fshape = s * 2
        #
        # Calculate the # of pixels at a particular point
        #
        i, j = np.mgrid[-s[0]: s[0], -s[1]: s[1]]
        unit = np.abs(i * j).astype(float)
        unit[unit < 1] = 1  # keeps from dividing by zero in some places
        #
        # Normalize the pixel values around zero which does not affect the
        # correlation, keeps some of the sums of multiplications from
        # losing precision and precomputes t(x-u,y-v) - t_mean
        #
        pixels1 = pixels1 - np.mean(pixels1)
        pixels2 = pixels2 - np.mean(pixels2)
        #
        # Lewis uses an image, f and a template t. He derives a normalized
        # cross correlation, ncc(u,v) =
        # sum((f(x,y)-f_mean(u,v))*(t(x-u,y-v)-t_mean),x,y) /
        # sqrt(sum((f(x,y)-f_mean(u,v))**2,x,y) * (sum((t(x-u,y-v)-t_mean)**2,x,y)
        #
        # From here, he finds that the numerator term, f_mean(u,v)*(t...) is zero
        # leaving f(x,y)*(t(x-u,y-v)-t_mean) which is a convolution of f
        # by t-t_mean.
        #
        fp1 = fft2(pixels1, fshape.tolist())
        fp2 = fft2(pixels2, fshape.tolist())
        corr12 = ifft2(fp1 * fp2.conj()).real

        #
        # Use the trick of Lewis here - compute the cumulative sums
        # in a fashion that accounts for the parts that are off the
        # edge of the template.
        #
        # We do this in quadrants:
        # q0 q1
        # q2 q3
        # For the first,
        # q0 is the sum over pixels1[i:,j:] - sum i,j backwards
        # q1 is the sum over pixels1[i:,:j] - sum i backwards, j forwards
        # q2 is the sum over pixels1[:i,j:] - sum i forwards, j backwards
        # q3 is the sum over pixels1[:i,:j] - sum i,j forwards
        #
        # The second is done as above but reflected lr and ud
        #
        p1_si = pixels1.shape[0]
        p1_sj = pixels1.shape[1]
        p1_sum = np.zeros(fshape)
        p1_sum[:p1_si, :p1_sj] = cumsum_quadrant(pixels1, False, False)
        p1_sum[:p1_si, -p1_sj:] = cumsum_quadrant(pixels1, False, True)
        p1_sum[-p1_si:, :p1_sj] = cumsum_quadrant(pixels1, True, False)
        p1_sum[-p1_si:, -p1_sj:] = cumsum_quadrant(pixels1, True, True)
        #
        # Divide the sum over the # of elements summed-over
        #
        p1_mean = p1_sum / unit

        p2_si = pixels2.shape[0]
        p2_sj = pixels2.shape[1]
        p2_sum = np.zeros(fshape)
        p2_sum[:p2_si, :p2_sj] = cumsum_quadrant(pixels2, False, False)
        p2_sum[:p2_si, -p2_sj:] = cumsum_quadrant(pixels2, False, True)
        p2_sum[-p2_si:, :p2_sj] = cumsum_quadrant(pixels2, True, False)
        p2_sum[-p2_si:, -p2_sj:] = cumsum_quadrant(pixels2, True, True)
        p2_sum = np.fliplr(np.flipud(p2_sum))
        p2_mean = p2_sum / unit
        #
        # Once we have the means for u,v, we can calculate the
        # variance-like parts of the equation. We have to multiply
        # the mean^2 by the # of elements being summed-over
        # to account for the mean being summed that many times.
        #
        p1sd = np.sum(pixels1 ** 2) - p1_mean ** 2 * np.product(s)
        p2sd = np.sum(pixels2 ** 2) - p2_mean ** 2 * np.product(s)
        #
        # There's always chance of roundoff error for a zero value
        # resulting in a negative sd, so limit the sds here
        #
        sd = np.sqrt(np.maximum(p1sd * p2sd, 0))
        corrnorm = corr12 / sd
        #
        # There's not much information for points where the standard
        # deviation is less than 1/100 of the maximum. We exclude these
        # from consideration.
        #
        corrnorm[(unit < np.product(s) / 2) & (sd < np.mean(sd) / 100)] = 0
        i, j = np.unravel_index(np.argmax(corrnorm), fshape)
        #
        # Reflect values that fall into the second half
        #
        if i > pixels1.shape[0]:
            i = i - fshape[0]
        if j > pixels1.shape[1]:
            j = j - fshape[1]
        return j, i

    @staticmethod
    def align_mutual_information(pixels1, pixels2, mask1, mask2):
        """Align the second image with the first using mutual information

        returns the x,y offsets to add to image1's indexes to align it with
        image2

        The algorithm computes the mutual information content of the two
        images, offset by one in each direction (including diagonal) and
        then picks the direction in which there is the most mutual information.
        From there, it tries all offsets again and so on until it reaches
        a local maximum.
        """
        #
        # TODO: Possibly use all 3 dimensions for color some day
        #
        if pixels1.ndim == 3:
            pixels1 = np.mean(pixels1, 2)
        if pixels2.ndim == 3:
            pixels2 = np.mean(pixels2, 2)

        def mutualinf(x, y, maskx, masky):
            x = x[maskx & masky]
            y = y[maskx & masky]
            return entropy(x) + entropy(y) - entropy2(x, y)

        maxshape = np.maximum(pixels1.shape, pixels2.shape)
        pixels1 = reshape_image(pixels1, maxshape)
        pixels2 = reshape_image(pixels2, maxshape)
        mask1 = reshape_image(mask1, maxshape)
        mask2 = reshape_image(mask2, maxshape)

        best = mutualinf(pixels1, pixels2, mask1, mask2)
        i = 0
        j = 0
        while True:
            last_i = i
            last_j = j
            for new_i in range(last_i - 1, last_i + 2):
                for new_j in range(last_j - 1, last_j + 2):
                    if new_i == 0 and new_j == 0:
                        continue
                    p2, p1 = offset_slice(pixels2, pixels1, new_i, new_j)
                    m2, m1 = offset_slice(mask2, mask1, new_i, new_j)
                    info = mutualinf(p1, p2, m1, m2)
                    if info > best:
                        best = info
                        i = new_i
                        j = new_j
            if i == last_i and j == last_j:
                return j, i

    @staticmethod
    def apply_alignment(
            workspace, input_image_name, output_image_name, off_x, off_y, shape
    ):
        """Apply an alignment to the input image to result in the output image

        workspace - image set's workspace passed to run

        input_image_name - name of the image to be aligned

        output_image_name - name of the resultant image

        off_x, off_y - offset of the resultant image relative to the original

        shape - shape of the resultant image
        """

        image = workspace.image_set.get_image(input_image_name)
        pixel_data = image.pixel_data
        if pixel_data.ndim == 2:
            output_shape = (shape[0], shape[1], 1)
            planes = [pixel_data]
        else:
            output_shape = (shape[0], shape[1], pixel_data.shape[2])
            planes = [pixel_data[:, :, i] for i in range(pixel_data.shape[2])]
        output_pixels = np.zeros(output_shape, pixel_data.dtype)
        for i, plane in enumerate(planes):
            #
            # Copy the input to the output
            #
            p1, p2 = offset_slice(plane, output_pixels[:, :, i], off_y, off_x)
            p2[:, :] = p1[:, :]
        if pixel_data.ndim == 2:
            output_pixels.shape = output_pixels.shape[:2]
        output_mask = np.zeros(shape, bool)
        p1, p2 = offset_slice(image.mask, output_mask, off_y, off_x)
        p2[:, :] = p1[:, :]
        if np.all(output_mask):
            output_mask = None
        crop_mask = np.zeros(image.pixel_data.shape, bool)
        p1, p2 = offset_slice(crop_mask, output_pixels, off_y, off_x)
        p1[:, :] = True
        if np.all(crop_mask):
            crop_mask = None
        output_image = Image(
            output_pixels, mask=output_mask, crop_mask=crop_mask, parent_image=image
        )
        workspace.image_set.add(output_image_name, output_image)

    def adjust_offsets(self, offsets, shapes):
        """Adjust the offsets and shapes for output

        workspace - workspace passed to "run"

        offsets - i,j offsets for each image

        shapes - shapes of the input images

        names - pairs of input / output names

        Based on the crop mode, adjust the offsets and shapes to optimize
        the cropping.
        """
        offsets = np.array(offsets)
        shapes = np.array(shapes)
        if self.crop_mode == C_CROP:
            # modify the offsets so that all are negative
            max_offset = np.max(offsets, 0)
            offsets = offsets - max_offset[np.newaxis, :]
            #
            # Reduce each shape by the amount chopped off
            #
            shapes += offsets
            #
            # Pick the smallest in each of the dimensions and repeat for all
            #
            shape = np.min(shapes, 0)
            shapes = np.tile(shape, len(shapes))
            shapes.shape = offsets.shape
        elif self.crop_mode == C_PAD:
            #
            # modify the offsets so that they are all positive
            #
            min_offset = np.min(offsets, 0)
            offsets = offsets - min_offset[np.newaxis, :]
            #
            # Expand each shape by the top-left padding
            #
            shapes += offsets
            #
            # Pick the largest in each of the dimensions and repeat for all
            #
            shape = np.max(shapes, 0)
            shapes = np.tile(shape, len(shapes))
            shapes.shape = offsets.shape
        return offsets.tolist(), shapes.tolist()

    def get_categories(self, pipeline, object_name):
        if object_name == "Image":
            return [C_ALIGN]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == "Image" and category == C_ALIGN:
            return ["Xshift", "Yshift"]
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        if measurement in self.get_measurements(pipeline, object_name, category):
            return [self.first_output_image.value, self.second_output_image.value] + [
                additional.output_image_name.value
                for additional in self.additional_images
            ]
        return []

    def get_measurement_columns(self, pipeline):
        """return the offset measurements"""

        targets = [self.first_output_image.value, self.second_output_image.value] + [
            additional.output_image_name.value for additional in self.additional_images
        ]
        columns = []
        for axis in ("X", "Y"):
            columns += [
                ("Image", MEASUREMENT_FORMAT % (axis, target), COLTYPE_INTEGER,)
                for target in targets
            ]
        return columns

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            # Moved final settings (alignment method, cropping) to the top
            setting_values = setting_values[-2:] + setting_values[:-2]
            variable_revision_number = 2

        if variable_revision_number == 2:
            # wants_cropping changed to crop_mode
            setting_values = (
                    setting_values[:1]
                    + [C_CROP if setting_values[1] == "Yes" else C_SAME_SIZE]
                    + setting_values[2:]
            )
            variable_revision_number = 3

        return setting_values, variable_revision_number


def offset_slice(pixels1, pixels2, i, j):
    """Return two sliced arrays where the first slice is offset by i,j
    relative to the second slice.

    """
    if i < 0:
        height = min(pixels1.shape[0] + i, pixels2.shape[0])
        p1_imin = -i
        p2_imin = 0
    else:
        height = min(pixels1.shape[0], pixels2.shape[0] - i)
        p1_imin = 0
        p2_imin = i
    p1_imax = p1_imin + height
    p2_imax = p2_imin + height
    if j < 0:
        width = min(pixels1.shape[1] + j, pixels2.shape[1])
        p1_jmin = -j
        p2_jmin = 0
    else:
        width = min(pixels1.shape[1], pixels2.shape[1] - j)
        p1_jmin = 0
        p2_jmin = j
    p1_jmax = p1_jmin + width
    p2_jmax = p2_jmin + width

    p1 = pixels1[p1_imin:p1_imax, p1_jmin:p1_jmax]
    p2 = pixels2[p2_imin:p2_imax, p2_jmin:p2_jmax]
    return p1, p2


def cumsum_quadrant(x, i_forwards, j_forwards):
    """Return the cumulative sum going in the i, then j direction

    x - the matrix to be summed
    i_forwards - sum from 0 to end in the i direction if true
    j_forwards - sum from 0 to end in the j direction if true
    """
    if i_forwards:
        x = x.cumsum(0)
    else:
        x = np.flipud(np.flipud(x).cumsum(0))
    if j_forwards:
        return x.cumsum(1)
    else:
        return np.fliplr(np.fliplr(x).cumsum(1))


def entropy(x):
    """The entropy of x as if x is a probability distribution"""
    histogram = scind.histogram(x.astype(float), np.min(x), np.max(x), 256)
    n = np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram != 0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram)) / n
    else:
        return 0


def entropy2(x, y):
    """Joint entropy of paired samples X and Y"""
    #
    # Bin each image into 256 gray levels
    #
    x = (stretch(x) * 255).astype(int)
    y = (stretch(y) * 255).astype(int)
    #
    # create an image where each pixel with the same X & Y gets
    # the same value
    #
    xy = 256 * x + y
    xy = xy.flatten()
    sparse = scipy.sparse.coo_matrix(
        (np.ones(xy.shape, dtype=np.int32), (xy, np.zeros(xy.shape, dtype=np.int32)))
    )
    histogram = sparse.toarray()
    n = np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram > 0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram)) / n
    else:
        return 0


def reshape_image(source, new_shape):
    """Reshape an image to a larger shape, padding with zeros"""
    if tuple(source.shape) == tuple(new_shape):
        return source

    result = np.zeros(new_shape, source.dtype)
    result[: source.shape[0], : source.shape[1]] = source
    return result
