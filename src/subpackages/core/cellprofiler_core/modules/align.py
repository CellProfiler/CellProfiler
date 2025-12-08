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
# import scipy.ndimage as scind
# import scipy.sparse
# from centrosome.filter import stretch
# from scipy.fftpack import fft2, ifft2

from ..constants.measurement import COLTYPE_INTEGER
from ..image import Image
from ..module import Module
from ..setting import Divider, SettingsGroup
from ..setting.choice import Choice
from ..setting.do_something import DoSomething, RemoveSettingButton
from ..setting.subscriber import ImageSubscriber
from ..setting.text import ImageName

from cellprofiler_library.opts.align import AlignmentMethod, CropMode, AdditionalAlignmentChoice, MEASUREMENT_FORMAT, M_ALL, C_ALIGN
from cellprofiler_library.modules._align import adjust_offsets, apply_alignment, align_images
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

""".format(
    **{
        "M_MUTUAL_INFORMATION": AlignmentMethod.MUTUAL_INFORMATION.value,
        "M_CROSS_CORRELATION": AlignmentMethod.CROSS_CORRELATION.value,
    }
)
        )

        self.crop_mode = Choice(
            "Crop mode",
            [CropMode.CROP.value, CropMode.PAD.value, CropMode.SAME_SIZE.value],
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
   """.format(
       **{
           "C_CROP": CropMode.CROP.value,
           "C_PAD": CropMode.PAD.value,
           "C_SAME_SIZE": CropMode.SAME_SIZE.value,
       }
   ),
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
                [AdditionalAlignmentChoice.SIMILARLY.value, AdditionalAlignmentChoice.SEPARATELY.value],
                doc="""\
An additional image can either be aligned similarly to the second one or
a separate alignment to the first image can be calculated:

-  *%(A_SIMILARLY)s:* The same alignment measurements obtained from the
   first two input images are applied to this additional image.
-  *%(A_SEPARATELY)s:* A new set of alignment measurements are
   calculated for this additional image using the alignment method
   specified with respect to the first input image.
""".format(
    **{
        "A_SIMILARLY": AdditionalAlignmentChoice.SIMILARLY.value,
        "A_SEPARATELY": AdditionalAlignmentChoice.SEPARATELY.value,
    }
),
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
            if additional.align_choice == AdditionalAlignmentChoice.SIMILARLY.value:
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
        offsets, shapes = adjust_offsets(offsets, shapes, self.crop_mode)

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
        image1_mask = image1.mask
        image2_mask = image2.mask
        return align_images(image1_pixels, image2_pixels, image1_mask, image2_mask, self.alignment_method)


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
        image_mask = image.mask

        output_pixels, output_mask, crop_mask = apply_alignment(
            pixel_data, 
            image_mask,
            off_x, 
            off_y, 
            shape,
        )

        output_image = Image(
            output_pixels, mask=output_mask, crop_mask=crop_mask, parent_image=image
        )
        workspace.image_set.add(output_image_name, output_image)

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
                    + [CropMode.CROP.value if setting_values[1] == "Yes" else CropMode.SAME_SIZE.value]
                    + setting_values[2:]
            )
            variable_revision_number = 3

        return setting_values, variable_revision_number
