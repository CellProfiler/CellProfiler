"""
Closing
=======

**Closing** is the erosion of the dilation of an image. It’s used to
remove pepper noise (small dark spots) and connect small bright cracks. 
See `this tutorial <https://scikit-image.org/docs/dev/auto_examples/applications/plot_morphology.html#closing>`__ for more information.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES           NO
============ ============ ===============

"""

from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import StructuringElement
from cellprofiler.library.modules import closing

from ._help import HELP_FOR_STREL


class Closing(ImageProcessing):
    category = "Advanced"

    module_name = "Closing"

    variable_revision_number = 1

    def create_settings(self):
        super(Closing, self).create_settings()

        self.structuring_element = StructuringElement(
            allow_planewise=True, doc=HELP_FOR_STREL
        )

        self.planewise = Binary(
            text="Planewise closing",
            value=False,
            doc="""\
Select "*{YES}*" to perform closing on a per-plane level. 
This will perform closing on each plane of a 
3D image, rather than on the image as a whole.
**Note**: Planewise operations will be considerably slower.
""".format(
                **{"YES": "Yes"}
            ),
        )

    def settings(self):
        __settings__ = super(Closing, self).settings()

        return __settings__ + [self.structuring_element, self.planewise]

    def visible_settings(self):
        __settings__ = super(Closing, self).settings()

        return __settings__ + [self.structuring_element, self.planewise]

    def run(self, workspace):

        x = workspace.image_set.get_image(self.x_name.value)

        self.function = (
            lambda image, structuring_element, structuring_element_size, planewise: closing(
                image,
                structuring_element=self.structuring_element.shape,
                diameter=self.structuring_element.size,
                planewise=self.planewise.value,
            )
        )   

        # is_strel_2d = self.structuring_element.value.ndim == 2

        # is_img_2d = x.pixel_data.ndim == 2

        # if is_strel_2d and not is_img_2d:

            # self.function = planewise_morphology_closing

        # elif not is_strel_2d and is_img_2d:

        #     raise NotImplementedError(
        #         "A 3D structuring element cannot be applied to a 2D image."
        #     )

        # else:

        #     self.function = skimage.morphology.closing

        super(Closing, self).run(workspace)


def planewise_morphology_closing(x_data, structuring_element):
    y_data = numpy.zeros_like(x_data)

    for index, plane in enumerate(x_data):

        y_data[index] = skimage.morphology.closing(plane, structuring_element)

    return y_data
