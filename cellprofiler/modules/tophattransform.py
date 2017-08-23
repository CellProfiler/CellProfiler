# coding=utf-8

"""
Top-hat transform
=================

Perform a black or white top-hat transform on grayscale pixel data.

Top-hat transforms are useful for extracting small elements and details
from images and volumes.
"""

import skimage.morphology

import cellprofiler.module
import cellprofiler.setting


class TopHatTransform(cellprofiler.module.ImageProcessing):
    module_name = "TopHatTransform"

    variable_revision_number = 1

    def create_settings(self):
        super(TopHatTransform, self).create_settings()

        self.operation_name = cellprofiler.setting.Choice(
            choices=[
                "Black top-hat transform",
                "White top-hat transform"
            ],
            text="Operation",
            value="Black top-hat transform",
            doc="""
            Select the top-hat transformation:
            <ul>
                <li><i>Black top-hat transform</i>: This operation returns the dark spots of the image that are smaller
                than the structuring element. Note that dark spots in the original image are bright spots after the
                black top hat.</li>
                <li><i>White top-hat transform</i>: This operation returns the bright spots of the image that are
                smaller than the structuring element.</li>
            </ul>
            """
        )

        self.structuring_element = cellprofiler.setting.StructuringElement()

    def settings(self):
        __settings__ = super(TopHatTransform, self).settings()

        return __settings__ + [
            self.structuring_element,
            self.operation_name
        ]

    def visible_settings(self):
        __settings__ = super(TopHatTransform, self).visible_settings()

        return __settings__ + [
            self.operation_name,
            self.structuring_element
        ]

    def run(self, workspace):
        self.function = tophat_transform

        super(TopHatTransform, self).run(workspace)


def tophat_transform(image, structuring_element, operation):
    if operation == "Black top-hat transform":
        return skimage.morphology.black_tophat(image, selem=structuring_element)

    return skimage.morphology.white_tophat(image, selem=structuring_element)
