# -*- coding: utf-8 -*-

"""

Clear objects connected to the label image border.

"""

import skimage.measure
import skimage.segmentation

import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting


class ClearBorder(cellprofiler.module.ObjectProcessing):
    category = "Advanced"

    module_name = "ClearBorder"

    variable_revision_number = 1

    def create_settings(self):
        super(ClearBorder, self).create_settings()

        self.buffer_size = cellprofiler.setting.Integer(
            doc="""
            The width of the border examined. By default, only objects that 
            touch the outside of the image are removed.
            """,
            minval=0,
            text="Buffer size",
            value=0
        )

    def settings(self):
        __settings__ = super(ClearBorder, self).settings()

        return __settings__ + [
            self.buffer_size
        ]

    def visible_settings(self):
        __settings__ = super(ClearBorder, self).visible_settings()

        return __settings__ + [
            self.buffer_size
        ]

    def run(self, workspace):
        self.function = skimage.segmentation.clear_border

        super(ClearBorder, self).run(workspace)
