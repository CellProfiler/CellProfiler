# coding=utf-8

"""
Watershed
=========

**Watershed** is a segmentation algorithm. It is used to separate
different objects in an image.
"""

import mahotas
import numpy
import scipy.ndimage
import skimage.color
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.transform

import cellprofiler.image
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting


class Watershed(cellprofiler.module.ImageSegmentation):
    category = "Advanced"

    module_name = "Watershed"

    variable_revision_number = 1

    def create_settings(self):
        super(Watershed, self).create_settings()

        self.operation = cellprofiler.setting.Choice(
            "Generate from",
            [
                "Distance",
                "Markers"
            ],
            "Distance",
            doc="""Select a method of inputs for the watershed algorithm:
            <ul>
                <li>
                    <i>Distance</i> (default): This is classical nuclei segmentation using watershed. Your "Input" image
                    should be a binary image. Markers and other inputs for the watershed algorithm will be
                    automatically generated.
                </li>
                <br>
                <li>
                    <i>Markers</i>: Use manually generated markers and supply an optional mask for watershed. Watershed
                    works best when the "Input" image has high intensity surrounding regions of interest and low intensity
                    inside regions of interest. Refer to the documentation for the other available options for more
                    information.
                </li>
            </ul>
            """
        )

        self.markers_name = cellprofiler.setting.ImageNameSubscriber(
            "Markers",
            doc="An image marking the approximate centers of the objects for "
                "segmentation. "
        )

        self.mask_name = cellprofiler.setting.ImageNameSubscriber(
            "Mask",
            can_be_blank=True,
            doc="Optional. Only regions not blocked by the mask will be "
                "segmented. "
        )

        self.connectivity = cellprofiler.setting.Integer(
            minval=1,
            text="Connectivity",
            value=8,
        )

        self.downsample = cellprofiler.setting.Integer(
            doc="Downsample an n-dimensional image by local averaging. If "
                "the downsampling factor is 1, the image is not downsampled.",
            minval=1,
            text="Downsample",
            value=1
        )

    def settings(self):
        __settings__ = super(Watershed, self).settings()

        return __settings__ + [
            self.operation,
            self.markers_name,
            self.mask_name,
            self.connectivity,
            self.downsample
        ]

    def visible_settings(self):
        __settings__ = super(Watershed, self).settings()

        __settings__ = __settings__ + [
            self.operation
        ]

        if self.operation.value == "Distance":
            __settings__ = __settings__ + [
                self.connectivity,
                self.downsample
            ]
        else:
            __settings__ = __settings__ + [
                self.markers_name,
                self.mask_name
            ]

        return __settings__

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        if self.operation.value == "Distance":
            original_shape = x_data.shape

            factor = self.downsample.value

            if factor > 1:
                if x.volumetric:
                    factors = (1, factor, factor)
                else:
                    factors = (factor, factor)

                x_data = skimage.transform.downscale_local_mean(
                    x_data,
                    factors
                )

            threshold = skimage.filters.threshold_otsu(x_data)

            x_data = x_data > threshold

            distance = scipy.ndimage.distance_transform_edt(x_data)

            distance = mahotas.stretch(distance)

            surface = distance.max() - distance

            if x.volumetric:
                footprint = numpy.ones(
                    (
                        self.connectivity.value,
                        self.connectivity.value,
                        self.connectivity.value
                    )
                )
            else:
                footprint = numpy.ones(
                    (
                        self.connectivity.value,
                        self.connectivity.value
                    )
                )

            peaks = mahotas.regmax(distance, footprint)

            if x.volumetric:
                markers, _ = mahotas.label(peaks, numpy.ones((16, 16, 16)))
            else:
                markers, _ = mahotas.label(peaks, numpy.ones((16, 16)))

            y_data = mahotas.cwatershed(surface, markers)

            y_data = y_data * x_data

            if factor > 1:
                y_data = skimage.transform.resize(
                    y_data,
                    original_shape,
                    mode="edge",
                    order=0,
                    preserve_range=True
                )

                y_data = numpy.rint(y_data).astype(numpy.uint16)
        else:
            markers_name = self.markers_name.value

            markers = images.get_image(markers_name)

            markers_data = markers.pixel_data

            if x.multichannel:
                x_data = skimage.color.rgb2gray(x_data)

            if markers.multichannel:
                markers_data = skimage.color.rgb2gray(markers_data)

            mask_data = None

            if not self.mask_name.is_blank:
                mask_name = self.mask_name.value

                mask = images.get_image(mask_name)

                mask_data = mask.pixel_data

            y_data = skimage.morphology.watershed(
                image=x_data,
                markers=markers_data,
                mask=mask_data
            )

        y_data = skimage.measure.label(y_data)

        objects = cellprofiler.object.Objects()

        objects.segmented = y_data

        objects.parent_image = x

        workspace.object_set.add_objects(objects, y_name)

        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = x.pixel_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions
