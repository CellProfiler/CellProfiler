"""

Watershed

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting
import numpy
import SimpleITK


class Watershed(cellprofiler.module.Module):
    category = "Volumetric"
    module_name = "Watershed"
    variable_revision_number = 1

    def create_settings(self):
        self.input_image = cellprofiler.setting.ImageNameSubscriber(
            "Input image",
            "---SELECT---"
        )

        self.output_object = cellprofiler.setting.ObjectNameProvider(
            "Output object",
            ""
        )

        self.mask = cellprofiler.setting.ImageNameSubscriber(
            "Mask",
            "---SELECT---"
        )

        self.invert_mask = cellprofiler.setting.Binary(
            "Invert Mask",
            False
        )

    def settings(self):
        return [
            self.input_image,
            self.output_object,
            self.mask,
            self.invert_mask
        ]

    def visible_settings(self):
        return [
            self.input_image,
            self.output_object,
            self.mask,
            self.invert_mask
        ]

    def run(self, workspace):
        images = workspace.image_set

        image = images.get_image(self.input_image.value)

        mask = images.get_image(self.mask.value)

        if self.invert_mask.value:
            features = SimpleITK.GetImageFromArray(numpy.logical_not(mask.pixel_data) * 1.0)
        else:
            features = SimpleITK.GetImageFromArray(mask.pixel_data * 1.0)

        features_watershed = SimpleITK.MorphologicalWatershed(
            features,
            markWatershedLine=True,
            fullyConnected=False
        ) # TODO: Expose/configure "level" option

        # TODO: Requires objects to be cleared from edges.
        # Alternatively, remove the mode (scipy.stats) though finding
        # the mode is expensive. Perhaps perform over only one slice?
        connected_segmentation = SimpleITK.ConnectedComponent(features_watershed != features_watershed[0,0,0])

        filled = SimpleITK.BinaryFillhole(connected_segmentation != 0)

        distances = SimpleITK.SignedMaurerDistanceMap(
            filled,
            insideIsPositive=False,
            squaredDistance=False,
            useImageSpacing=False
        )

        distances_watershed = SimpleITK.MorphologicalWatershed(
            distances,
            markWatershedLine=False,
            level=1
        ) # TODO: What is "level"? Do we tune it?
        # Level refers to "minimum dynamic of minima" -- excludes minima lower than minima at "level"?

        segmentation = SimpleITK.Mask(
            distances_watershed,
            SimpleITK.Cast(
                connected_segmentation,
                distances_watershed.GetPixelID()
            )
        )

        segmentation = SimpleITK.GetArrayFromImage(segmentation)

        output_object = cellprofiler.object.Objects()
        output_object.segmented = segmentation
        workspace.object_set.add_objects(output_object, self.output_object.value)

        if self.show_window:
            workspace.display_data.image = image.pixel_data
            workspace.display_data.segmentation = segmentation

    def display(self, workspace, figure):
        dimensions = (2, 1)

        image = workspace.display_data.image[16]
        segmentation = workspace.display_data.segmentation[16]

        figure.set_subplots(dimensions)

        figure.subplot_imshow(
            0,
            0,
            image,
            colormap="gray"
        )

        figure.subplot_imshow_labels(
            1,
            0,
            segmentation
        )
