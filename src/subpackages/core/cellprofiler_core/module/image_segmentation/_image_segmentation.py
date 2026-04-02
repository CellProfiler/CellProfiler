from cellprofiler_core.object import Objects

from .._module import Module
from ...constants.measurement import COLTYPE_FLOAT
from ...constants.measurement import COLTYPE_INTEGER
from ...constants.measurement import C_COUNT
from ...constants.measurement import C_LOCATION
from ...constants.measurement import C_NUMBER
from ...constants.measurement import FF_COUNT
from ...constants.measurement import FTR_CENTER_X
from ...constants.measurement import FTR_CENTER_Y
from ...constants.measurement import FTR_CENTER_Z
from ...constants.measurement import FTR_OBJECT_NUMBER
from ...constants.measurement import IMAGE
from ...constants.measurement import M_LOCATION_CENTER_X
from ...constants.measurement import M_LOCATION_CENTER_Y
from ...constants.measurement import M_LOCATION_CENTER_Z
from ...constants.measurement import M_NUMBER_OBJECT_NUMBER
from ...setting.subscriber import ImageSubscriber
from ...setting.text import LabelName
from cellprofiler_core.utilities.core.workspace import add_library_measurements_to_workspace_measurements
from cellprofiler_library.functions.measurement import get_image_segmentation_measurements

class ImageSegmentation(Module):
    category = "Image Segmentation"

    def __init__(self):
        self.validate_image = None
        super().__init__()

    def add_measurements(self, workspace, object_name=None):
        if object_name is None:
            object_name = self.y_name.value

        objects = workspace.object_set.get_objects(object_name)
        # I have tested this when debugging relateobjects while working on the library/object_count _measurements branch and confirmed that it works.
        lib_measurements = get_image_segmentation_measurements(
            objects.segmented,
            objects.volumetric,
            objects.count,
            object_name,
        )

        add_library_measurements_to_workspace_measurements(workspace.measurements, lib_measurements)

    def create_settings(self):
        self.x_name = ImageSubscriber(
            "Select the input image", doc="Select the image you want to use."
        )

        self.y_name = LabelName(
            "Name the output object",
            self.__class__.__name__,
            doc="Enter the name you want to call the object produced by this module.",
        )

    def display(self, workspace, figure):
        layout = (2, 1)

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions, subplots=layout
        )

        figure.subplot_imshow(
            colormap="gray",
            image=workspace.display_data.x_data,
            title=self.x_name.value,
            x=0,
            y=0,
        )

        figure.subplot_imshow_labels(
            background_image=workspace.display_data.x_data,
            image=workspace.display_data.y_data,
            sharexy=figure.subplot(0, 0),
            title=self.y_name.value,
            x=1,
            y=0,
        )

    def get_categories(self, pipeline, object_name):
        if object_name == IMAGE:
            return [C_COUNT]

        if object_name == self.y_name.value:
            return [
                C_LOCATION,
                C_NUMBER,
            ]

        return []

    def get_measurement_columns(self, pipeline, object_name=None):
        if object_name is None:
            object_name = self.y_name.value

        return [
            (object_name, M_LOCATION_CENTER_X, COLTYPE_FLOAT,),
            (object_name, M_LOCATION_CENTER_Y, COLTYPE_FLOAT,),
            (object_name, M_LOCATION_CENTER_Z, COLTYPE_FLOAT,),
            (object_name, M_NUMBER_OBJECT_NUMBER, COLTYPE_INTEGER,),
            (IMAGE, FF_COUNT % object_name, COLTYPE_INTEGER,),
        ]

    def get_measurements(self, pipeline, object_name, category):
        if object_name == IMAGE and category == C_COUNT:
            return [self.y_name.value]

        if object_name == self.y_name.value:
            if category == C_LOCATION:
                return [
                    FTR_CENTER_X,
                    FTR_CENTER_Y,
                    FTR_CENTER_Z,
                ]

            if category == C_NUMBER:
                return [FTR_OBJECT_NUMBER]

        return []

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        args = (setting.value for setting in self.settings()[2:])

        if self.validate_image:
            self.validate_image(x)

        y_data = self.function(x_data, *args)

        y = Objects()

        y.segmented = y_data

        y.parent_image = x.parent_image

        objects = workspace.object_set

        objects.add_objects(y, y_name)

        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions

    def settings(self):
        return [self.x_name, self.y_name]

    def visible_settings(self):
        return [self.x_name, self.y_name]

    def volumetric(self):
        return True
