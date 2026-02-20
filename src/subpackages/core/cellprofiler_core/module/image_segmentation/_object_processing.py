from cellprofiler_core.setting.subscriber import LabelSubscriber

from ._image_segmentation import ImageSegmentation
from ...constants.measurement import COLTYPE_INTEGER
from ...constants.measurement import C_CHILDREN
from ...constants.measurement import C_NUMBER
from ...constants.measurement import C_PARENT
from ...constants.measurement import FF_CHILDREN_COUNT
from ...constants.measurement import FF_COUNT
from ...constants.measurement import FF_PARENT
from ...constants.measurement import FTR_OBJECT_NUMBER
from ...object import Objects
from cellprofiler_library.functions.measurement import get_object_processing_measurements
from cellprofiler_core.utilities.core.workspace import add_library_measurements_to_workspace_measurements

class ObjectProcessing(ImageSegmentation):
    category = "Object Processing"

    def add_measurements(
        self, workspace, input_object_name=None, output_object_name=None
    ):
        if input_object_name is None:
            input_object_name = self.x_name.value

        if output_object_name is None:
            output_object_name = self.y_name.value

        #
        # Output object name, labels, volumetric, count needed to run get_image_segmentation_measurements
        #
        output_objects = workspace.object_set.get_objects(output_object_name)
        output_objects_labels = output_objects.segmented
        output_objects_volumetric = output_objects.volumetric
        output_objects_count = output_objects.count

        #
        # Input object name, labels, ijv, and output object ijv needed to run relate_children inside get_object_processing_measurements, This is the parent object
        #
        input_objects = workspace.object_set.get_objects(input_object_name)
        input_objects_labels = input_objects.segmented
        input_objects_ijv = input_objects.ijv
        output_objects_ijv = output_objects.ijv

        lib_measurements = get_object_processing_measurements(
            output_objects_labels, output_objects_volumetric, output_objects_count, 
            output_object_name, output_objects_ijv, 
            input_object_name, input_objects_labels, input_objects_ijv, 
        )
        add_library_measurements_to_workspace_measurements(workspace.measurements, lib_measurements)


    def create_settings(self):
        super(ObjectProcessing, self).create_settings()

        self.x_name = LabelSubscriber(
            "Select the input object", doc="Select the object you want to use."
        )

    def display(self, workspace, figure):
        layout = (2, 1)

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions, subplots=layout
        )

        figure.subplot_imshow_labels(
            image=workspace.display_data.x_data, title=self.x_name.value, x=0, y=0
        )

        figure.subplot_imshow_labels(
            image=workspace.display_data.y_data,
            sharexy=figure.subplot(0, 0),
            title=self.y_name.value,
            x=1,
            y=0,
        )

    def get_categories(self, pipeline, object_name):
        if object_name == self.x_name.value:
            return [C_CHILDREN]

        categories = super(ObjectProcessing, self).get_categories(pipeline, object_name)

        if object_name == self.y_name.value:
            return categories + [C_PARENT]

        return categories

    def get_measurement_columns(self, pipeline, additional_objects=None):
        if additional_objects is None:
            additional_objects = []
        object_names = [(self.x_name.value, self.y_name.value)] + additional_objects

        columns = [
            super(ObjectProcessing, self).get_measurement_columns(
                pipeline, output_object_name
            )
            + [
                (
                    input_object_name,
                    FF_CHILDREN_COUNT % output_object_name,
                    COLTYPE_INTEGER,
                ),
                (output_object_name, FF_PARENT % input_object_name, COLTYPE_INTEGER,),
            ]
            for (input_object_name, output_object_name) in object_names
        ]

        return sum(columns, [])

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.x_name.value and category == C_CHILDREN:
            return ["%s_Count" % self.y_name.value]

        if object_name == self.y_name.value:
            if category == C_NUMBER:
                return [FTR_OBJECT_NUMBER]

            if category == C_PARENT:
                return [self.x_name.value]

        return super(ObjectProcessing, self).get_measurements(
            pipeline, object_name, category
        )

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        objects = workspace.object_set

        x = objects.get_objects(x_name)

        dimensions = x.dimensions

        x_data = x.segmented

        args = (setting.value for setting in self.settings()[2:])

        y_data = self.function(x_data, *args)

        y = Objects()

        y.segmented = y_data

        y.parent_image = x.parent_image

        objects.add_objects(y, y_name)

        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions

    def volumetric(self):
        return True
