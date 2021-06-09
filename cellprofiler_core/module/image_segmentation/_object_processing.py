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


class ObjectProcessing(ImageSegmentation):
    category = "Object Processing"

    def add_measurements(
        self, workspace, input_object_name=None, output_object_name=None
    ):
        if input_object_name is None:
            input_object_name = self.x_name.value

        if output_object_name is None:
            output_object_name = self.y_name.value

        super(ObjectProcessing, self).add_measurements(workspace, output_object_name)

        objects = workspace.object_set.get_objects(output_object_name)

        parent_objects = workspace.object_set.get_objects(input_object_name)

        children_per_parent, parents_of_children = parent_objects.relate_children(
            objects
        )

        workspace.measurements.add_measurement(
            input_object_name,
            FF_CHILDREN_COUNT % output_object_name,
            children_per_parent,
        )

        workspace.measurements.add_measurement(
            output_object_name, FF_PARENT % input_object_name, parents_of_children,
        )

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
