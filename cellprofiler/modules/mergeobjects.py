"""

"""

import scipy.ndimage
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting

class MergeObjects(cellprofiler.module.Module):
    module_name = "MergeObjects"
    category = "Image Processing"
    variable_revision_number = 1

    def create_settings(self):
        self.input_object_a = cellprofiler.setting.ObjectNameSubscriber(
            "Input object name:"
        )

        self.input_object_b = cellprofiler.setting.ObjectNameSubscriber(
            "Input object name:"
        )

        self.output_object = cellprofiler.setting.ObjectNameProvider(
            "Output object name:"
        )

    def settings(self):
        return [
            self.input_object_a,
            self.input_object_b,
            self.output_object
        ]

    def visible_settings(self):
        return [
            self.input_object_a,
            self.input_object_b,
            self.output_object
        ]

    def run(self, workspace):
        input_object_a_name = self.input_object_a.value
        input_object_b_name = self.input_object_b.value
        output_object_name = self.output_object.value

        input_object_a = workspace.get_objects(input_object_a_name)
        input_object_b = workspace.get_objects(input_object_b_name)

        input_object_a_segmented = input_object_a.segmented
        input_object_b_segmented = input_object_b.segmented

        merged_segmented = input_object_a_segmented + input_object_b_segmented

        labeled_image, object_count = scipy.ndimage.label(merged_segmented)

        output_object = cellprofiler.object.Objects()
        output_object.set_segmented(labeled_image)

        workspace.object_set.add_objects(output_object, output_object_name)

        if self.show_window:
            workspace.display_data.input_object_a_name = input_object_a_name
            workspace.display_data.input_object_a = input_object_a.segmented
            workspace.display_data.input_object_b_name = input_object_b_name
            workspace.display_data.input_object_b = input_object_b.segmented
            workspace.display_data.output_object_name = output_object_name
            workspace.display_data.output_object = output_object.segmented

    #
    # display lets you use matplotlib to display your results.
    #
    def display(self, workspace, figure):
        figure.set_subplots((3, 1))

        # TODO: zoom in on all 3 images simultaneously

        # Display image 3 times w/ input object a, input object b, and merged output object:
        figure.subplot_imshow_labels(0, 0, workspace.display_data.input_object_a,
                                     workspace.display_data.input_object_a_name)
        figure.subplot_imshow_labels(1, 0, workspace.display_data.input_object_b,
                                     workspace.display_data.input_object_b_name)
        figure.subplot_imshow_labels(2, 0, workspace.display_data.output_object,
                                     workspace.display_data.output_object_name)
