"""

"""

import numpy
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

        self.operation = cellprofiler.setting.Choice(
            "Operation",
            [
                "A",
                "B",
                "Intersection",
                "Union",
                "XOR"
            ]
        )

    def settings(self):
        return [
            self.input_object_a,
            self.input_object_b,
            self.output_object,
            self.operation
        ]

    def visible_settings(self):
        return [
            self.input_object_a,
            self.input_object_b,
            self.output_object,
            self.operation
        ]

    # TODO: make static?
    def overlay_objects(self, labels_top, labels_bottom):
        labels_bottom[labels_top > 0] = labels_bottom.max() + labels_top[labels_top > 0]
        object = cellprofiler.object.Objects()
        object.segmented = labels_bottom
        return object

    def run(self, workspace):
        input_object_a_name = self.input_object_a.value
        input_object_b_name = self.input_object_b.value
        output_object_name = self.output_object.value

        input_object_a = workspace.get_objects(input_object_a_name)
        input_object_b = workspace.get_objects(input_object_b_name)

        s1 = input_object_a.segmented
        s2 = input_object_b.segmented

        if self.operation == "Intersection":
            # Increment the intersections of the segments by the maximum label on
            # one of the segmentation matrices. Then overlay that on top of the unaltered
            # segmentation matrix. Less likely to overflow than multiplication.
            c = numpy.max([s1.max(), s2.max()])
            intersection = numpy.logical_and(s1 > 0, s2 > 0)
            s1[intersection] = c + s1[intersection] # TODO: augmented assignment?

            output_object = self.overlay_objects(s1, s2)
        elif self.operation == "Union":
            c = numpy.max([s1.max(), s2.max()]) + 1
            union = numpy.logical_xor(s1 > 0, s2 > 0)
            s1[union] = c + s2[union] # TODO: augmented assignment?

            output_object = self.overlay_objects(s1, s2)
        elif self.operation == "A":
            output_object = self.overlay_objects(s1, s2)
        elif self.operation == "B":
            output_object = self.overlay_objects(s2, s1)
        elif self.operation == "XOR":
            zeros = numpy.logical_not(numpy.logical_xor(s1 > 0, s2 > 0))
            s1[zeros] = 0
            s2[zeros] = 0
            s2[s2 > 0] = s1.max() + s2[s2 > 0]
            merged = s1 + s2
            output_object = cellprofiler.object.Objects()
            output_object.segmented = merged

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
