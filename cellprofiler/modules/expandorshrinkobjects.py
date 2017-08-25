# coding=utf-8

"""
**Expand Or Shrink Objects** expands or shrinks objects by a defined
distance.

The module expands or shrinks objects by adding or removing border
pixels. You can specify a certain number of border pixels to be added or
removed, expand objects until they are almost touching or shrink objects
down to a point. Objects are never lost using this module (shrinking
stops when an object becomes a single pixel). The module can separate
touching objects without otherwise shrinking the objects.

**ExpandOrShrinkObjects** can perform some specialized morphological
operations that remove pixels without completely removing an object. See
the Settings help (below) for more detail.

*Special note on saving images:* You can use the settings in this module
to pass object outlines along to the module **OverlayOutlines** and then
save them with the **SaveImages** module. You can also pass the
identified objects themselves along to the object processing module
**ConvertToImage** and then save them with the **SaveImages** module.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^

**Image measurements:**

-  *Count:* Number of expanded/shrunken objects in the image.

**Object measurements:**

-  *Location\_X, Location\_Y:* Pixel (*X,Y*) coordinates of the center
   of mass of the expanded/shrunken objects.

See also **Identify** modules.
"""

import centrosome.cpmorphology
import centrosome.outline
import numpy
import scipy.ndimage

import cellprofiler.gui.help
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.identify
import cellprofiler.object
import cellprofiler.setting

O_SHRINK_INF = "Shrink objects to a point"
O_EXPAND_INF = "Expand objects until touching"
O_DIVIDE = "Add partial dividing lines between objects"
O_SHRINK = "Shrink objects by a specified number of pixels"
O_EXPAND = "Expand objects by a specified number of pixels"
O_SKELETONIZE = "Skeletonize each object"
O_SPUR = "Remove spurs"
O_ALL = [O_SHRINK_INF, O_EXPAND_INF, O_DIVIDE, O_SHRINK, O_EXPAND, O_SKELETONIZE, O_SPUR]


class ExpandOrShrinkObjects(cellprofiler.module.Module):
    module_name = "ExpandOrShrinkObjects"
    category = "Object Processing"
    variable_revision_number = 1

    def create_settings(self):
        self.object_name = cellprofiler.setting.ObjectNameSubscriber(
            "Select the input objects",
            cellprofiler.setting.NONE,
            doc="Select the objects that you want to expand or shrink."
        )

        self.output_object_name = cellprofiler.setting.ObjectNameProvider(
            "Name the output objects",
            "ShrunkenNuclei",
            doc="Enter a name for the resulting objects."
        )

        self.operation = cellprofiler.setting.Choice(
            "Select the operation",
            O_ALL,
            doc="""
            Choose the operation that you want to perform:
            <ul>
                <li><i>{O_SHRINK_INF}:</i> Remove all pixels but one from filled objects. Thin objects with
                holes to loops unless the "fill" option is checked.</li>
                <li><i>{O_EXPAND_INF}:</i> Expand objects, assigning every pixel in the image to an object.
                Background pixels are assigned to the nearest object.</li>
                <li><i>{O_DIVIDE}:</i> Remove pixels from an object that are adjacent to another object's
                pixels unless doing so would change the object's Euler number (break an object in two, remove
                the object completely or open a hole in an object).</li>
                <li><i>{O_SHRINK}:</i> Remove pixels around the perimeter of an object unless doing so would
                change the object's Euler number (break the object in two, remove the object completely or open
                a hole in the object). You can specify the number of times perimeter pixels should be removed.
                Processing stops automatically when there are no more pixels to remove.</li>
                <li><i>{O_EXPAND}:</i> Expand each object by adding background pixels adjacent to the image.
                You can choose the number of times to expand. Processing stops automatically if there are no
                more background pixels.</li>
                <li><i>{O_SKELETONIZE}:</i> Erode each object to its skeleton.</li>
                <li><i>{O_SPUR}:</i> Remove or reduce the length of spurs in a skeletonized image. The
                algorithm reduces spur size by the number of pixels indicated in the setting <i>Number of
                pixels by which to expand or shrink</i>.</li>
            </ul>
            """.format(**{
                "O_DIVIDE": O_DIVIDE,
                "O_EXPAND": O_EXPAND,
                "O_EXPAND_INF": O_EXPAND_INF,
                "O_SHRINK": O_SHRINK,
                "O_SHRINK_INF": O_SHRINK_INF,
                "O_SKELETONIZE": O_SKELETONIZE,
                "O_SPUR": O_SPUR
            })
        )

        self.iterations = cellprofiler.setting.Integer("Number of pixels by which to expand or shrink", 1, minval=1)

        self.wants_fill_holes = cellprofiler.setting.Binary(
            "Fill holes in objects so that all objects shrink to a single point?",
            False,
            doc="""
            <i>(Used only if one of the "shrink" options selected)</i><br>
            Select <i>{YES}</i> to ensure that each object will shrink to a single point, by filling the holes
            in each object.
            <p>Select <i>{NO}</i> to preserve the Euler number. in this case, the shrink algorithm preserves
            each object's Euler number, which means that it will erode an object with a hole to a ring in order
            to keep the hole. An object with two holes will be shrunk to two rings connected by a line in order
            to keep from breaking up the object or breaking the hole.</p>
            """.format(**{
                "NO": cellprofiler.setting.NO,
                "YES": cellprofiler.setting.YES
            })
        )

        self.wants_outlines = cellprofiler.setting.Binary(
            "Retain the outlines of the identified objects?",
            False,
            doc=cellprofiler.gui.help.RETAINING_OUTLINES_HELP
        )

        self.outlines_name = cellprofiler.setting.OutlineNameProvider(
            "Name the outline image",
            "ShrunkenNucleiOutlines",
            doc=cellprofiler.gui.help.NAMING_OUTLINES_HELP
        )

    def settings(self):
        return [
            self.object_name,
            self.output_object_name,
            self.operation,
            self.iterations,
            self.wants_fill_holes,
            self.wants_outlines,
            self.outlines_name
        ]

    def visible_settings(self):
        result = [
            self.object_name,
            self.output_object_name,
            self.operation
        ]

        if self.operation in [O_SHRINK, O_EXPAND, O_SPUR]:
            result += [self.iterations]

        if self.operation in [O_SHRINK, O_SHRINK_INF]:
            result += [self.wants_fill_holes]

        result += [self.wants_outlines]

        if self.wants_outlines.value:
            result += [self.outlines_name]

        return result

    def run(self, workspace):
        input_objects = workspace.object_set.get_objects(self.object_name.value)

        output_objects = cellprofiler.object.Objects()

        output_objects.segmented = self.do_labels(input_objects.segmented)

        if (input_objects.has_small_removed_segmented and self.operation not in (O_EXPAND, O_EXPAND_INF, O_DIVIDE)):
            output_objects.small_removed_segmented = self.do_labels(input_objects.small_removed_segmented)

        if (input_objects.has_unedited_segmented and self.operation not in (O_EXPAND, O_EXPAND_INF, O_DIVIDE)):

            output_objects.unedited_segmented = self.do_labels(input_objects.unedited_segmented)

        workspace.object_set.add_objects(output_objects, self.output_object_name.value)

        cellprofiler.modules.identify.add_object_count_measurements(
            workspace.measurements,
            self.output_object_name.value,
            numpy.max(output_objects.segmented)
        )

        cellprofiler.modules.identify.add_object_location_measurements(
            workspace.measurements,
            self.output_object_name.value,
            output_objects.segmented
        )

        if self.wants_outlines.value:
            outline_image = cellprofiler.image.Image(
                centrosome.outline.outline(output_objects.segmented) > 0,
                parent_image=input_objects.parent_image
            )

            workspace.image_set.add(self.outlines_name.value, outline_image)

        if self.show_window:
            workspace.display_data.input_objects_segmented = input_objects.segmented

            workspace.display_data.output_objects_segmented = output_objects.segmented

    def display(self, workspace, figure):
        input_objects_segmented = workspace.display_data.input_objects_segmented

        output_objects_segmented = workspace.display_data.output_objects_segmented

        figure.set_subplots((2, 1))

        figure.subplot_imshow_labels(
            0,
            0,
            input_objects_segmented,
            self.object_name.value
        )

        figure.subplot_imshow_labels(
            1,
            0,
            output_objects_segmented,
            self.output_object_name.value,
            sharexy=figure.subplot(0, 0)
        )

    def do_labels(self, labels):
        '''Run whatever transformation on the given labels matrix'''
        if (self.operation in (O_SHRINK, O_SHRINK_INF) and self.wants_fill_holes.value):
            labels = centrosome.cpmorphology.fill_labeled_holes(labels)

        if self.operation == O_SHRINK_INF:
            return centrosome.cpmorphology.binary_shrink(labels)

        if self.operation == O_SHRINK:
            return centrosome.cpmorphology.binary_shrink(labels, iterations=self.iterations.value)

        if self.operation in (O_EXPAND, O_EXPAND_INF):
            if self.operation == O_EXPAND_INF:
                distance = numpy.max(labels.shape)
            else:
                distance = self.iterations.value

            background = labels == 0

            distances, (i, j) = scipy.ndimage.distance_transform_edt(background, return_indices=True)

            out_labels = labels.copy()

            mask = (background & (distances <= distance))

            out_labels[mask] = labels[i[mask], j[mask]]

            return out_labels

        if self.operation == O_DIVIDE:
            #
            # A pixel must be adjacent to some other label and the object
            # must not disappear.
            #
            adjacent_mask = centrosome.cpmorphology.adjacent(labels)

            thinnable_mask = centrosome.cpmorphology.binary_shrink(labels, 1) != 0

            out_labels = labels.copy()

            out_labels[adjacent_mask & ~ thinnable_mask] = 0

            return out_labels

        if self.operation == O_SKELETONIZE:
            return centrosome.cpmorphology.skeletonize_labels(labels)

        if self.operation == O_SPUR:
            return centrosome.cpmorphology.spur(labels, iterations=self.iterations.value)

        raise NotImplementedError("Unsupported operation: %s" % self.operation.value)

    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        if from_matlab and variable_revision_number == 2:
            inf = setting_values[4] == "Inf"

            if setting_values[3] == "Expand":
                operation = O_EXPAND_INF if inf else O_EXPAND
            elif setting_values[3] == "Shrink":
                operation = (O_SHRINK_INF if inf else O_DIVIDE if setting_values[4] == "0" else O_SHRINK)

            iterations = "1" if inf else setting_values[4]

            wants_outlines = setting_values[5] != cellprofiler.setting.DO_NOT_USE

            setting_values = (
                setting_values[:2] + [
                    operation,
                    iterations,
                    cellprofiler.setting.NO,
                    cellprofiler.setting.YES if wants_outlines else cellprofiler.setting.NO,
                    setting_values[5]
                ]
            )

            from_matlab = False

            variable_revision_number = 1

        return setting_values, variable_revision_number, from_matlab

    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        columns = cellprofiler.modules.identify.get_object_measurement_columns(self.output_object_name.value)
        return columns

    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        categories = []
        if object_name == cellprofiler.measurement.IMAGE:
            categories += ["Count"]
        if object_name == self.output_object_name:
            categories += ("Location", "Number")
        return categories

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        result = []

        if object_name == cellprofiler.measurement.IMAGE:
            if category == "Count":
                result += [self.output_object_name.value]
        if object_name == self.output_object_name:
            if category == "Location":
                result += ["Center_X", "Center_Y"]
            elif category == "Number":
                result += ["Object_Number"]
        return result


#
# backwards compatability
#
ExpandOrShrink = ExpandOrShrinkObjects
