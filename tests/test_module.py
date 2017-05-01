import numpy

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.object
import cellprofiler.workspace


class TestObjectProcessing():
    def test_get_categories_image(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_categories(None, cellprofiler.measurement.IMAGE)

        expected = [
            cellprofiler.measurement.C_COUNT
        ]

        assert actual == expected

    def test_get_categories_input_object(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_categories(None, "Objects")

        expected = [
            cellprofiler.measurement.C_CHILDREN
        ]

        assert actual == expected

    def test_get_categories_output_object(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_categories(None, "ObjectProcessing")

        expected = [
            cellprofiler.measurement.C_LOCATION,
            cellprofiler.measurement.C_NUMBER,
            cellprofiler.measurement.C_PARENT
        ]

        assert actual == expected

    def test_get_categories_other(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_categories(None, "foo")

        expected = []

        assert actual == expected

    def test_get_measurement_columns(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurement_columns(None)

        expected = [
            (
                "ObjectProcessing",
                cellprofiler.measurement.M_LOCATION_CENTER_X,
                cellprofiler.measurement.COLTYPE_FLOAT
            ),
            (
                "ObjectProcessing",
                cellprofiler.measurement.M_LOCATION_CENTER_Y,
                cellprofiler.measurement.COLTYPE_FLOAT
            ),
            (
                "ObjectProcessing",
                cellprofiler.measurement.M_LOCATION_CENTER_Z,
                cellprofiler.measurement.COLTYPE_FLOAT
            ),
            (
                "ObjectProcessing",
                cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                cellprofiler.measurement.COLTYPE_INTEGER
            ),
            (
                cellprofiler.measurement.IMAGE,
                cellprofiler.measurement.FF_COUNT % "ObjectProcessing",
                cellprofiler.measurement.COLTYPE_INTEGER
            ),
            (
                "Objects",
                cellprofiler.measurement.FF_CHILDREN_COUNT % "ObjectProcessing",
                cellprofiler.measurement.COLTYPE_INTEGER
            ),
            (
                "ObjectProcessing",
                cellprofiler.measurement.FF_PARENT % "Objects",
                cellprofiler.measurement.COLTYPE_INTEGER
            )
        ]

        assert actual == expected

    def test_get_measurement_columns_additional_objects(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurement_columns(None, additional_objects=[("additional_input", "additional_output")])

        expected = [
            (
                "ObjectProcessing",
                cellprofiler.measurement.M_LOCATION_CENTER_X,
                cellprofiler.measurement.COLTYPE_FLOAT
            ),
            (
                "ObjectProcessing",
                cellprofiler.measurement.M_LOCATION_CENTER_Y,
                cellprofiler.measurement.COLTYPE_FLOAT
            ),
            (
                "ObjectProcessing",
                cellprofiler.measurement.M_LOCATION_CENTER_Z,
                cellprofiler.measurement.COLTYPE_FLOAT
            ),
            (
                "ObjectProcessing",
                cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                cellprofiler.measurement.COLTYPE_INTEGER
            ),
            (
                cellprofiler.measurement.IMAGE,
                cellprofiler.measurement.FF_COUNT % "ObjectProcessing",
                cellprofiler.measurement.COLTYPE_INTEGER
            ),
            (
                "Objects",
                cellprofiler.measurement.FF_CHILDREN_COUNT % "ObjectProcessing",
                cellprofiler.measurement.COLTYPE_INTEGER
            ),
            (
                "ObjectProcessing",
                cellprofiler.measurement.FF_PARENT % "Objects",
                cellprofiler.measurement.COLTYPE_INTEGER
            ),
            (
                "additional_output",
                cellprofiler.measurement.M_LOCATION_CENTER_X,
                cellprofiler.measurement.COLTYPE_FLOAT
            ),
            (
                "additional_output",
                cellprofiler.measurement.M_LOCATION_CENTER_Y,
                cellprofiler.measurement.COLTYPE_FLOAT
            ),
            (
                "additional_output",
                cellprofiler.measurement.M_LOCATION_CENTER_Z,
                cellprofiler.measurement.COLTYPE_FLOAT
            ),
            (
                "additional_output",
                cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                cellprofiler.measurement.COLTYPE_INTEGER
            ),
            (
                cellprofiler.measurement.IMAGE,
                cellprofiler.measurement.FF_COUNT % "additional_output",
                cellprofiler.measurement.COLTYPE_INTEGER
            ),
            (
                "additional_input",
                cellprofiler.measurement.FF_CHILDREN_COUNT % "additional_output",
                cellprofiler.measurement.COLTYPE_INTEGER
            ),
            (
                "additional_output",
                cellprofiler.measurement.FF_PARENT % "additional_input",
                cellprofiler.measurement.COLTYPE_INTEGER
            )
        ]

        assert actual == expected

    def test_get_measurements_image_count(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, cellprofiler.measurement.IMAGE, cellprofiler.measurement.C_COUNT)

        expected = [
            "ObjectProcessing"
        ]

        assert actual == expected

    def test_get_measurements_image_other(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, cellprofiler.measurement.IMAGE, "foo")

        expected = []

        assert actual == expected

    def test_get_measurements_input_object_children(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, "Objects", cellprofiler.measurement.C_CHILDREN)

        expected = [
            cellprofiler.measurement.FF_COUNT % "ObjectProcessing"
        ]

        assert actual == expected

    def test_get_measurements_input_object_other(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, "Objects", "foo")

        expected = []

        assert actual == expected

    def test_get_measurements_output_object_location(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, "ObjectProcessing", cellprofiler.measurement.C_LOCATION)

        expected = [
            cellprofiler.measurement.FTR_CENTER_X,
            cellprofiler.measurement.FTR_CENTER_Y,
            cellprofiler.measurement.FTR_CENTER_Z
        ]

        assert actual == expected

    def test_get_measurements_output_object_number(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, "ObjectProcessing", cellprofiler.measurement.C_NUMBER)

        expected = [
            cellprofiler.measurement.FTR_OBJECT_NUMBER
        ]

        assert actual == expected

    def test_get_measurements_output_object_parent(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, "ObjectProcessing", cellprofiler.measurement.C_PARENT)

        expected = [
            "Objects"
        ]

        assert actual == expected

    def test_get_measurements_output_object_other(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, "ObjectProcessing", "foo")

        expected = []

        assert actual == expected

    def test_get_measurements_other_other(self):
        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, "foo", "bar")

        expected = []

        assert actual == expected

    def test_add_measurements(self):
        measurements = cellprofiler.measurement.Measurements()

        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        object_set = cellprofiler.object.ObjectSet()

        labels = numpy.zeros((30, 30), dtype=numpy.uint8)

        i, j = numpy.mgrid[-15:15, -7:23]
        labels[i ** 2 + j ** 2 <= 25] = 1

        i, j = numpy.mgrid[-15:15, -22:8]
        labels[i ** 2 + j ** 2 <= 16] = 2

        objects = cellprofiler.object.Objects()

        objects.segmented = labels

        object_set.add_objects(objects, "ObjectProcessing")

        parent_labels = numpy.zeros((30, 30), dtype=numpy.uint8)

        i, j = numpy.mgrid[-15:15, -15:15]
        parent_labels[i ** 2 + j ** 2 <= 196] = 1

        parent_objects = cellprofiler.object.Objects()

        parent_objects.segmented = parent_labels

        object_set.add_objects(parent_objects, "Objects")

        workspace = cellprofiler.workspace.Workspace(
            pipeline=None,
            module=module,
            image_set=None,
            object_set=object_set,
            measurements=measurements,
            image_set_list=None
        )

        module.add_measurements(workspace)

        expected_center_x = [7.0, 22.0]

        actual_center_x = measurements.get_measurement(
            "ObjectProcessing",
            cellprofiler.measurement.M_LOCATION_CENTER_X
        )

        numpy.testing.assert_array_equal(actual_center_x, expected_center_x)

        expected_center_y = [15.0, 15.0]

        actual_center_y = measurements.get_measurement(
            "ObjectProcessing",
            cellprofiler.measurement.M_LOCATION_CENTER_Y
        )

        numpy.testing.assert_array_equal(actual_center_y, expected_center_y)

        expected_center_z = [0.0, 0.0]

        actual_center_z = measurements.get_measurement(
            "ObjectProcessing",
            cellprofiler.measurement.M_LOCATION_CENTER_Z
        )

        numpy.testing.assert_array_equal(actual_center_z, expected_center_z)

        expected_object_number = [1.0, 2.0]

        actual_object_number = measurements.get_measurement(
            "ObjectProcessing",
            cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER
        )

        numpy.testing.assert_array_equal(actual_object_number, expected_object_number)

        expected_count = [2.0]

        actual_count = measurements.get_measurement(
            cellprofiler.measurement.IMAGE,
            cellprofiler.measurement.FF_COUNT % "ObjectProcessing"
        )

        numpy.testing.assert_array_equal(actual_count, expected_count)

        expected_children_count = [2]

        actual_children_count = measurements.get_measurement(
            "Objects",
            cellprofiler.measurement.FF_CHILDREN_COUNT % "ObjectProcessing"
        )

        numpy.testing.assert_array_equal(actual_children_count, expected_children_count)

        expected_parents = [1, 1]

        actual_parents = measurements.get_measurement(
            "ObjectProcessing",
            cellprofiler.measurement.FF_PARENT % "Objects"
        )

        numpy.testing.assert_array_equal(actual_parents, expected_parents)

    def test_run(self):
        measurements = cellprofiler.measurement.Measurements()

        module = cellprofiler.module.ObjectProcessing()

        module.x_name.value = "Objects"

        object_set = cellprofiler.object.ObjectSet()

        parent_objects = cellprofiler.object.Objects()

        parent_objects.segmented = numpy.ones((10, 10), dtype=numpy.uint8)

        object_set.add_objects(parent_objects, "Objects")

        workspace = cellprofiler.workspace.Workspace(
            pipeline=None,
            module=module,
            image_set=None,
            object_set=object_set,
            measurements=measurements,
            image_set_list=None
        )

        module.function = lambda x: x

        module.run(workspace)

        assert measurements.has_feature(
            "ObjectProcessing",
            cellprofiler.measurement.M_LOCATION_CENTER_X
        )

        assert measurements.has_feature(
            "ObjectProcessing",
            cellprofiler.measurement.M_LOCATION_CENTER_Y
        )

        assert measurements.has_feature(
            "ObjectProcessing",
            cellprofiler.measurement.M_LOCATION_CENTER_Z
        )

        assert measurements.has_feature(
            "ObjectProcessing",
            cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER
        )

        assert measurements.has_feature(
            cellprofiler.measurement.IMAGE,
            cellprofiler.measurement.FF_COUNT % "ObjectProcessing"
        )

        assert measurements.has_feature(
            "Objects",
            cellprofiler.measurement.FF_CHILDREN_COUNT % "ObjectProcessing"
        )

        assert measurements.has_feature(
            "ObjectProcessing",
            cellprofiler.measurement.FF_PARENT % "Objects"
        )


class TestImageSegmentation():
    def test_get_categories_image(self):
        module = cellprofiler.module.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_categories(None, cellprofiler.measurement.IMAGE)

        expected = [
            cellprofiler.measurement.C_COUNT
        ]

        assert actual == expected

    def test_get_categories_output_object(self):
        module = cellprofiler.module.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_categories(None, "ImageSegmentation")

        expected = [
            cellprofiler.measurement.C_LOCATION,
            cellprofiler.measurement.C_NUMBER
        ]

        assert actual == expected

    def test_get_categories_other(self):
        module = cellprofiler.module.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_categories(None, "foo")

        expected = []

        assert actual == expected

    def test_get_measurement_columns(self):
        module = cellprofiler.module.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurement_columns(None)

        expected = [
            (
                "ImageSegmentation",
                cellprofiler.measurement.M_LOCATION_CENTER_X,
                cellprofiler.measurement.COLTYPE_FLOAT
            ),
            (
                "ImageSegmentation",
                cellprofiler.measurement.M_LOCATION_CENTER_Y,
                cellprofiler.measurement.COLTYPE_FLOAT
            ),
            (
                "ImageSegmentation",
                cellprofiler.measurement.M_LOCATION_CENTER_Z,
                cellprofiler.measurement.COLTYPE_FLOAT
            ),
            (
                "ImageSegmentation",
                cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                cellprofiler.measurement.COLTYPE_INTEGER
            ),
            (
                cellprofiler.measurement.IMAGE,
                cellprofiler.measurement.FF_COUNT % "ImageSegmentation",
                cellprofiler.measurement.COLTYPE_INTEGER
            )
        ]

        assert actual == expected

    def test_get_measurements_image_count(self):
        module = cellprofiler.module.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurements(None, cellprofiler.measurement.IMAGE, cellprofiler.measurement.C_COUNT)

        expected = [
            "ImageSegmentation"
        ]

        assert actual == expected

    def test_get_measurements_image_other(self):
        module = cellprofiler.module.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurements(None, cellprofiler.measurement.IMAGE, "foo")

        expected = []

        assert actual == expected

    def test_get_measurements_output_object_location(self):
        module = cellprofiler.module.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurements(None, "ImageSegmentation", cellprofiler.measurement.C_LOCATION)

        expected = [
            cellprofiler.measurement.FTR_CENTER_X,
            cellprofiler.measurement.FTR_CENTER_Y,
            cellprofiler.measurement.FTR_CENTER_Z
        ]

        assert actual == expected

    def test_get_measurements_output_object_number(self):
        module = cellprofiler.module.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurements(None, "ImageSegmentation", cellprofiler.measurement.C_NUMBER)

        expected = [
            cellprofiler.measurement.FTR_OBJECT_NUMBER
        ]

        assert actual == expected

    def test_get_measurements_output_object_other(self):
        module = cellprofiler.module.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurements(None, "ImageSegmentation", "foo")

        expected = []

        assert actual == expected

    def test_get_measurements_other_other(self):
        module = cellprofiler.module.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurements(None, "foo", "bar")

        expected = []

        assert actual == expected

    def test_add_measurements(self):
        measurements = cellprofiler.measurement.Measurements()

        module = cellprofiler.module.ImageSegmentation()

        module.x_name.value = "Image"

        image = cellprofiler.image.Image(image=numpy.zeros((30, 30)))

        image_set_list = cellprofiler.image.ImageSetList()

        image_set = image_set_list.get_image_set(0)

        image_set.add("Image", image)

        object_set = cellprofiler.object.ObjectSet()

        labels = numpy.zeros((30, 30), dtype=numpy.uint8)

        i, j = numpy.mgrid[-15:15, -7:23]
        labels[i ** 2 + j ** 2 <= 25] = 1

        i, j = numpy.mgrid[-15:15, -22:8]
        labels[i ** 2 + j ** 2 <= 16] = 2

        objects = cellprofiler.object.Objects()

        objects.segmented = labels

        object_set.add_objects(objects, "ImageSegmentation")

        workspace = cellprofiler.workspace.Workspace(
            pipeline=None,
            module=module,
            image_set=image_set,
            object_set=object_set,
            measurements=measurements,
            image_set_list=image_set_list
        )

        module.add_measurements(workspace)

        expected_center_x = [7.0, 22.0]

        actual_center_x = measurements.get_measurement(
            "ImageSegmentation",
            cellprofiler.measurement.M_LOCATION_CENTER_X
        )

        numpy.testing.assert_array_equal(actual_center_x, expected_center_x)

        expected_center_y = [15.0, 15.0]

        actual_center_y = measurements.get_measurement(
            "ImageSegmentation",
            cellprofiler.measurement.M_LOCATION_CENTER_Y
        )

        numpy.testing.assert_array_equal(actual_center_y, expected_center_y)

        expected_center_z = [0.0, 0.0]

        actual_center_z = measurements.get_measurement(
            "ImageSegmentation",
            cellprofiler.measurement.M_LOCATION_CENTER_Z
        )

        numpy.testing.assert_array_equal(actual_center_z, expected_center_z)

        expected_object_number = [1.0, 2.0]

        actual_object_number = measurements.get_measurement(
            "ImageSegmentation",
            cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER
        )

        numpy.testing.assert_array_equal(actual_object_number, expected_object_number)

        expected_count = [2.0]

        actual_count = measurements.get_measurement(
            cellprofiler.measurement.IMAGE,
            cellprofiler.measurement.FF_COUNT % "ImageSegmentation"
        )

        numpy.testing.assert_array_equal(actual_count, expected_count)

    def test_run(self):
        measurements = cellprofiler.measurement.Measurements()

        module = cellprofiler.module.ImageSegmentation()

        module.x_name.value = "Image"

        image = cellprofiler.image.Image(image=numpy.zeros((30, 30)))

        image_set_list = cellprofiler.image.ImageSetList()

        image_set = image_set_list.get_image_set(0)

        image_set.add("Image", image)

        object_set = cellprofiler.object.ObjectSet()

        workspace = cellprofiler.workspace.Workspace(
            pipeline=None,
            module=module,
            image_set=image_set,
            object_set=object_set,
            measurements=measurements,
            image_set_list=image_set_list
        )

        module.function = lambda x: numpy.ones((30, 30), dtype=numpy.uint8)

        module.run(workspace)

        assert measurements.has_feature(
            "ImageSegmentation",
            cellprofiler.measurement.M_LOCATION_CENTER_X
        )

        assert measurements.has_feature(
            "ImageSegmentation",
            cellprofiler.measurement.M_LOCATION_CENTER_Y
        )

        assert measurements.has_feature(
            "ImageSegmentation",
            cellprofiler.measurement.M_LOCATION_CENTER_Z
        )

        assert measurements.has_feature(
            "ImageSegmentation",
            cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER
        )

        assert measurements.has_feature(
            cellprofiler.measurement.IMAGE,
            cellprofiler.measurement.FF_COUNT % "ImageSegmentation"
        )
