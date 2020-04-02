import numpy

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.module
import cellprofiler.object
import cellprofiler.workspace


class TestObjectProcessing:
    def test_get_categories_image(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_categories(None, cellprofiler_core.measurement.IMAGE)

        expected = [cellprofiler_core.measurement.C_COUNT]

        assert actual == expected

    def test_get_categories_input_object(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_categories(None, "Objects")

        expected = [cellprofiler_core.measurement.C_CHILDREN]

        assert actual == expected

    def test_get_categories_output_object(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_categories(None, "ObjectProcessing")

        expected = [
            cellprofiler_core.measurement.C_LOCATION,
            cellprofiler_core.measurement.C_NUMBER,
            cellprofiler_core.measurement.C_PARENT,
        ]

        assert actual == expected

    def test_get_categories_other(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_categories(None, "foo")

        expected = []

        assert actual == expected

    def test_get_measurement_columns(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurement_columns(None)

        expected = [
            (
                "ObjectProcessing",
                cellprofiler_core.measurement.M_LOCATION_CENTER_X,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.measurement.M_LOCATION_CENTER_Y,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.measurement.M_LOCATION_CENTER_Z,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.measurement.M_NUMBER_OBJECT_NUMBER,
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
            (
                cellprofiler_core.measurement.IMAGE,
                cellprofiler_core.measurement.FF_COUNT % "ObjectProcessing",
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
            (
                "Objects",
                cellprofiler_core.measurement.FF_CHILDREN_COUNT % "ObjectProcessing",
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.measurement.FF_PARENT % "Objects",
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
        ]

        assert actual == expected

    def test_get_measurement_columns_additional_objects(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurement_columns(
            None, additional_objects=[("additional_input", "additional_output")]
        )

        expected = [
            (
                "ObjectProcessing",
                cellprofiler_core.measurement.M_LOCATION_CENTER_X,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.measurement.M_LOCATION_CENTER_Y,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.measurement.M_LOCATION_CENTER_Z,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.measurement.M_NUMBER_OBJECT_NUMBER,
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
            (
                cellprofiler_core.measurement.IMAGE,
                cellprofiler_core.measurement.FF_COUNT % "ObjectProcessing",
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
            (
                "Objects",
                cellprofiler_core.measurement.FF_CHILDREN_COUNT % "ObjectProcessing",
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
            (
                "ObjectProcessing",
                cellprofiler_core.measurement.FF_PARENT % "Objects",
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
            (
                "additional_output",
                cellprofiler_core.measurement.M_LOCATION_CENTER_X,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                "additional_output",
                cellprofiler_core.measurement.M_LOCATION_CENTER_Y,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                "additional_output",
                cellprofiler_core.measurement.M_LOCATION_CENTER_Z,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                "additional_output",
                cellprofiler_core.measurement.M_NUMBER_OBJECT_NUMBER,
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
            (
                cellprofiler_core.measurement.IMAGE,
                cellprofiler_core.measurement.FF_COUNT % "additional_output",
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
            (
                "additional_input",
                cellprofiler_core.measurement.FF_CHILDREN_COUNT % "additional_output",
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
            (
                "additional_output",
                cellprofiler_core.measurement.FF_PARENT % "additional_input",
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
        ]

        assert actual == expected

    def test_get_measurements_image_count(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(
            None, cellprofiler_core.measurement.IMAGE, cellprofiler_core.measurement.C_COUNT
        )

        expected = ["ObjectProcessing"]

        assert actual == expected

    def test_get_measurements_image_other(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, cellprofiler_core.measurement.IMAGE, "foo")

        expected = []

        assert actual == expected

    def test_get_measurements_input_object_children(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(
            None, "Objects", cellprofiler_core.measurement.C_CHILDREN
        )

        expected = [cellprofiler_core.measurement.FF_COUNT % "ObjectProcessing"]

        assert actual == expected

    def test_get_measurements_input_object_other(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, "Objects", "foo")

        expected = []

        assert actual == expected

    def test_get_measurements_output_object_location(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(
            None, "ObjectProcessing", cellprofiler_core.measurement.C_LOCATION
        )

        expected = [
            cellprofiler_core.measurement.FTR_CENTER_X,
            cellprofiler_core.measurement.FTR_CENTER_Y,
            cellprofiler_core.measurement.FTR_CENTER_Z,
        ]

        assert actual == expected

    def test_get_measurements_output_object_number(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(
            None, "ObjectProcessing", cellprofiler_core.measurement.C_NUMBER
        )

        expected = [cellprofiler_core.measurement.FTR_OBJECT_NUMBER]

        assert actual == expected

    def test_get_measurements_output_object_parent(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(
            None, "ObjectProcessing", cellprofiler_core.measurement.C_PARENT
        )

        expected = ["Objects"]

        assert actual == expected

    def test_get_measurements_output_object_other(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, "ObjectProcessing", "foo")

        expected = []

        assert actual == expected

    def test_get_measurements_other_other(self):
        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

        module.x_name.value = "Objects"

        actual = module.get_measurements(None, "foo", "bar")

        expected = []

        assert actual == expected

    def test_add_measurements(self):
        measurements = cellprofiler_core.measurement.Measurements()

        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

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
            image_set_list=None,
        )

        module.add_measurements(workspace)

        expected_center_x = [7.0, 22.0]

        actual_center_x = measurements.get_measurement(
            "ObjectProcessing", cellprofiler_core.measurement.M_LOCATION_CENTER_X
        )

        numpy.testing.assert_array_equal(actual_center_x, expected_center_x)

        expected_center_y = [15.0, 15.0]

        actual_center_y = measurements.get_measurement(
            "ObjectProcessing", cellprofiler_core.measurement.M_LOCATION_CENTER_Y
        )

        numpy.testing.assert_array_equal(actual_center_y, expected_center_y)

        expected_center_z = [0.0, 0.0]

        actual_center_z = measurements.get_measurement(
            "ObjectProcessing", cellprofiler_core.measurement.M_LOCATION_CENTER_Z
        )

        numpy.testing.assert_array_equal(actual_center_z, expected_center_z)

        expected_object_number = [1.0, 2.0]

        actual_object_number = measurements.get_measurement(
            "ObjectProcessing", cellprofiler_core.measurement.M_NUMBER_OBJECT_NUMBER
        )

        numpy.testing.assert_array_equal(actual_object_number, expected_object_number)

        expected_count = [2.0]

        actual_count = measurements.get_measurement(
            cellprofiler_core.measurement.IMAGE,
            cellprofiler_core.measurement.FF_COUNT % "ObjectProcessing",
        )

        numpy.testing.assert_array_equal(actual_count, expected_count)

        expected_children_count = [2]

        actual_children_count = measurements.get_measurement(
            "Objects", cellprofiler_core.measurement.FF_CHILDREN_COUNT % "ObjectProcessing"
        )

        numpy.testing.assert_array_equal(actual_children_count, expected_children_count)

        expected_parents = [1, 1]

        actual_parents = measurements.get_measurement(
            "ObjectProcessing", cellprofiler_core.measurement.FF_PARENT % "Objects"
        )

        numpy.testing.assert_array_equal(actual_parents, expected_parents)

    def test_run(self):
        measurements = cellprofiler_core.measurement.Measurements()

        module = cellprofiler_core.module.image_segmentation.ObjectProcessing()

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
            image_set_list=None,
        )

        module.function = lambda x: x

        module.run(workspace)

        assert measurements.has_feature(
            "ObjectProcessing", cellprofiler_core.measurement.M_LOCATION_CENTER_X
        )

        assert measurements.has_feature(
            "ObjectProcessing", cellprofiler_core.measurement.M_LOCATION_CENTER_Y
        )

        assert measurements.has_feature(
            "ObjectProcessing", cellprofiler_core.measurement.M_LOCATION_CENTER_Z
        )

        assert measurements.has_feature(
            "ObjectProcessing", cellprofiler_core.measurement.M_NUMBER_OBJECT_NUMBER
        )

        assert measurements.has_feature(
            cellprofiler_core.measurement.IMAGE,
            cellprofiler_core.measurement.FF_COUNT % "ObjectProcessing",
        )

        assert measurements.has_feature(
            "Objects", cellprofiler_core.measurement.FF_CHILDREN_COUNT % "ObjectProcessing"
        )

        assert measurements.has_feature(
            "ObjectProcessing", cellprofiler_core.measurement.FF_PARENT % "Objects"
        )


class TestImageSegmentation:
    def test_get_categories_image(self):
        module = cellprofiler_core.module.image_segmentation.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_categories(None, cellprofiler_core.measurement.IMAGE)

        expected = [cellprofiler_core.measurement.C_COUNT]

        assert actual == expected

    def test_get_categories_output_object(self):
        module = cellprofiler_core.module.image_segmentation.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_categories(None, "ImageSegmentation")

        expected = [
            cellprofiler_core.measurement.C_LOCATION,
            cellprofiler_core.measurement.C_NUMBER,
        ]

        assert actual == expected

    def test_get_categories_other(self):
        module = cellprofiler_core.module.image_segmentation.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_categories(None, "foo")

        expected = []

        assert actual == expected

    def test_get_measurement_columns(self):
        module = cellprofiler_core.module.image_segmentation.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurement_columns(None)

        expected = [
            (
                "ImageSegmentation",
                cellprofiler_core.measurement.M_LOCATION_CENTER_X,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                "ImageSegmentation",
                cellprofiler_core.measurement.M_LOCATION_CENTER_Y,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                "ImageSegmentation",
                cellprofiler_core.measurement.M_LOCATION_CENTER_Z,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                "ImageSegmentation",
                cellprofiler_core.measurement.M_NUMBER_OBJECT_NUMBER,
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
            (
                cellprofiler_core.measurement.IMAGE,
                cellprofiler_core.measurement.FF_COUNT % "ImageSegmentation",
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
        ]

        assert actual == expected

    def test_get_measurements_image_count(self):
        module = cellprofiler_core.module.image_segmentation.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurements(
            None, cellprofiler_core.measurement.IMAGE, cellprofiler_core.measurement.C_COUNT
        )

        expected = ["ImageSegmentation"]

        assert actual == expected

    def test_get_measurements_image_other(self):
        module = cellprofiler_core.module.image_segmentation.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurements(None, cellprofiler_core.measurement.IMAGE, "foo")

        expected = []

        assert actual == expected

    def test_get_measurements_output_object_location(self):
        module = cellprofiler_core.module.image_segmentation.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurements(
            None, "ImageSegmentation", cellprofiler_core.measurement.C_LOCATION
        )

        expected = [
            cellprofiler_core.measurement.FTR_CENTER_X,
            cellprofiler_core.measurement.FTR_CENTER_Y,
            cellprofiler_core.measurement.FTR_CENTER_Z,
        ]

        assert actual == expected

    def test_get_measurements_output_object_number(self):
        module = cellprofiler_core.module.image_segmentation.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurements(
            None, "ImageSegmentation", cellprofiler_core.measurement.C_NUMBER
        )

        expected = [cellprofiler_core.measurement.FTR_OBJECT_NUMBER]

        assert actual == expected

    def test_get_measurements_output_object_other(self):
        module = cellprofiler_core.module.image_segmentation.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurements(None, "ImageSegmentation", "foo")

        expected = []

        assert actual == expected

    def test_get_measurements_other_other(self):
        module = cellprofiler_core.module.image_segmentation.ImageSegmentation()

        module.x_name.value = "Image"

        actual = module.get_measurements(None, "foo", "bar")

        expected = []

        assert actual == expected

    def test_add_measurements(self):
        measurements = cellprofiler_core.measurement.Measurements()

        module = cellprofiler_core.module.image_segmentation.ImageSegmentation()

        module.x_name.value = "Image"

        image = cellprofiler_core.image.Image(image=numpy.zeros((30, 30)))

        image_set_list = cellprofiler_core.image.ImageSetList()

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
            image_set_list=image_set_list,
        )

        module.add_measurements(workspace)

        expected_center_x = [7.0, 22.0]

        actual_center_x = measurements.get_measurement(
            "ImageSegmentation", cellprofiler_core.measurement.M_LOCATION_CENTER_X
        )

        numpy.testing.assert_array_equal(actual_center_x, expected_center_x)

        expected_center_y = [15.0, 15.0]

        actual_center_y = measurements.get_measurement(
            "ImageSegmentation", cellprofiler_core.measurement.M_LOCATION_CENTER_Y
        )

        numpy.testing.assert_array_equal(actual_center_y, expected_center_y)

        expected_center_z = [0.0, 0.0]

        actual_center_z = measurements.get_measurement(
            "ImageSegmentation", cellprofiler_core.measurement.M_LOCATION_CENTER_Z
        )

        numpy.testing.assert_array_equal(actual_center_z, expected_center_z)

        expected_object_number = [1.0, 2.0]

        actual_object_number = measurements.get_measurement(
            "ImageSegmentation", cellprofiler_core.measurement.M_NUMBER_OBJECT_NUMBER
        )

        numpy.testing.assert_array_equal(actual_object_number, expected_object_number)

        expected_count = [2.0]

        actual_count = measurements.get_measurement(
            cellprofiler_core.measurement.IMAGE,
            cellprofiler_core.measurement.FF_COUNT % "ImageSegmentation",
        )

        numpy.testing.assert_array_equal(actual_count, expected_count)

    def test_run(self):
        measurements = cellprofiler_core.measurement.Measurements()

        module = cellprofiler_core.module.image_segmentation.ImageSegmentation()

        module.x_name.value = "Image"

        image = cellprofiler_core.image.Image(image=numpy.zeros((30, 30)))

        image_set_list = cellprofiler_core.image.ImageSetList()

        image_set = image_set_list.get_image_set(0)

        image_set.add("Image", image)

        object_set = cellprofiler.object.ObjectSet()

        workspace = cellprofiler.workspace.Workspace(
            pipeline=None,
            module=module,
            image_set=image_set,
            object_set=object_set,
            measurements=measurements,
            image_set_list=image_set_list,
        )

        module.function = lambda x: numpy.ones((30, 30), dtype=numpy.uint8)

        module.run(workspace)

        assert measurements.has_feature(
            "ImageSegmentation", cellprofiler_core.measurement.M_LOCATION_CENTER_X
        )

        assert measurements.has_feature(
            "ImageSegmentation", cellprofiler_core.measurement.M_LOCATION_CENTER_Y
        )

        assert measurements.has_feature(
            "ImageSegmentation", cellprofiler_core.measurement.M_LOCATION_CENTER_Z
        )

        assert measurements.has_feature(
            "ImageSegmentation", cellprofiler_core.measurement.M_NUMBER_OBJECT_NUMBER
        )

        assert measurements.has_feature(
            cellprofiler_core.measurement.IMAGE,
            cellprofiler_core.measurement.FF_COUNT % "ImageSegmentation",
        )
