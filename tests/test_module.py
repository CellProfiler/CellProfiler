import cellprofiler.measurement
import cellprofiler.module


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
                cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
                cellprofiler.measurement.COLTYPE_INTEGER
            ),
            (
                cellprofiler.measurement.IMAGE,
                "ObjectProcessing",
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
            cellprofiler.measurement.FTR_CENTER_Y
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
