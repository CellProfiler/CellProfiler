import hashlib
import io
import os
import re
import tempfile

import bioformats
import bioformats.formatreader
import numpy
import six

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.loaddata
import cellprofiler.modules.loadimages
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.setting
import cellprofiler.workspace
import tests.modules

cellprofiler.preferences.set_headless()

OBJECTS_NAME = "objects"

tests.modules.maybe_download_sbs()

test_folder = "loaddata"

test_path = os.path.join(tests.modules.example_images_directory(), test_folder)

test_filename = "image.tif"

test_shape = (13, 15)

path = tests.modules.maybe_download_example_image(
    [test_folder], test_filename, shape=test_shape
)

with open(path, "rb") as fd:
    test_md5 = hashlib.md5(fd.read()).hexdigest()


def make_pipeline(csv_text, name=None):
    if name is None:
        handle, name = tempfile.mkstemp(".csv")
        fd = os.fdopen(handle, "w")
    else:
        fd = open(name, "w")
    fd.write(csv_text)
    fd.close()
    csv_path, csv_file = os.path.split(name)
    module = cellprofiler.modules.loaddata.LoadText()
    module.csv_directory.dir_choice = cellprofiler.setting.ABSOLUTE_FOLDER_NAME
    module.csv_directory.custom_path = csv_path
    module.csv_file_name.value = csv_file
    module.set_module_num(1)
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.add_module(module)

    def error_callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.RunExceptionEvent)

    pipeline.add_listener(error_callback)
    return pipeline, module, name


def test_revision():
    """Remember to update this and write another test on new revision"""
    assert cellprofiler.modules.loaddata.LoadData().variable_revision_number == 6


def test_load_v4():
    with open("./tests/resources/modules/loaddata/v4.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.loaddata.LoadData)
    assert module.csv_file_name == "1049_Metadata.csv"
    assert (
        module.csv_directory.dir_choice
        == cellprofiler.setting.DEFAULT_INPUT_FOLDER_NAME
    )
    assert module.wants_images
    assert module.rescale
    assert not module.wants_image_groupings
    assert not module.wants_rows
    assert module.row_range.min == 10
    assert module.row_range.max == 36
    assert len(module.metadata_fields.selections) == 1
    assert module.metadata_fields.selections[0] == "Well"


def test_load_v5():
    with open("./tests/resources/modules/loaddata/v5.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.loaddata.LoadData)
    assert module.csv_file_name == "1049_Metadata.csv"
    assert module.csv_directory.dir_choice == cellprofiler.setting.ABSOLUTE_FOLDER_NAME
    assert (
        module.csv_directory.custom_path
        == r"x:\projects\NightlyBuild\trunk\ExampleImages\ExampleSBSImages"
    )
    assert module.wants_images
    assert (
        module.image_directory.dir_choice
        == cellprofiler.setting.DEFAULT_INPUT_FOLDER_NAME
    )
    assert module.rescale
    assert module.wants_image_groupings
    assert not module.wants_rows
    assert module.row_range.min == 1
    assert module.row_range.max == 100000
    assert len(module.metadata_fields.selections) == 2
    assert module.metadata_fields.selections[0] == "Column"
    assert module.metadata_fields.selections[1] == "Row"


def test_load_v6():
    with open("./tests/resources/modules/loaddata/v6.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.loaddata.LoadData)
    assert module.csv_file_name == "1049_Metadata.csv"
    assert module.csv_directory.dir_choice == cellprofiler.setting.ABSOLUTE_FOLDER_NAME
    assert (
        module.csv_directory.custom_path
        == r"x:\projects\NightlyBuild\trunk\ExampleImages\ExampleSBSImages"
    )
    assert module.wants_images
    assert (
        module.image_directory.dir_choice
        == cellprofiler.setting.DEFAULT_INPUT_FOLDER_NAME
    )
    assert module.rescale
    assert module.wants_image_groupings
    assert not module.wants_rows
    assert module.row_range.min == 1
    assert module.row_range.max == 100000
    assert len(module.metadata_fields.selections) == 2
    assert module.metadata_fields.selections[0] == "Column"
    assert module.metadata_fields.selections[1] == "Row"


def test_string_image_measurement():
    csv_text = """"Test_Measurement"
"Hello, world"
"""
    pipeline, module, filename = make_pipeline(csv_text)
    m = pipeline.run()
    data = m.get_current_image_measurement("Test_Measurement")
    assert not numpy.isreal(data)
    assert data == "Hello, world"
    os.remove(filename)


def test_float_image_measurement():
    csv_text = """"Test_Measurement"
1.5
"""
    pipeline, module, filename = make_pipeline(csv_text)
    m = pipeline.run()
    data = m.get_current_image_measurement("Test_Measurement")
    assert numpy.isreal(data)
    assert round(abs(data - 1.5), 7) == 0
    os.remove(filename)


def test_int_image_measurement():
    csv_text = """"Test_Measurement"
1
"""
    pipeline, module, filename = make_pipeline(csv_text)
    m = pipeline.run()
    data = m.get_current_image_measurement("Test_Measurement")
    assert isinstance(data, numpy.int), "data is type %s, not np.int" % (type(data))
    assert data == 1
    os.remove(filename)


def test_long_int_image_measurement():
    csv_text = """"Test_Measurement"
1234567890123
"""
    pipeline, module, filename = make_pipeline(csv_text)
    m = pipeline.run()
    data = m.get_current_image_measurement("Test_Measurement")
    assert isinstance(
        data, six.text_type
    ), "Expected <type 'six.text_type'> got %s" % type(data)
    assert data == "1234567890123"
    os.remove(filename)


def test_metadata():
    csv_text = """"Metadata_Plate"
"P-12345"
"""
    pipeline, module, filename = make_pipeline(csv_text)
    m = pipeline.run()
    data = m.get_current_image_measurement("Metadata_Plate")
    assert data == "P-12345"
    os.remove(filename)


def test_metadata_row_and_column():
    csv_text = """"Metadata_Row","Metadata_Column"
"C","03"
"""
    pipeline, module, filename = make_pipeline(csv_text)
    columns = module.get_measurement_columns(pipeline)
    assert any(
        [
            c[0] == cellprofiler.measurement.IMAGE
            and c[1] == "Metadata_Row"
            and c[2] == "varchar(1)"
            for c in columns
        ]
    )
    assert any(
        [
            c[0] == cellprofiler.measurement.IMAGE
            and c[1] == "Metadata_Column"
            and c[2] == "varchar(2)"
            for c in columns
        ]
    )
    assert any(
        [
            c[0] == cellprofiler.measurement.IMAGE
            and c[1] == "Metadata_Well"
            and c[2] == "varchar(3)"
            for c in columns
        ]
    )
    m = pipeline.run()
    features = module.get_measurements(
        pipeline, cellprofiler.measurement.IMAGE, cellprofiler.measurement.C_METADATA
    )
    for feature, expected in (("Row", "C"), ("Column", "03"), ("Well", "C03")):
        assert feature in features
        value = m.get_current_image_measurement(
            "_".join((cellprofiler.measurement.C_METADATA, feature))
        )
        assert value == expected


def test_metadata_row_and_column_and_well():
    csv_text = """"Metadata_Row","Metadata_Column","Metadata_Well"
"C","03","B14"
"""
    pipeline, module, filename = make_pipeline(csv_text)
    columns = module.get_measurement_columns(pipeline)
    assert any(
        [
            c[0] == cellprofiler.measurement.IMAGE
            and c[1] == "Metadata_Row"
            and c[2] == "varchar(1)"
            for c in columns
        ]
    )
    assert any(
        [
            c[0] == cellprofiler.measurement.IMAGE
            and c[1] == "Metadata_Column"
            and c[2] == "varchar(2)"
            for c in columns
        ]
    )
    assert any(
        [
            c[0] == cellprofiler.measurement.IMAGE
            and c[1] == "Metadata_Well"
            and c[2] == "varchar(3)"
            for c in columns
        ]
    )
    m = pipeline.run()
    features = module.get_measurements(
        pipeline, cellprofiler.measurement.IMAGE, cellprofiler.measurement.C_METADATA
    )
    for feature, expected in (("Row", "C"), ("Column", "03"), ("Well", "B14")):
        assert feature in features
        value = m.get_current_image_measurement(
            "_".join((cellprofiler.measurement.C_METADATA, feature))
        )
        assert value == expected


def test_load_file():
    csv_text = """"Image_FileName_DNA","Image_PathName_DNA"
"%s","%s"
""" % (
        test_filename,
        test_path,
    )
    pipeline, module, filename = make_pipeline(csv_text)
    c0_ran = [False]

    def callback(workspace):
        imgset = workspace.image_set
        image = imgset.get_image("DNA")
        pixels = image.pixel_data
        assert pixels.shape[0] == test_shape[0]
        c0_ran[0] = True

    c0 = C0()
    c0.callback = callback
    c0.set_module_num(2)
    pipeline.add_module(c0)

    try:
        m = pipeline.run()
        assert isinstance(m, cellprofiler.measurement.Measurements)
        assert c0_ran[0]
        hexdigest = m.get_current_image_measurement("MD5Digest_DNA")
        assert hexdigest == test_md5
        assert "PathName_DNA" in m.get_feature_names(cellprofiler.measurement.IMAGE)
        assert m.get_current_image_measurement("PathName_DNA") == test_path
        assert "FileName_DNA" in m.get_feature_names(cellprofiler.measurement.IMAGE)
        assert m.get_current_image_measurement("FileName_DNA") == test_filename
    finally:
        os.remove(filename)


def test_dont_load_file():
    csv_text = """"Image_FileName_DNA","Image_PathName_DNA"
"%s","%s"
""" % (
        test_filename,
        test_path,
    )
    pipeline, module, filename = make_pipeline(csv_text)
    c0_ran = [False]

    def callback(workspace):
        imgset = workspace.image_set
        assert len(imgset.names) == 0
        c0_ran[0] = True

    c0 = C0()
    c0.callback = callback
    c0.set_module_num(1)
    pipeline.add_module(c0)
    try:
        module.wants_images.value = False
        pipeline.run()
        assert c0_ran[0]
    finally:
        os.remove(filename)


def test_load_planes():
    file_name = "RLM1 SSN3 300308 008015000.flex"
    tests.modules.maybe_download_tesst_image(file_name)
    path = tests.modules.testimages_directory()
    pathname = os.path.join(path, file_name)
    url = cellprofiler.modules.loadimages.pathname2url(pathname)
    ftrs = (
        cellprofiler.measurement.C_URL,
        cellprofiler.measurement.C_SERIES,
        cellprofiler.measurement.C_FRAME,
    )
    channels = ("Channel1", "Channel2")
    header = ",".join(
        [",".join(["_".join((ftr, channel)) for ftr in ftrs]) for channel in channels]
    )

    csv_lines = [header]
    for series in range(4):
        csv_lines.append(
            ",".join(['"%s","%d","%d"' % (url, series, frame) for frame in range(2)])
        )
    csv_text = "\n".join(csv_lines)
    pipeline, module, filename = make_pipeline(csv_text)
    assert isinstance(module, cellprofiler.modules.loaddata.LoadData)
    m = cellprofiler.measurement.Measurements()
    image_set_list = cellprofiler.image.ImageSetList()
    try:
        workspace = cellprofiler.workspace.Workspace(
            pipeline, module, m, None, m, image_set_list
        )
        assert module.prepare_run(workspace)
        pixel_hashes = []
        for i in range(4):
            m.next_image_set(i + 1)
            module.run(workspace)
            chashes = []
            for channel in channels:
                pixel_data = m.get_image(channel).pixel_data
                h = hashlib.md5()
                h.update(pixel_data)
                chashes.append(h.digest())
            assert chashes[0] != chashes[1]
            for j, ph in enumerate(pixel_hashes):
                for k, phh in enumerate(ph):
                    for l, phd in enumerate(chashes):
                        assert phh != phd
            pixel_hashes.append(chashes)
    finally:
        os.remove(filename)


def test_some_rows():
    csv_text = """"Test_Measurement"
1
2
3
4
5
6
7
8
9
10
"""
    pipeline, module, filename = make_pipeline(csv_text)
    module.wants_rows.value = True
    module.row_range.min = 4
    module.row_range.max = 6
    m = pipeline.run()
    assert isinstance(m, cellprofiler.measurement.Measurements)
    data = m.get_all_measurements(cellprofiler.measurement.IMAGE, "Test_Measurement")
    assert numpy.all(data == numpy.arange(4, 7))
    os.remove(filename)


def test_img_717():
    """Regression test of img-717, column without underbar"""
    csv_text = """"Image","Test_Measurement"
"foo",1
"foo",2
"foo",3
"foo",4
"foo",5
"foo",6
"foo",7
"foo",8
"foo",9
"foo",10
"""
    pipeline, module, filename = make_pipeline(csv_text)
    module.wants_rows.value = True
    module.row_range.min = 4
    module.row_range.max = 6
    m = pipeline.run()
    assert isinstance(m, cellprofiler.measurement.Measurements)
    data = m.get_all_measurements(cellprofiler.measurement.IMAGE, "Test_Measurement")
    assert numpy.all(data == numpy.arange(4, 7))
    os.remove(filename)


def test_alternate_image_start():
    csv_text = """"Metadata_Measurement"
1
2
3
4
5
6
7
8
9
10
"""
    pipeline, module, filename = make_pipeline(csv_text)
    m = pipeline.run(image_set_start=2)
    data = m.get_all_measurements(
        cellprofiler.measurement.IMAGE, "Metadata_Measurement"
    )
    assert all([data[i - 2] == i for i in range(2, 11)])
    os.remove(filename)


def test_get_measurement_columns():
    """Test the get_measurement_columns method"""
    colnames = ("Integer_Measurement", "Float_Measurement", "String_Measurement")
    coltypes = [
        cellprofiler.measurement.COLTYPE_INTEGER,
        cellprofiler.measurement.COLTYPE_FLOAT,
        cellprofiler.measurement.COLTYPE_VARCHAR_FORMAT % 9,
    ]
    csv_text = (
        """"%s","%s","%s"
                1,1,1
                2,1.5,"Hi"
                3,1,"Hello"
                4,1.7,"Hola"
                5,1.2,"Bonjour"
                6,1.5,"Gutentag"
                7,1.1,"Hej"
                8,2.3,"Bevakasha"
                """
        % colnames
    )
    pipeline, module, filename = make_pipeline(csv_text)
    columns = module.get_measurement_columns(pipeline)
    for colname, coltype in zip(colnames, coltypes):
        assert any(
            [
                (
                    column[0] == cellprofiler.measurement.IMAGE
                    and column[1] == colname
                    and column[2] == coltype
                )
                for column in columns
            ]
        ), ("Failed to find %s" % colname)
    os.remove(filename)


def test_file_name_measurement_columns():
    """Regression test bug IMG-315

    A csv header of Image_FileName_Foo or Image_PathName_Foo should
    yield column names of FileName_Foo and PathName_Foo
    """
    colnames = ("Image_FileName_Foo", "Image_PathName_Foo")
    csv_text = (
        """"%s","%s"
                "Channel1-01.tif","/imaging/analysis/2500_01_01_Jones"
                "Channel1-02.tif","/imaging/analysis/2500_01_01_Jones"
                """
        % colnames
    )
    pipeline, module, filename = make_pipeline(csv_text)
    try:
        columns = module.get_measurement_columns(pipeline)
        assert "FileName_Foo" in [c[1] for c in columns]
        assert "PathName_Foo" in [c[1] for c in columns]
    finally:
        os.remove(filename)


def test_long_integer_column():
    """This is a regression test of IMG-644 where a 13-digit number got turned into an int"""
    colnames = ("Long_Integer_Measurement", "Float_Measurement", "String_Measurement")
    coltypes = [
        cellprofiler.measurement.COLTYPE_VARCHAR_FORMAT % 13,
        cellprofiler.measurement.COLTYPE_FLOAT,
        cellprofiler.measurement.COLTYPE_VARCHAR_FORMAT % 9,
    ]
    csv_text = (
        """"%s","%s","%s"
                1,1,1
                2,1.5,"Hi"
                3,1,"Hello"
                4,1.7,"Hola"
                5,1.2,"Bonjour"
                6,1.5,"Gutentag"
                7,1.1,"Hej"
                1234567890123,2.3,"Bevakasha"
                """
        % colnames
    )
    pipeline, module, filename = make_pipeline(csv_text)
    columns = module.get_measurement_columns(pipeline)
    fmt = "%15s %30s %20s"
    print((fmt % ("Object", "Feature", "Type")))
    for object_name, feature, coltype in columns:
        print((fmt % (object_name, feature, coltype)))
    for colname, coltype in zip(colnames, coltypes):
        assert any(
            [
                (
                    column[0] == cellprofiler.measurement.IMAGE
                    and column[1] == colname
                    and column[2] == coltype
                )
                for column in columns
            ]
        ), ("Failed to find %s" % colname)
    os.remove(filename)


def test_objects_measurement_columns():
    csv_text = """%s_%s,%s_%s
Channel1-01-A-01.tif,/imaging/analysis/trunk/ExampleImages/ExampleSBSImages
""" % (
        cellprofiler.measurement.C_OBJECTS_FILE_NAME,
        OBJECTS_NAME,
        cellprofiler.measurement.C_OBJECTS_PATH_NAME,
        OBJECTS_NAME,
    )
    pipeline, module, filename = make_pipeline(csv_text)
    columns = module.get_measurement_columns(pipeline)
    expected_columns = (
        (
            cellprofiler.measurement.IMAGE,
            cellprofiler.measurement.C_OBJECTS_URL + "_" + OBJECTS_NAME,
        ),
        (
            cellprofiler.measurement.IMAGE,
            cellprofiler.measurement.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME,
        ),
        (
            cellprofiler.measurement.IMAGE,
            cellprofiler.measurement.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME,
        ),
        (
            cellprofiler.measurement.IMAGE,
            cellprofiler.measurement.C_COUNT + "_" + OBJECTS_NAME,
        ),
        (OBJECTS_NAME, cellprofiler.measurement.M_LOCATION_CENTER_X),
        (OBJECTS_NAME, cellprofiler.measurement.M_LOCATION_CENTER_Y),
        (OBJECTS_NAME, cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER),
    )
    for column in columns:
        assert any(
            [
                True
                for object_name, feature in expected_columns
                if object_name == column[0] and feature == column[1]
            ]
        )
    for object_name, feature in expected_columns:
        assert any(
            [
                True
                for column in columns
                if object_name == column[0] and feature == column[1]
            ]
        )


def test_get_groupings():
    """Test the get_groupings method"""
    dir = os.path.join(tests.modules.example_images_directory(), "ExampleSBSImages")
    pattern = "Channel1-[0-9]{2}-(?P<ROW>[A-H])-(?P<COL>[0-9]{2})\\.tif"
    csv_text = '"Image_FileName_Cytoplasm","Image_PathName_Cytoplasm","Metadata_ROW","Metadata_COL"\n'
    for filename in os.listdir(dir):
        match = re.match(pattern, filename)
        if match:
            csv_text += '"%s","%s","%s","%s"\n' % (
                filename,
                dir,
                match.group("ROW"),
                match.group("COL"),
            )
    pipeline, module, filename = make_pipeline(csv_text)
    assert isinstance(module, cellprofiler.modules.loaddata.LoadText)
    module.wants_images.value = True
    module.wants_image_groupings.value = True
    module.metadata_fields.value = "ROW"
    image_set_list = cellprofiler.image.ImageSetList()
    measurements = cellprofiler.measurement.Measurements()
    workspace = cellprofiler.workspace.Workspace(
        pipeline, module, None, None, measurements, image_set_list
    )
    module.prepare_run(workspace)
    keys, groupings = module.get_groupings(workspace)
    assert len(keys) == 1
    assert keys[0] == "Metadata_ROW"
    assert len(groupings) == 8
    my_rows = [g[0]["Metadata_ROW"] for g in groupings]
    my_rows.sort()
    assert "".join(my_rows) == "ABCDEFGH"
    for grouping in groupings:
        row = grouping[0]["Metadata_ROW"]
        module.prepare_group(
            cellprofiler.workspace.Workspace(
                pipeline, module, None, None, measurements, image_set_list
            ),
            grouping[0],
            grouping[1],
        )
        for image_number in grouping[1]:
            image_set = image_set_list.get_image_set(image_number - 1)
            measurements.next_image_set(image_number)
            workspace = cellprofiler.workspace.Workspace(
                pipeline,
                module,
                image_set,
                cellprofiler.object.ObjectSet(),
                measurements,
                image_set_list,
            )
            module.run(workspace)
            provider = image_set.get_image_provider("Cytoplasm")
            match = re.search(pattern, provider.get_filename())
            assert match
            assert row == match.group("ROW")


def test_load_bcb_file():
    csv_text = """ELN_RUN_ID,CBIP_RUN_ID,ASSAY_PLATE_BARCODE,\
MX_PLATE_ID,ASSAY_WELL_POSITION,ASSAY_WELL_ROLE,SITE_X,SITE_Y,\
MICROSCOPE,SOURCE_DESCRIPTION,DATE_CREATED,FILE_PATH,FILE_NAME,\
CPD_PLATE_MAP_NAME,CPD_WELL_POSITION,BROAD_ID,\
CPD_MMOL_CONC,SOURCE_NAME,SOURCE_COMPOUND_NAME,CPD_SMILES
"4012-10-W01-01-02","4254","BR00021547","20777","N01","COMPOUND",\
"2","2","GS IX Micro","DAPI","2010/03/19 06:01:12","%s",\
"%s","C-4012-00-D80-001_Rev3","N01",\
"BRD-K71194192-001-01-6","2.132352941","ChemBridge","",\
"Oc1ccnc(SCC(=O)Nc2ccc(Oc3ccccc3)cc2)n1"
""" % (
        test_path,
        test_filename,
    )
    pipeline, module, filename = make_pipeline(csv_text)
    c0_ran = [False]

    def callback(workspace):
        imgset = workspace.image_set
        image = imgset.get_image("DAPI")
        pixels = image.pixel_data
        assert pixels.shape[0] == test_shape[0]
        c0_ran[0] = True

    c0 = C0()
    c0.callback = callback
    c0.set_module_num(2)
    pipeline.add_module(c0)

    try:
        m = pipeline.run()
        assert isinstance(m, cellprofiler.measurement.Measurements)
        assert c0_ran[0]
        hexdigest = m.get_current_image_measurement("MD5Digest_DAPI")
        assert hexdigest == test_md5
        assert "PathName_DAPI" in m.get_feature_names(cellprofiler.measurement.IMAGE)
        assert m.get_current_image_measurement("PathName_DAPI") == test_path
        assert "FileName_DAPI" in m.get_feature_names(cellprofiler.measurement.IMAGE)
        assert m.get_current_image_measurement("FileName_DAPI") == test_filename
    finally:
        os.remove(filename)


def test_scaling():
    """Test loading an image scaled and unscaled"""
    folder = "loaddata"
    file_name = "1-162hrh2ax2.tif"
    path = tests.modules.make_12_bit_image(folder, file_name, (22, 18))
    csv_text = (
        "Image_PathName_MyFile,Image_FileName_MyFile\n" "%s,%s\n" % os.path.split(path)
    )
    c0_image = []
    for rescale in (False, True):
        pipeline, module, filename = make_pipeline(csv_text)
        try:
            module.rescale.value = rescale

            def callback(workspace):
                imgset = workspace.image_set
                image = imgset.get_image("MyFile")
                pixels = image.pixel_data
                c0_image.append(pixels.copy())

            c0 = C0()
            c0.callback = callback
            c0.set_module_num(2)
            pipeline.add_module(c0)
            pipeline.run()
        finally:
            os.remove(filename)
    unscaled, scaled = c0_image
    numpy.testing.assert_almost_equal(unscaled * 65535.0 / 4095.0, scaled)


def test_load_objects():
    r = numpy.random.RandomState()
    r.seed(1101)
    labels = r.randint(0, 10, size=(30, 20)).astype(numpy.uint8)
    handle, name = tempfile.mkstemp(".png")
    bioformats.write_image(name, labels, bioformats.PT_UINT8)
    os.close(handle)
    png_path, png_file = os.path.split(name)
    sbs_dir = os.path.join(tests.modules.example_images_directory(), "ExampleSBSImages")
    csv_text = """%s_%s,%s_%s,%s_DNA,%s_DNA
%s,%s,Channel2-01-A-01.tif,%s
""" % (
        cellprofiler.measurement.C_OBJECTS_FILE_NAME,
        OBJECTS_NAME,
        cellprofiler.measurement.C_OBJECTS_PATH_NAME,
        OBJECTS_NAME,
        cellprofiler.measurement.C_FILE_NAME,
        cellprofiler.measurement.C_PATH_NAME,
        png_file,
        png_path,
        sbs_dir,
    )
    pipeline, module, csv_name = make_pipeline(csv_text)
    assert isinstance(pipeline, cellprofiler.pipeline.Pipeline)
    assert isinstance(module, cellprofiler.modules.loaddata.LoadData)
    module.wants_images.value = True
    try:
        image_set_list = cellprofiler.image.ImageSetList()
        measurements = cellprofiler.measurement.Measurements()
        workspace = cellprofiler.workspace.Workspace(
            pipeline, module, None, None, measurements, image_set_list
        )
        pipeline.prepare_run(workspace)
        key_names, g = pipeline.get_groupings(workspace)
        assert len(g) == 1
        module.prepare_group(workspace, g[0][0], g[0][1])
        image_set = image_set_list.get_image_set(g[0][1][0] - 1)
        object_set = cellprofiler.object.ObjectSet()
        workspace = cellprofiler.workspace.Workspace(
            pipeline, module, image_set, object_set, measurements, image_set_list
        )
        module.run(workspace)
        objects = object_set.get_objects(OBJECTS_NAME)
        assert numpy.all(objects.segmented == labels)
        assert (
            measurements.get_current_image_measurement(
                cellprofiler.measurement.FF_COUNT % OBJECTS_NAME
            )
            == 9
        )
        for feature in (
            cellprofiler.measurement.M_LOCATION_CENTER_X,
            cellprofiler.measurement.M_LOCATION_CENTER_Y,
            cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
        ):
            value = measurements.get_current_measurement(OBJECTS_NAME, feature)
            assert len(value) == 9
    finally:
        bioformats.formatreader.clear_image_reader_cache()
        os.remove(name)
        os.remove(csv_name)


# def test_load_unicode():
#     base_directory = tempfile.mkdtemp()
#     directory = u"\u2211\u03B1"
#     filename = u"\u03B2.jpg"
#     base_path = os.path.join(base_directory, directory)
#     os.mkdir(base_path)
#     path = os.path.join(base_path, filename)
#     csv_filename = u"\u03b3.csv"
#     csv_path = os.path.join(base_path, csv_filename)
#     unicode_value = u"\u03b4.csv"
#     try:
#         r = np.random.RandomState()
#         r.seed(1101)
#         labels = r.randint(0, 10, size=(30, 20)).astype(np.uint8)
#         write_image(path, labels, PT_UINT8)
#         csv_text = ("Image_FileName_MyFile,Image_PathName_MyFile,Metadata_Unicode\n"
#                     "%s,%s,%s\n" %
#                     (filename.encode('utf8'), base_path.encode('utf8'),
#                      unicode_value.encode('utf8')))
#         pipeline, module, _ = make_pipeline(csv_text, csv_path)
#         image_set_list = cpi.ImageSetList()
#         m = cpmeas.Measurements()
#         workspace = cpw.Workspace(pipeline, module, None, None,
#                                   m, image_set_list)
#         assertTrue(module.prepare_run(workspace))
#         assertEqual(len(m.get_image_numbers()), 1)
#         key_names, group_list = pipeline.get_groupings(workspace)
#         assertEqual(len(group_list), 1)
#         group_keys, image_numbers = group_list[0]
#         assertEqual(len(image_numbers), 1)
#         module.prepare_group(workspace, group_keys, image_numbers)
#         image_set = image_set_list.get_image_set(image_numbers[0] - 1)
#         workspace = cpw.Workspace(pipeline, module, image_set,
#                                   cpo.ObjectSet(), m, image_set_list)
#         module.run(workspace)
#         pixel_data = image_set.get_image("MyFile").pixel_data
#         assertEqual(pixel_data.shape[0], 30)
#         assertEqual(pixel_data.shape[1], 20)
#         value = m.get_current_image_measurement("Metadata_Unicode")
#         assertEqual(value, unicode_value)
#     finally:
#         if os.path.exists(path):
#             try:
#                 os.unlink(path)
#             except:
#                 pass
#
#         if os.path.exists(csv_path):
#             try:
#                 os.unlink(csv_path)
#             except:
#                 pass
#         if os.path.exists(base_path):
#             try:
#                 os.rmdir(base_path)
#             except:
#                 pass
#         if os.path.exists(base_directory):
#             try:
#                 os.rmdir(base_directory)
#             except:
#                 pass


def test_load_filename():
    #
    # Load a file, only specifying the FileName in the CSV
    #
    csv_text = (
        """"Image_FileName_DNA"
                "%s"
                """
        % test_filename
    )
    pipeline, module, filename = make_pipeline(csv_text)
    assert isinstance(module, cellprofiler.modules.loaddata.LoadData)
    module.image_directory.dir_choice = cellprofiler.setting.ABSOLUTE_FOLDER_NAME
    module.image_directory.custom_path = test_path
    m = cellprofiler.measurement.Measurements()
    workspace = cellprofiler.workspace.Workspace(
        pipeline,
        module,
        m,
        cellprofiler.object.ObjectSet(),
        m,
        cellprofiler.image.ImageSetList(),
    )
    assert module.prepare_run(workspace)
    assert (
        m.get_measurement(cellprofiler.measurement.IMAGE, "FileName_DNA", 1)
        == test_filename
    )
    path = m.get_measurement(cellprofiler.measurement.IMAGE, "PathName_DNA", 1)
    assert path == test_path
    assert m.get_measurement(
        cellprofiler.measurement.IMAGE, "URL_DNA", 1
    ) == cellprofiler.modules.loadimages.pathname2url(
        os.path.join(test_path, test_filename)
    )
    module.prepare_group(workspace, {}, [1])
    module.run(workspace)
    img = workspace.image_set.get_image("DNA", must_be_grayscale=True)
    assert tuple(img.pixel_data.shape) == test_shape


def test_load_url():
    #
    # Load, only specifying URL
    #
    csv_text = """"Image_URL_DNA"
"{cp_logo_url}"
"http:{cp_logo_url_filename}"
"bogusurl.png"
""".format(
        **{
            "cp_logo_url": tests.modules.cp_logo_url,
            "cp_logo_url_filename": tests.modules.cp_logo_url_filename,
        }
    )
    pipeline, module, filename = make_pipeline(csv_text)
    assert isinstance(module, cellprofiler.modules.loaddata.LoadData)
    m = cellprofiler.measurement.Measurements()
    workspace = cellprofiler.workspace.Workspace(
        pipeline,
        module,
        m,
        cellprofiler.object.ObjectSet(),
        m,
        cellprofiler.image.ImageSetList(),
    )
    assert module.prepare_run(workspace)
    assert (
        m.get_measurement(cellprofiler.measurement.IMAGE, "FileName_DNA", 1)
        == tests.modules.cp_logo_url_filename
    )
    path = m.get_measurement(cellprofiler.measurement.IMAGE, "PathName_DNA", 1)
    assert path == tests.modules.cp_logo_url_folder
    assert m[cellprofiler.measurement.IMAGE, "URL_DNA", 1] == tests.modules.cp_logo_url
    assert (
        m[cellprofiler.measurement.IMAGE, "FileName_DNA", 2]
        == tests.modules.cp_logo_url_filename
    )
    assert m[cellprofiler.measurement.IMAGE, "PathName_DNA", 2] == "http:"
    assert m[cellprofiler.measurement.IMAGE, "FileName_DNA", 3] == "bogusurl.png"
    assert m[cellprofiler.measurement.IMAGE, "PathName_DNA", 3] == ""
    module.prepare_group(workspace, {}, [1])
    module.run(workspace)
    img = workspace.image_set.get_image("DNA", must_be_color=True)
    assert tuple(img.pixel_data.shape) == tests.modules.cp_logo_url_shape


def test_extra_fields():
    #
    # Regression test of issue #853, extra fields
    #
    csv_text = """"Image_URL_DNA"
"{cp_logo_url}", "foo"
"http:{cp_logo_url_filename}"
"bogusurl.png"
""".format(
        **{
            "cp_logo_url": tests.modules.cp_logo_url,
            "cp_logo_url_filename": tests.modules.cp_logo_url_filename,
        }
    )
    pipeline, module, filename = make_pipeline(csv_text)
    assert isinstance(module, cellprofiler.modules.loaddata.LoadData)
    m = cellprofiler.measurement.Measurements()
    workspace = cellprofiler.workspace.Workspace(
        pipeline,
        module,
        m,
        cellprofiler.object.ObjectSet(),
        m,
        cellprofiler.image.ImageSetList(),
    )
    assert module.prepare_run(workspace)
    assert (
        m.get_measurement(cellprofiler.measurement.IMAGE, "FileName_DNA", 1)
        == tests.modules.cp_logo_url_filename
    )
    path = m.get_measurement(cellprofiler.measurement.IMAGE, "PathName_DNA", 1)
    assert path == tests.modules.cp_logo_url_folder
    assert (
        m.get_measurement(cellprofiler.measurement.IMAGE, "URL_DNA", 1)
        == tests.modules.cp_logo_url
    )
    assert (
        m[cellprofiler.measurement.IMAGE, "FileName_DNA", 2]
        == tests.modules.cp_logo_url_filename
    )
    assert m[cellprofiler.measurement.IMAGE, "PathName_DNA", 2] == "http:"
    assert m[cellprofiler.measurement.IMAGE, "FileName_DNA", 3] == "bogusurl.png"
    assert m[cellprofiler.measurement.IMAGE, "PathName_DNA", 3] == ""
    module.prepare_group(workspace, {}, [1])
    module.run(workspace)
    img = workspace.image_set.get_image("DNA", must_be_color=True)
    assert tuple(img.pixel_data.shape) == tests.modules.cp_logo_url_shape


def test_extra_lines():
    #
    # Regression test of issue #1211 - extra line at end / blank lines
    #
    dir = os.path.join(tests.modules.example_images_directory(), "ExampleSBSImages")
    file_name = "Channel2-01-A-01.tif"

    csv_text = """"Image_FileName_DNA","Image_PathName_DNA"
"%s","%s"

""" % (
        file_name,
        dir,
    )
    pipeline, module, filename = make_pipeline(csv_text)
    try:
        assert isinstance(module, cellprofiler.modules.loaddata.LoadData)
        m = cellprofiler.measurement.Measurements()
        workspace = cellprofiler.workspace.Workspace(
            pipeline,
            module,
            m,
            cellprofiler.object.ObjectSet(),
            m,
            cellprofiler.image.ImageSetList(),
        )
        assert module.prepare_run(workspace)
        assert isinstance(m, cellprofiler.measurement.Measurements)
        assert m.image_set_count == 1
        assert "FileName_DNA" in m.get_feature_names(cellprofiler.measurement.IMAGE)
        assert m[cellprofiler.measurement.IMAGE, "FileName_DNA", 1] == file_name
    finally:
        os.remove(filename)


def test_extra_lines_skip_rows():
    #
    # Regression test of issue #1211 - extra line at end / blank lines
    # Different code path from 13_04
    #
    path = os.path.join(tests.modules.example_images_directory(), "ExampleSBSImages")
    file_names = ["Channel2-01-A-01.tif", "Channel2-02-A-02.tif"]

    csv_text = """"Image_FileName_DNA","Image_PathName_DNA"

"%s","%s"

"%s","%s"

""" % (
        file_names[0],
        path,
        file_names[1],
        path,
    )
    pipeline, module, filename = make_pipeline(csv_text)
    try:
        assert isinstance(module, cellprofiler.modules.loaddata.LoadData)
        m = cellprofiler.measurement.Measurements()
        workspace = cellprofiler.workspace.Workspace(
            pipeline,
            module,
            m,
            cellprofiler.object.ObjectSet(),
            m,
            cellprofiler.image.ImageSetList(),
        )
        module.wants_rows.value = True
        module.row_range.min = 2
        module.row_range.max = 3
        assert module.prepare_run(workspace)
        assert isinstance(m, cellprofiler.measurement.Measurements)
        assert m.image_set_count == 1
        assert "FileName_DNA" in m.get_feature_names(cellprofiler.measurement.IMAGE)
        assert m[cellprofiler.measurement.IMAGE, "FileName_DNA", 1] == file_names[0]
    finally:
        os.remove(filename)


def test_load_default_input_folder():
    # Regression test of issue #1365 - load a file from the default
    # input folder and check that PathName_xxx is absolute
    csv_text = '''"Image_FileName_DNA","Image_PathName_DNA"\n"%s","%s"''' % (
        test_filename,
        test_path,
    )
    pipeline, module, filename = make_pipeline(csv_text)
    try:
        assert isinstance(module, cellprofiler.modules.loaddata.LoadData)
        module.image_directory.dir_choice = cellprofiler.setting.ABSOLUTE_FOLDER_NAME
        module.image_directory.custom_path = test_path
        m = cellprofiler.measurement.Measurements()
        workspace = cellprofiler.workspace.Workspace(
            pipeline,
            module,
            m,
            cellprofiler.object.ObjectSet(),
            m,
            cellprofiler.image.ImageSetList(),
        )
        assert module.prepare_run(workspace)
        assert (
            m.get_measurement(cellprofiler.measurement.IMAGE, "FileName_DNA", 1)
            == test_filename
        )
        path_out = m.get_measurement(cellprofiler.measurement.IMAGE, "PathName_DNA", 1)
        assert test_path == path_out
        assert m.get_measurement(
            cellprofiler.measurement.IMAGE, "URL_DNA", 1
        ) == cellprofiler.modules.loadimages.pathname2url(
            os.path.join(test_path, test_filename)
        )
        module.prepare_group(workspace, {}, [1])
        module.run(workspace)
        img = workspace.image_set.get_image("DNA", must_be_grayscale=True)
        assert tuple(img.pixel_data.shape) == test_shape
    finally:
        os.remove(filename)


class C0(cellprofiler.module.Module):
    module_name = "C0"
    variable_revision_number = 1

    def create_settings(self):
        callback = None

    def settings(self):
        return []

    def run(self, workspace):
        callback(workspace)
