import io
import os
import tempfile

import cellprofiler.measurement
import cellprofiler.modules.images
import cellprofiler.modules.metadata
import cellprofiler.pipeline
import cellprofiler.setting
import cellprofiler.workspace

OME_XML = open(
    os.path.join(os.path.split(__file__)[0], "../resources/omexml.xml"), "r"
).read()


def test_load_v1():
    with open("./tests/resources/modules/metadata/v1.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.metadata.Metadata)
    assert module.wants_metadata
    assert len(module.extraction_methods) == 2
    em0, em1 = module.extraction_methods
    assert em0.extraction_method == cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
    assert em0.source == cellprofiler.modules.metadata.XM_FILE_NAME
    assert (
        em0.file_regexp.value
        == r"^Channel(?P<ChannelNumber>[12])-(?P<Index>[0-9]+)-(?P<WellRow>[A-H])-(?P<WellColumn>[0-9]{2}).tif$"
    )
    assert em0.folder_regexp.value == r"(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$"
    assert em0.filter_choice == cellprofiler.modules.metadata.F_ALL_IMAGES
    assert em0.filter == 'or (file does contain "Channel2")'
    assert not em0.wants_case_insensitive

    assert em1.extraction_method == cellprofiler.modules.metadata.X_IMPORTED_EXTRACTION
    assert em1.source == cellprofiler.modules.metadata.XM_FOLDER_NAME
    assert em1.filter_choice == cellprofiler.modules.metadata.F_FILTERED_IMAGES
    assert (
        em1.csv_location.get_dir_choice() == cellprofiler.setting.ABSOLUTE_FOLDER_NAME
    )
    assert em1.csv_location.get_custom_path() == "/imaging/analysis"
    assert em1.csv_filename.value == "metadata.csv"
    assert (
        em1.csv_joiner.value
        == "[{'Image Metadata': u'ChannelNumber', 'CSV Metadata': u'Wavelength'}]"
    )
    assert not em1.wants_case_insensitive


def test_load_v2():
    with open("./tests/resources/modules/metadata/v2.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.metadata.Metadata)
    assert module.wants_metadata
    assert len(module.extraction_methods) == 2
    em0, em1 = module.extraction_methods
    assert em0.extraction_method == cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
    assert em0.source == cellprofiler.modules.metadata.XM_FILE_NAME
    assert (
        em0.file_regexp.value
        == r"^Channel(?P<ChannelNumber>[12])-(?P<Index>[0-9]+)-(?P<WellRow>[A-H])-(?P<WellColumn>[0-9]{2}).tif$"
    )
    assert em0.folder_regexp.value == r"(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$"
    assert em0.filter_choice == cellprofiler.modules.metadata.F_ALL_IMAGES
    assert em0.filter == 'or (file does contain "Channel2")'
    assert not em0.wants_case_insensitive

    assert em1.extraction_method == cellprofiler.modules.metadata.X_IMPORTED_EXTRACTION
    assert em1.source == cellprofiler.modules.metadata.XM_FOLDER_NAME
    assert em1.filter_choice == cellprofiler.modules.metadata.F_FILTERED_IMAGES
    assert (
        em1.csv_location.get_dir_choice() == cellprofiler.setting.ABSOLUTE_FOLDER_NAME
    )
    assert em1.csv_location.get_custom_path() == "/imaging/analysis"
    assert em1.csv_filename.value == "metadata.csv"
    assert (
        em1.csv_joiner.value
        == "[{'Image Metadata': u'ChannelNumber', 'CSV Metadata': u'Wavelength'}]"
    )
    assert em1.wants_case_insensitive


def test_load_v3():
    with open("./tests/resources/modules/metadata/v3.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.metadata.Metadata)
    assert module.wants_metadata
    assert module.data_type_choice == cellprofiler.modules.metadata.DTC_TEXT
    assert len(module.extraction_methods) == 2
    em0, em1 = module.extraction_methods
    assert em0.extraction_method == cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
    assert em0.source == cellprofiler.modules.metadata.XM_FILE_NAME
    assert (
        em0.file_regexp.value
        == r"^Channel(?P<ChannelNumber>[12])-(?P<Index>[0-9]+)-(?P<WellRow>[A-H])-(?P<WellColumn>[0-9]{2}).tif$"
    )
    assert em0.folder_regexp.value == r"(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$"
    assert em0.filter_choice == cellprofiler.modules.metadata.F_ALL_IMAGES
    assert em0.filter == 'or (file does contain "Channel2")'
    assert not em0.wants_case_insensitive

    assert em1.extraction_method == cellprofiler.modules.metadata.X_IMPORTED_EXTRACTION
    assert em1.source == cellprofiler.modules.metadata.XM_FOLDER_NAME
    assert em1.filter_choice == cellprofiler.modules.metadata.F_FILTERED_IMAGES
    assert (
        em1.csv_location.get_dir_choice() == cellprofiler.setting.ABSOLUTE_FOLDER_NAME
    )
    assert em1.csv_location.get_custom_path() == "/imaging/analysis"
    assert em1.csv_filename.value == "metadata.csv"
    assert (
        em1.csv_joiner.value
        == "[{'Image Metadata': u'ChannelNumber', 'CSV Metadata': u'Wavelength'}]"
    )
    assert em1.wants_case_insensitive


def test_load_v4():
    with open("./tests/resources/modules/metadata/v4.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.metadata.Metadata)
    assert module.wants_metadata
    assert module.data_type_choice == cellprofiler.modules.metadata.DTC_CHOOSE
    d = cellprofiler.setting.DataTypes.decode_data_types(module.data_types.value_text)
    for k, v in (
        ("Index", cellprofiler.setting.DataTypes.DT_NONE),
        ("WellRow", cellprofiler.setting.DataTypes.DT_TEXT),
        ("WellColumn", cellprofiler.setting.DataTypes.DT_FLOAT),
        ("ChannelNumber", cellprofiler.setting.DataTypes.DT_INTEGER),
    ):
        assert k in d
        assert d[k] == v
    assert len(module.extraction_methods) == 2
    em0, em1 = module.extraction_methods
    assert em0.extraction_method == cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
    assert em0.source == cellprofiler.modules.metadata.XM_FILE_NAME
    assert (
        em0.file_regexp.value
        == r"^Channel(?P<ChannelNumber>[12])-(?P<Index>[0-9]+)-(?P<WellRow>[A-H])-(?P<WellColumn>[0-9]{2}).tif$"
    )
    assert em0.folder_regexp.value == r"(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$"
    assert em0.filter_choice == cellprofiler.modules.metadata.F_ALL_IMAGES
    assert em0.filter == 'or (file does contain "Channel2")'
    assert not em0.wants_case_insensitive

    assert em1.extraction_method == cellprofiler.modules.metadata.X_IMPORTED_EXTRACTION
    assert em1.source == cellprofiler.modules.metadata.XM_FOLDER_NAME
    assert em1.filter_choice == cellprofiler.modules.metadata.F_FILTERED_IMAGES
    assert (
        em1.csv_location.get_dir_choice() == cellprofiler.setting.ABSOLUTE_FOLDER_NAME
    )
    assert em1.csv_location.get_custom_path() == "/imaging/analysis"
    assert em1.csv_filename.value == "metadata.csv"
    assert (
        em1.csv_joiner.value
        == "[{'Image Metadata': u'ChannelNumber', 'CSV Metadata': u'Wavelength'}]"
    )
    assert em1.wants_case_insensitive


def test_load_v5():
    with open("./tests/resources/modules/metadata/v5.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]

    em0, em1 = module.extraction_methods

    assert (
        em0.csv_location.get_dir_choice() == cellprofiler.setting.ABSOLUTE_FOLDER_NAME
    )
    assert em0.csv_location.get_custom_path() == "/imaging/analysis"
    assert em0.csv_filename.value == "metadata.csv"

    assert em1.csv_location.get_dir_choice() == cellprofiler.setting.URL_FOLDER_NAME
    assert em1.csv_location.get_custom_path() == "https://cellprofiler.org"
    assert em1.csv_filename.value == "metadata.csv"


def check(module, url, dd, keys=None, xml=None):
    """Check that running the metadata module on a url generates the expected dictionary"""
    pipeline = cellprofiler.pipeline.Pipeline()
    imgs = cellprofiler.modules.images.Images()
    imgs.filter_choice.value = cellprofiler.modules.images.FILTER_CHOICE_NONE
    imgs.module_num = 1
    pipeline.add_module(imgs)
    module.set_module_num(2)
    pipeline.add_module(module)
    pipeline.add_urls([url])
    m = cellprofiler.measurement.Measurements()
    workspace = cellprofiler.workspace.Workspace(pipeline, module, None, None, m, None)
    file_list = workspace.file_list
    file_list.add_files_to_filelist([url])
    if xml is not None:
        file_list.add_metadata(url, xml)
    ipds = pipeline.get_image_plane_details(workspace)
    assert len(ipds) == len(dd)
    for d, ipd in zip(dd, ipds):
        assert dict(ipd.metadata, **d) == ipd.metadata
    all_keys = list(pipeline.get_available_metadata_keys().keys())
    if keys is not None:
        for key in keys:
            assert key in all_keys


def test_get_metadata_from_filename():
    module = cellprofiler.modules.metadata.Metadata()
    module.wants_metadata.value = True
    em = module.extraction_methods[0]
    em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
    em.extraction_method.value = cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
    em.source.value = cellprofiler.modules.metadata.XM_FILE_NAME
    em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-H][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
    em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
    url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
    check(
        module,
        url,
        [{"Plate": "P-12345", "Well": "B08", "Site": "5", "Wavelength": "2"}],
        ("Plate", "Well", "Site", "Wavelength"),
    )


def test_get_metadata_from_path():
    module = cellprofiler.modules.metadata.Metadata()
    module.wants_metadata.value = True
    em = module.extraction_methods[0]
    em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
    em.extraction_method.value = cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
    em.source.value = cellprofiler.modules.metadata.XM_FOLDER_NAME
    em.folder_regexp.value = r".*[/\\](?P<Plate>.+)$"
    em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
    url = "file:/imaging/analysis/P-12345/_B08_s5_w2.tif"
    check(module, url, [{"Plate": "P-12345"}], ("Plate",))


def test_filter_positive():
    module = cellprofiler.modules.metadata.Metadata()
    module.wants_metadata.value = True
    em = module.extraction_methods[0]
    em.filter_choice.value = cellprofiler.modules.metadata.F_FILTERED_IMAGES
    em.filter.value = 'or (file does contain "B08")'
    em.extraction_method.value = cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
    em.source.value = cellprofiler.modules.metadata.XM_FILE_NAME
    em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-H][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
    em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
    url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
    check(
        module,
        url,
        [{"Plate": "P-12345", "Well": "B08", "Site": "5", "Wavelength": "2"}],
    )


def test_filter_negative():
    module = cellprofiler.modules.metadata.Metadata()
    module.wants_metadata.value = True
    em = module.extraction_methods[0]
    em.filter_choice.value = cellprofiler.modules.metadata.F_FILTERED_IMAGES
    em.filter.value = 'or (file doesnot contain "B08")'
    em.extraction_method.value = cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
    em.source.value = cellprofiler.modules.metadata.XM_FILE_NAME
    em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-H][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
    em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
    url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
    check(
        module,
        url,
        [{"Plate": "P-12345", "Well": "B08", "Site": "5", "Wavelength": "2"}],
    )


def test_imported_extraction():
    metadata_csv = """WellName,Treatment,Dose,Counter
B08,DMSO,0,1
C10,BRD041618,1.5,2
"""
    filenum, path = tempfile.mkstemp(suffix=".csv")
    fd = os.fdopen(filenum, "w")
    fd.write(metadata_csv)
    fd.close()
    try:
        module = cellprofiler.modules.metadata.Metadata()
        module.wants_metadata.value = True
        module.data_type_choice.value = cellprofiler.modules.metadata.DTC_CHOOSE
        module.data_types.value = cellprofiler.setting.json.dumps(
            dict(
                Plate=cellprofiler.setting.DataTypes.DT_TEXT,
                Well=cellprofiler.setting.DataTypes.DT_TEXT,
                WellName=cellprofiler.setting.DataTypes.DT_NONE,
                Treatment=cellprofiler.setting.DataTypes.DT_TEXT,
                Dose=cellprofiler.setting.DataTypes.DT_FLOAT,
                Counter=cellprofiler.setting.DataTypes.DT_NONE,
            )
        )
        module.add_extraction_method()
        em = module.extraction_methods[0]
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
        em.extraction_method.value = cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
        em.source.value = cellprofiler.modules.metadata.XM_FILE_NAME
        em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-Ha-h][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES

        em = module.extraction_methods[1]
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
        em.extraction_method.value = cellprofiler.modules.metadata.X_IMPORTED_EXTRACTION
        directory, filename = os.path.split(path)
        em.csv_location.value = "{}|{}".format(
            cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory
        )
        em.csv_filename.value = filename
        em.csv_joiner.value = '[{"%s":"WellName","%s":"Well"}]' % (
            module.CSV_JOIN_NAME,
            module.IPD_JOIN_NAME,
        )
        url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
        check(
            module,
            url,
            [
                {
                    "Plate": "P-12345",
                    "Well": "B08",
                    "Site": "5",
                    "Wavelength": "2",
                    "Treatment": "DMSO",
                    "Dose": "0",
                    "Counter": "1",
                }
            ],
        )
        url = "file:/imaging/analysis/P-12345_C10_s2_w3.tif"
        check(
            module,
            url,
            [
                {
                    "Plate": "P-12345",
                    "Well": "C10",
                    "Site": "2",
                    "Wavelength": "3",
                    "Treatment": "BRD041618",
                    "Dose": "1.5",
                    "Counter": "2",
                }
            ],
        )
        url = "file:/imaging/analysis/P-12345_A01_s2_w3.tif"
        check(
            module,
            url,
            [{"Plate": "P-12345", "Well": "A01", "Site": "2", "Wavelength": "3"}],
        )
        pipeline = cellprofiler.pipeline.Pipeline()
        imgs = cellprofiler.modules.images.Images()
        imgs.filter_choice.value = cellprofiler.modules.images.FILTER_CHOICE_NONE
        imgs.module_num = 1
        pipeline.add_module(imgs)
        module.set_module_num(2)
        pipeline.add_module(module)
        columns = module.get_measurement_columns(pipeline)
        assert not any([c[1] == "Counter" for c in columns])
        for feature_name, data_type in (
            ("Metadata_Treatment", cellprofiler.measurement.COLTYPE_VARCHAR_FILE_NAME),
            ("Metadata_Dose", cellprofiler.measurement.COLTYPE_FLOAT),
        ):
            assert any(
                [
                    c[0] == cellprofiler.measurement.IMAGE
                    and c[1] == feature_name
                    and c[2] == data_type
                    for c in columns
                ]
            )
    finally:
        try:
            os.unlink(path)
        except:
            pass


def test_imported_extraction_case_insensitive():
    metadata_csv = """WellName,Treatment
b08,DMSO
C10,BRD041618
"""
    filenum, path = tempfile.mkstemp(suffix=".csv")
    fd = os.fdopen(filenum, "w")
    fd.write(metadata_csv)
    fd.close()
    try:
        module = cellprofiler.modules.metadata.Metadata()
        module.wants_metadata.value = True
        module.add_extraction_method()
        em = module.extraction_methods[0]
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
        em.extraction_method.value = cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
        em.source.value = cellprofiler.modules.metadata.XM_FILE_NAME
        em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-Ha-h][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES

        em = module.extraction_methods[1]
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
        em.extraction_method.value = cellprofiler.modules.metadata.X_IMPORTED_EXTRACTION
        directory, filename = os.path.split(path)
        em.csv_location.value = "{}|{}".format(
            cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory
        )
        em.csv_filename.value = filename
        em.csv_joiner.value = '[{"%s":"WellName","%s":"Well"}]' % (
            module.CSV_JOIN_NAME,
            module.IPD_JOIN_NAME,
        )
        em.wants_case_insensitive.value = True
        url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
        check(
            module,
            url,
            [
                {
                    "Plate": "P-12345",
                    "Well": "B08",
                    "Site": "5",
                    "Wavelength": "2",
                    "Treatment": "DMSO",
                }
            ],
        )
        url = "file:/imaging/analysis/P-12345_c10_s2_w3.tif"
        check(
            module,
            url,
            [
                {
                    "Plate": "P-12345",
                    "Well": "c10",
                    "Site": "2",
                    "Wavelength": "3",
                    "Treatment": "BRD041618",
                }
            ],
        )
        url = "file:/imaging/analysis/P-12345_A01_s2_w3.tif"
        check(
            module,
            url,
            [{"Plate": "P-12345", "Well": "A01", "Site": "2", "Wavelength": "3"}],
        )
    finally:
        try:
            os.unlink(path)
        except:
            pass


def test_imported_extraction_case_sensitive():
    metadata_csv = """WellName,Treatment
b08,DMSO
C10,BRD041618
"""
    filenum, path = tempfile.mkstemp(suffix=".csv")
    fd = os.fdopen(filenum, "w")
    fd.write(metadata_csv)
    fd.close()
    try:
        module = cellprofiler.modules.metadata.Metadata()
        module.wants_metadata.value = True
        module.add_extraction_method()
        em = module.extraction_methods[0]
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
        em.extraction_method.value = cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
        em.source.value = cellprofiler.modules.metadata.XM_FILE_NAME
        em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-H][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES

        em = module.extraction_methods[1]
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
        em.extraction_method.value = cellprofiler.modules.metadata.X_IMPORTED_EXTRACTION
        directory, filename = os.path.split(path)
        em.csv_location.value = "{}|{}".format(
            cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory
        )
        em.csv_filename.value = filename
        em.csv_joiner.value = '[{"%s":"WellName","%s":"Well"}]' % (
            module.CSV_JOIN_NAME,
            module.IPD_JOIN_NAME,
        )
        em.wants_case_insensitive.value = False
        url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
        check(
            module,
            url,
            [{"Plate": "P-12345", "Well": "B08", "Site": "5", "Wavelength": "2"}],
        )
        url = "file:/imaging/analysis/P-12345_C10_s2_w3.tif"
        check(
            module,
            url,
            [
                {
                    "Plate": "P-12345",
                    "Well": "C10",
                    "Site": "2",
                    "Wavelength": "3",
                    "Treatment": "BRD041618",
                }
            ],
        )
        url = "file:/imaging/analysis/P-12345_A01_s2_w3.tif"
        check(
            module,
            url,
            [{"Plate": "P-12345", "Well": "A01", "Site": "2", "Wavelength": "3"}],
        )
    finally:
        try:
            os.unlink(path)
        except:
            pass


def test_numeric_joining():
    # Check that Metadata correctly joins metadata items
    # that are supposed to be extracted as numbers
    metadata_csv = """Site,Treatment
05,DMSO
02,BRD041618
"""
    filenum, path = tempfile.mkstemp(suffix=".csv")
    fd = os.fdopen(filenum, "w")
    fd.write(metadata_csv)
    fd.close()
    try:
        module = cellprofiler.modules.metadata.Metadata()
        module.wants_metadata.value = True
        module.data_types.value = cellprofiler.setting.DataTypes.encode_data_types(
            {"Site": cellprofiler.setting.DataTypes.DT_INTEGER}
        )
        module.data_type_choice.value = cellprofiler.modules.metadata.DTC_CHOOSE
        module.add_extraction_method()
        em = module.extraction_methods[0]
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
        em.extraction_method.value = cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
        em.source.value = cellprofiler.modules.metadata.XM_FILE_NAME
        em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-H][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES

        em = module.extraction_methods[1]
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
        em.extraction_method.value = cellprofiler.modules.metadata.X_IMPORTED_EXTRACTION
        directory, filename = os.path.split(path)
        em.csv_location.value = "{}|{}".format(
            cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory
        )
        em.csv_filename.value = filename
        em.csv_joiner.value = '[{"%s":"Site","%s":"Site"}]' % (
            module.CSV_JOIN_NAME,
            module.IPD_JOIN_NAME,
        )
        em.wants_case_insensitive.value = False
        url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
        check(
            module,
            url,
            [
                {
                    "Plate": "P-12345",
                    "Well": "B08",
                    "Site": "5",
                    "Wavelength": "2",
                    "Treatment": "DMSO",
                }
            ],
        )
        url = "file:/imaging/analysis/P-12345_C10_s2_w3.tif"
        check(
            module,
            url,
            [
                {
                    "Plate": "P-12345",
                    "Well": "C10",
                    "Site": "2",
                    "Wavelength": "3",
                    "Treatment": "BRD041618",
                }
            ],
        )
        url = "file:/imaging/analysis/P-12345_A01_s3_w3.tif"
        check(
            module,
            url,
            [{"Plate": "P-12345", "Well": "A01", "Site": "3", "Wavelength": "3"}],
        )
    finally:
        try:
            os.unlink(path)
        except:
            pass


def test_too_many_columns():
    # Regression test of issue #853
    # Allow .csv files which have rows with more fields than there
    # are header fields.
    metadata_csv = """WellName,Treatment
b08,DMSO,foo
C10,BRD041618,bar
"""
    filenum, path = tempfile.mkstemp(suffix=".csv")
    fd = os.fdopen(filenum, "w")
    fd.write(metadata_csv)
    fd.close()
    try:
        module = cellprofiler.modules.metadata.Metadata()
        module.wants_metadata.value = True
        module.add_extraction_method()
        em = module.extraction_methods[0]
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
        em.extraction_method.value = cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
        em.source.value = cellprofiler.modules.metadata.XM_FILE_NAME
        em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-Ha-h][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES

        em = module.extraction_methods[1]
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
        em.extraction_method.value = cellprofiler.modules.metadata.X_IMPORTED_EXTRACTION
        directory, filename = os.path.split(path)
        em.csv_location.value = "{}|{}".format(
            cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory
        )
        em.csv_filename.value = filename
        em.csv_joiner.value = '[{"%s":"WellName","%s":"Well"}]' % (
            module.CSV_JOIN_NAME,
            module.IPD_JOIN_NAME,
        )
        em.wants_case_insensitive.value = True
        url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
        check(
            module,
            url,
            [
                {
                    "Plate": "P-12345",
                    "Well": "B08",
                    "Site": "5",
                    "Wavelength": "2",
                    "Treatment": "DMSO",
                }
            ],
        )
        url = "file:/imaging/analysis/P-12345_c10_s2_w3.tif"
        check(
            module,
            url,
            [
                {
                    "Plate": "P-12345",
                    "Well": "c10",
                    "Site": "2",
                    "Wavelength": "3",
                    "Treatment": "BRD041618",
                }
            ],
        )
        url = "file:/imaging/analysis/P-12345_A01_s2_w3.tif"
        check(
            module,
            url,
            [{"Plate": "P-12345", "Well": "A01", "Site": "2", "Wavelength": "3"}],
        )
    finally:
        try:
            os.unlink(path)
        except:
            pass


def test_well_row_column():
    # Make sure that Metadata_Well is generated if we have
    # Metadata_Row and Metadata_Column
    #
    for row_tag, column_tag in (
        ("row", "column"),
        ("wellrow", "wellcolumn"),
        ("well_row", "well_column"),
    ):
        module = cellprofiler.modules.metadata.Metadata()
        module.wants_metadata.value = True
        em = module.extraction_methods[0]
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
        em.extraction_method.value = cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
        em.source.value = cellprofiler.modules.metadata.XM_FILE_NAME
        em.file_regexp.value = (
            "^Channel(?P<Wavelength>[1-2])-"
            "(?P<%(row_tag)s>[A-H])-"
            "(?P<%(column_tag)s>[0-9]{2}).tif$"
        ) % locals()
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
        url = "file:/imaging/analysis/Channel1-C-05.tif"
        check(
            module,
            url,
            [
                {
                    "Wavelength": "1",
                    row_tag: "C",
                    column_tag: "05",
                    cellprofiler.measurement.FTR_WELL: "C05",
                }
            ],
        )
        pipeline = cellprofiler.pipeline.Pipeline()
        imgs = cellprofiler.modules.images.Images()
        imgs.filter_choice.value = cellprofiler.modules.images.FILTER_CHOICE_NONE
        imgs.module_num = 1
        pipeline.add_module(imgs)
        module.set_module_num(2)
        pipeline.add_module(module)
        assert cellprofiler.measurement.M_WELL in [
            c[1] for c in module.get_measurement_columns(pipeline)
        ]


def test_well_row_column_before_import():
    # Regression test for issue #1347
    # WellRow and WellColumn must be united asap so they can
    # be used downstream.
    #
    module = cellprofiler.modules.metadata.Metadata()
    module.wants_metadata.value = True
    em = module.extraction_methods[0]
    em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
    em.extraction_method.value = cellprofiler.modules.metadata.X_MANUAL_EXTRACTION
    em.source.value = cellprofiler.modules.metadata.XM_FILE_NAME
    em.file_regexp.value = (
        "^Channel(?P<Wavelength>[1-2])-" "(?P<%s>[A-H])-" "(?P<%s>[0-9]{2}).tif$"
    ) % (cellprofiler.measurement.FTR_ROW, cellprofiler.measurement.FTR_COLUMN)
    em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
    module.add_extraction_method()
    metadata_csv = """WellName,Treatment
C05,DMSO
"""
    filenum, path = tempfile.mkstemp(suffix=".csv")
    fd = os.fdopen(filenum, "w")
    fd.write(metadata_csv)
    fd.close()
    try:
        em = module.extraction_methods[1]
        em.extraction_method.value = cellprofiler.modules.metadata.X_IMPORTED_EXTRACTION
        em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
        em.extraction_method.value = cellprofiler.modules.metadata.X_IMPORTED_EXTRACTION
        directory, filename = os.path.split(path)
        em.csv_location.value = "{}|{}".format(
            cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory
        )
        em.csv_filename.value = filename
        em.csv_joiner.value = '[{"%s":"WellName","%s":"Well"}]' % (
            module.CSV_JOIN_NAME,
            module.IPD_JOIN_NAME,
        )
        url = "file:/imaging/analysis/Channel1-C-05.tif"
        check(
            module,
            url,
            [
                {
                    "Wavelength": "1",
                    cellprofiler.measurement.FTR_ROW: "C",
                    cellprofiler.measurement.FTR_COLUMN: "05",
                    "Treatment": "DMSO",
                    cellprofiler.measurement.FTR_WELL: "C05",
                }
            ],
        )
    except:
        os.remove(path)


def test_ome_metadata():
    # Test loading one URL with the humongous stack XML
    # (pat self on back if passes)
    module = cellprofiler.modules.metadata.Metadata()
    module.wants_metadata.value = True
    em = module.extraction_methods[0]
    em.filter_choice.value = cellprofiler.modules.metadata.F_ALL_IMAGES
    em.extraction_method.value = cellprofiler.modules.metadata.X_AUTOMATIC_EXTRACTION
    url = "file:/imaging/analysis/Channel1-C-05.tif"
    metadata = []
    for series in range(4):
        for z in range(36):
            metadata.append(
                dict(
                    Series=str(series),
                    Frame=str(z),
                    Plate="136570140804 96_Greiner",
                    Well="E11",
                    Site=str(series),
                    ChannelName="Exp1Cam1",
                    SizeX=str(688),
                    SizeY=str(512),
                    SizeZ=str(36),
                    SizeC=str(1),
                    SizeT=str(1),
                    Z=str(z),
                    C=str(0),
                    T=str(0),
                )
            )
    check(module, url, metadata, xml=OME_XML)
