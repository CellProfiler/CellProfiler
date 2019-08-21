"""test_ExportToSpreadsheet.py - test the ExportToSpreadsheet module
"""

import base64
import csv
import os
import tempfile

import numpy as np
import six
from six.moves import StringIO

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.pipeline as cpp
import cellprofiler.setting
import cellprofiler.workspace as cpw
import cellprofiler.modules.exporttospreadsheet as E
from cellprofiler.modules import identifyprimaryobjects
from cellprofiler.measurement import C_COUNT, M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y
from tests.modules import example_images_directory, maybe_download_sbs
import pytest

OBJECTS_NAME = "MyObjects"
IMG_MEAS = "my_image_measurement"
OBJ_MEAS = "my_object_measurement"


class TestExportToSpreadsheet:
    def setUp():
        output_dir = tempfile.mkdtemp()

    def tearDown():
        for file_name in os.listdir(output_dir):
            path = os.path.join(output_dir, file_name)
            if os.path.isdir(path):
                for ffiillee_nnaammee in os.listdir(path):
                    os.remove(os.path.join(path, ffiillee_nnaammee))
                os.rmdir(path)
            else:
                os.remove(path)
        os.rmdir(output_dir)
        output_dir = None

    def test_load_v3():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8948

ExportToSpreadsheet:[module_num:1|svn_version:\'8947\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Select or enter the column delimiter:Tab
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:No
    Add image/object numbers to output?:Yes
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:Yes
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:Yes
    Calculate the per-image standard deviation values for object measurements?:No
    Where do you want to save the files?:Custom folder with metadata
    Folder name\x3A:./\\<?Plate>
    Data to export:Image
    Combine these object measurements with those of the previous object?:No
    Name the data file (not including the output filename, if prepending was requested above):PFX_Image.csv
    Data to export:Nuclei
    Combine these object measurements with those of the previous object?:No
    Name the data file (not including the output filename, if prepending was requested above):Nuclei.csv
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert isinstance(module, E.ExportToSpreadsheet)
        assert module.delimiter_char == "\t"
        assert not module.add_metadata
        assert module.pick_columns
        assert not module.wants_aggregate_means
        assert module.wants_aggregate_medians
        assert not module.wants_aggregate_std
        assert module.directory.dir_choice == E.DEFAULT_OUTPUT_SUBFOLDER_NAME
        assert module.directory.custom_path == r"./\<?Plate>"
        assert len(module.object_groups) == 2
        for group, object_name, file_name in zip(
            module.object_groups, ("Image", "Nuclei"), ("PFX_Image.csv", "Nuclei.csv")
        ):
            assert group.name == object_name
            assert group.file_name == file_name
            assert not group.wants_automatic_file_name

    def test_load_v4():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9152

ExportToSpreadsheet:[module_num:1|svn_version:\'9144\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:No
    Add image metadata columns to your object data file?:No
    No longer used, always saved:No
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:No
    Where do you want to save the files?:Default output folder
    Folder name\x3A:.
    Export all measurements?:No
    Data to export:Image
    Combine these object measurements with those of the previous object?:No
    File name\x3A:Image.csv
    Use the object name for the file name?:Yes
    Data to export:Nuclei
    Combine these object measurements with those of the previous object?:No
    File name\x3A:Nuclei.csv
    Use the object name for the file name?:Yes
    Data to export:PropCells
    Combine these object measurements with those of the previous object?:No
    File name\x3A:PropCells.csv
    Use the object name for the file name?:Yes
    Data to export:DistanceCells
    Combine these object measurements with those of the previous object?:No
    File name\x3A:DistanceCells.csv
    Use the object name for the file name?:Yes
    Data to export:DistCytoplasm
    Combine these object measurements with those of the previous object?:No
    File name\x3A:DistCytoplasm.csv
    Use the object name for the file name?:Yes
    Data to export:PropCytoplasm
    Combine these object measurements with those of the previous object?:No
    File name\x3A:PropCytoplasm.csv
    Use the object name for the file name?:Yes
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert isinstance(module, E.ExportToSpreadsheet)
        assert module.delimiter == E.DELIMITER_COMMA
        assert not module.add_metadata
        assert not module.pick_columns
        assert not module.wants_aggregate_means
        assert not module.wants_aggregate_medians
        assert not module.wants_aggregate_std
        assert module.directory.dir_choice == E.DEFAULT_OUTPUT_FOLDER_NAME
        assert not module.wants_everything
        for group, object_name in zip(
            module.object_groups,
            (
                "Image",
                "Nuclei",
                "PropCells",
                "DistanceCells",
                "DistCytoplasm",
                "PropCytoplasm",
            ),
        ):
            assert group.name == object_name
            assert group.file_name == "%s.csv" % object_name
            assert not group.previous_file
            assert group.wants_automatic_file_name

    def test_load_v5():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9434

ExportToSpreadsheet:[module_num:1|svn_version:\'9434\'|variable_revision_number:5|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Tab
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:No
    No longer used, always saved:No
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:Yes
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:Yes
    Calculate the per-image standard deviation values for object measurements?:No
    Output file location:Default Output Folder
    Folder name://iodine/imaging_analysis/People/Lee
    Export all measurements?:No
    Press button to select measurements to export:Image\x7CFileName_rawGFP,Image\x7CFileName_IllumGFP,Image\x7CFileName_IllumDNA,Image\x7CFileName_rawDNA,Image\x7CMetadata_SBS_Doses,Image\x7CMetadata_Well,Image\x7CMetadata_Controls
    Data to export:Image
    Combine these object measurements with those of the previous object?:No
    File name:Image.csv
    Use the object name for the file name?:No
    Data to export:Nuclei
    Combine these object measurements with those of the previous object?:No
    File name:Nuclei.csv
    Use the object name for the file name?:No
    Data to export:PropCells
    Combine these object measurements with those of the previous object?:No
    File name:PropCells.csv
    Use the object name for the file name?:No
    Data to export:DistanceCells
    Combine these object measurements with those of the previous object?:No
    File name:DistanceCells.csv
    Use the object name for the file name?:No
    Data to export:DistCytoplasm
    Combine these object measurements with those of the previous object?:No
    File name:DistCytoplasm.csv
    Use the object name for the file name?:No
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert isinstance(module, E.ExportToSpreadsheet)
        assert module.delimiter == E.DELIMITER_TAB
        assert module.directory.dir_choice == E.DEFAULT_OUTPUT_FOLDER_NAME
        assert module.directory.custom_path == "//iodine/imaging_analysis/People/Lee"
        assert not module.add_metadata
        assert module.pick_columns
        assert all(
            [
                module.columns.get_measurement_object(x) == "Image"
                for x in module.columns.selections
            ]
        )
        assert len(module.columns.selections) == 7
        features = set(
            [
                module.columns.get_measurement_feature(x)
                for x in module.columns.selections
            ]
        )
        for feature in (
            "FileName_rawGFP",
            "FileName_IllumGFP",
            "FileName_IllumDNA",
            "FileName_rawDNA",
            "Metadata_SBS_Doses",
            "Metadata_Well",
            "Metadata_Controls",
        ):
            assert feature in features
        assert not module.wants_aggregate_means
        assert module.wants_aggregate_medians
        assert not module.wants_aggregate_std
        assert not module.wants_everything
        assert len(module.object_groups) == 5
        for i, (object_name, file_name) in enumerate(
            (
                ("Image", "Image.csv"),
                ("Nuclei", "Nuclei.csv"),
                ("PropCells", "PropCells.csv"),
                ("DistanceCells", "DistanceCells.csv"),
                ("DistCytoplasm", "DistCytoplasm.csv"),
            )
        ):
            group = module.object_groups[i]
            assert not group.previous_file
            assert not group.wants_automatic_file_name
            assert group.name == object_name
            assert group.file_name == file_name

    def test_load_v6():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9434

ExportToSpreadsheet:[module_num:1|svn_version:\'9434\'|variable_revision_number:6|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Tab
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:No
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:Yes
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:Yes
    Calculate the per-image standard deviation values for object measurements?:No
    Output file location:Default Output Folder\x7C//iodine/imaging_analysis/People/Lee
    Export all measurements?:No
    Press button to select measurements to export:Image\x7CFileName_rawGFP,Image\x7CFileName_IllumGFP,Image\x7CFileName_IllumDNA,Image\x7CFileName_rawDNA,Image\x7CMetadata_SBS_Doses,Image\x7CMetadata_Well,Image\x7CMetadata_Controls
    Data to export:Image
    Combine these object measurements with those of the previous object?:No
    File name:Image.csv
    Use the object name for the file name?:No
    Data to export:Nuclei
    Combine these object measurements with those of the previous object?:No
    File name:Nuclei.csv
    Use the object name for the file name?:No
    Data to export:PropCells
    Combine these object measurements with those of the previous object?:No
    File name:PropCells.csv
    Use the object name for the file name?:No
    Data to export:DistanceCells
    Combine these object measurements with those of the previous object?:No
    File name:DistanceCells.csv
    Use the object name for the file name?:No
    Data to export:DistCytoplasm
    Combine these object measurements with those of the previous object?:No
    File name:DistCytoplasm.csv
    Use the object name for the file name?:No

ExportToSpreadsheet:[module_num:2|svn_version:\'9434\'|variable_revision_number:6|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:Yes
    Limit output to a size that is allowed in Excel?:Yes
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:Yes
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:Yes
    Output file location:Default Input Folder\x7C//iodine/imaging_analysis/People/Lee
    Export all measurements?:Yes
    Press button to select measurements to export:Image\x7CFileName_rawGFP,Image\x7CFileName_IllumGFP,Image\x7CFileName_IllumDNA,Image\x7CFileName_rawDNA,Image\x7CMetadata_SBS_Doses,Image\x7CMetadata_Well,Image\x7CMetadata_Controls
    Data to export:Image
    Combine these object measurements with those of the previous object?:Yes
    File name:Image.csv
    Use the object name for the file name?:Yes

ExportToSpreadsheet:[module_num:3|svn_version:\'9434\'|variable_revision_number:6|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:Yes
    Limit output to a size that is allowed in Excel?:Yes
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:Yes
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:Yes
    Output file location:Default Input Folder sub-folder\x7C//iodine/imaging_analysis/People/Lee
    Export all measurements?:Yes
    Press button to select measurements to export:Image\x7CFileName_rawGFP,Image\x7CFileName_IllumGFP,Image\x7CFileName_IllumDNA,Image\x7CFileName_rawDNA,Image\x7CMetadata_SBS_Doses,Image\x7CMetadata_Well,Image\x7CMetadata_Controls
    Data to export:Image
    Combine these object measurements with those of the previous object?:Yes
    File name:Image.csv
    Use the object name for the file name?:Yes

ExportToSpreadsheet:[module_num:4|svn_version:\'9434\'|variable_revision_number:6|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:Yes
    Limit output to a size that is allowed in Excel?:Yes
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:Yes
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:Yes
    Output file location:Default Output Folder sub-folder\x7C//iodine/imaging_analysis/People/Lee
    Export all measurements?:Yes
    Press button to select measurements to export:Image\x7CFileName_rawGFP,Image\x7CFileName_IllumGFP,Image\x7CFileName_IllumDNA,Image\x7CFileName_rawDNA,Image\x7CMetadata_SBS_Doses,Image\x7CMetadata_Well,Image\x7CMetadata_Controls
    Data to export:Image
    Combine these object measurements with those of the previous object?:Yes
    File name:Image.csv
    Use the object name for the file name?:Yes

ExportToSpreadsheet:[module_num:5|svn_version:\'9434\'|variable_revision_number:6|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:Yes
    Limit output to a size that is allowed in Excel?:Yes
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:Yes
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:Yes
    Output file location:Elsewhere...\x7C//iodine/imaging_analysis/People/Lee
    Export all measurements?:Yes
    Press button to select measurements to export:Image\x7CFileName_rawGFP,Image\x7CFileName_IllumGFP,Image\x7CFileName_IllumDNA,Image\x7CFileName_rawDNA,Image\x7CMetadata_SBS_Doses,Image\x7CMetadata_Well,Image\x7CMetadata_Controls
    Data to export:Image
    Combine these object measurements with those of the previous object?:Yes
    File name:Image.csv
    Use the object name for the file name?:Yes
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 5
        module = pipeline.modules()[0]
        assert isinstance(module, E.ExportToSpreadsheet)
        assert module.delimiter == E.DELIMITER_TAB
        assert module.directory.dir_choice == E.DEFAULT_OUTPUT_FOLDER_NAME
        assert module.directory.custom_path == "//iodine/imaging_analysis/People/Lee"
        assert not module.add_metadata
        assert module.pick_columns
        assert all(
            [
                module.columns.get_measurement_object(x) == "Image"
                for x in module.columns.selections
            ]
        )
        assert len(module.columns.selections) == 7
        features = set(
            [
                module.columns.get_measurement_feature(x)
                for x in module.columns.selections
            ]
        )
        for feature in (
            "FileName_rawGFP",
            "FileName_IllumGFP",
            "FileName_IllumDNA",
            "FileName_rawDNA",
            "Metadata_SBS_Doses",
            "Metadata_Well",
            "Metadata_Controls",
        ):
            assert feature in features
        assert not module.wants_aggregate_means
        assert module.wants_aggregate_medians
        assert not module.wants_aggregate_std
        assert not module.wants_everything
        assert len(module.object_groups) == 5
        for i, (object_name, file_name) in enumerate(
            (
                ("Image", "Image.csv"),
                ("Nuclei", "Nuclei.csv"),
                ("PropCells", "PropCells.csv"),
                ("DistanceCells", "DistanceCells.csv"),
                ("DistCytoplasm", "DistCytoplasm.csv"),
            )
        ):
            group = module.object_groups[i]
            assert not group.previous_file
            assert not group.wants_automatic_file_name
            assert group.name == object_name
            assert group.file_name == file_name

        module = pipeline.modules()[1]
        assert isinstance(module, E.ExportToSpreadsheet)
        assert module.delimiter == E.DELIMITER_COMMA
        assert module.directory.dir_choice == E.DEFAULT_INPUT_FOLDER_NAME
        assert module.directory.custom_path == "//iodine/imaging_analysis/People/Lee"
        assert module.add_metadata
        assert not module.pick_columns
        assert module.wants_aggregate_means
        assert not module.wants_aggregate_medians
        assert module.wants_aggregate_std
        assert module.wants_everything
        group = module.object_groups[0]
        assert group.previous_file
        assert group.wants_automatic_file_name

        for module, dir_choice in zip(
            pipeline.modules()[2:],
            (
                E.DEFAULT_INPUT_SUBFOLDER_NAME,
                E.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                E.ABSOLUTE_FOLDER_NAME,
            ),
        ):
            assert isinstance(module, E.ExportToSpreadsheet)
            assert module.directory.dir_choice == dir_choice
        assert module.nan_representation == E.NANS_AS_NANS

    def test_load_v8():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20130503182624
ModuleCount:1
HasImagePlaneDetails:False

ExportToSpreadsheet:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:8|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:No
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:No
    Output file location:Elsewhere...\x7C/imaging/analysis/2005Projects
    Create a GenePattern GCT file?:No
    Select source of sample row name:Image filename
    Select the image to use as the identifier:GFP
    Select the metadata to use as the identifier:Metadata_GeneName
    Export all measurements, using default file names?:Yes
    Press button to select measurements to export:
    Representation of Nan/Inf:Null
    Data to export:Nuclei
    Combine these object measurements with those of the previous object?:No
    File name:Output.csv
    Use the object name for the file name?:Yes

"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert isinstance(module, E.ExportToSpreadsheet)
        assert module.delimiter == E.DELIMITER_COMMA
        assert not module.add_metadata
        assert not module.wants_aggregate_means
        assert not module.wants_aggregate_medians
        assert module.directory.dir_choice == E.ABSOLUTE_FOLDER_NAME
        assert module.directory.custom_path == "/imaging/analysis/2005Projects"
        assert not module.wants_genepattern_file
        assert module.how_to_specify_gene_name == E.GP_NAME_FILENAME
        assert module.use_which_image_for_gene_name == "GFP"
        assert module.gene_name_column == "Metadata_GeneName"
        assert module.wants_everything
        assert module.nan_representation == E.NANS_AS_NULLS
        assert module.object_groups[0].name == "Nuclei"
        assert not module.object_groups[0].previous_file
        assert module.object_groups[0].file_name == "Output.csv"
        assert module.object_groups[0].wants_automatic_file_name

    def test_load_v9():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20130503182624
ModuleCount:1
HasImagePlaneDetails:False

ExportToSpreadsheet:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:9|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select or enter the column delimiter:Comma (",")
    Add image metadata columns to your object data file?:No
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:No
    Output file location:Elsewhere...\x7C/imaging/analysis/2005Projects
    Create a GenePattern GCT file?:No
    Select source of sample row name:Image filename
    Select the image to use as the identifier:GFP
    Select the metadata to use as the identifier:Metadata_GeneName
    Export all measurements, using default file names?:Yes
    Press button to select measurements to export:
    Representation of Nan/Inf:Null
    Data to export:Nuclei
    Combine these object measurements with those of the previous object?:No
    File name:Output.csv
    Use the object name for the file name?:Yes

"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert isinstance(module, E.ExportToSpreadsheet)
        assert module.delimiter == E.DELIMITER_COMMA
        assert not module.add_metadata
        assert not module.wants_aggregate_means
        assert not module.wants_aggregate_medians
        assert module.directory.dir_choice == E.ABSOLUTE_FOLDER_NAME
        assert module.directory.custom_path == "/imaging/analysis/2005Projects"
        assert not module.wants_genepattern_file
        assert module.how_to_specify_gene_name == E.GP_NAME_FILENAME
        assert module.use_which_image_for_gene_name == "GFP"
        assert module.gene_name_column == "Metadata_GeneName"
        assert module.wants_everything
        assert module.nan_representation == E.NANS_AS_NULLS
        assert module.object_groups[0].name == "Nuclei"
        assert not module.object_groups[0].previous_file
        assert module.object_groups[0].file_name == "Output.csv"
        assert module.object_groups[0].wants_automatic_file_name
        assert not module.wants_prefix
        assert module.prefix == "MyExpt_"

    def test_load_v10():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20130503182624
ModuleCount:1
HasImagePlaneDetails:False

ExportToSpreadsheet:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:10|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select or enter the column delimiter:Comma (",")
    Add image metadata columns to your object data file?:No
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:No
    Output file location:Elsewhere...\x7C/imaging/analysis/2005Projects
    Create a GenePattern GCT file?:No
    Select source of sample row name:Image filename
    Select the image to use as the identifier:GFP
    Select the metadata to use as the identifier:Metadata_GeneName
    Export all measurements, using default file names?:Yes
    Press button to select measurements to export:
    Representation of Nan/Inf:Null
    Add a prefix to file names?:Yes
    Filename prefix\x7c:Fred
    Data to export:Nuclei
    Combine these object measurements with those of the previous object?:No
    File name:Output.csv
    Use the object name for the file name?:Yes

"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert isinstance(module, E.ExportToSpreadsheet)
        assert module.delimiter == E.DELIMITER_COMMA
        assert not module.add_metadata
        assert not module.wants_aggregate_means
        assert not module.wants_aggregate_medians
        assert module.directory.dir_choice == E.ABSOLUTE_FOLDER_NAME
        assert module.directory.custom_path == "/imaging/analysis/2005Projects"
        assert not module.wants_genepattern_file
        assert module.how_to_specify_gene_name == E.GP_NAME_FILENAME
        assert module.use_which_image_for_gene_name == "GFP"
        assert module.gene_name_column == "Metadata_GeneName"
        assert module.wants_everything
        assert module.nan_representation == E.NANS_AS_NULLS
        assert module.object_groups[0].name == "Nuclei"
        assert not module.object_groups[0].previous_file
        assert module.object_groups[0].file_name == "Output.csv"
        assert module.object_groups[0].wants_automatic_file_name
        assert module.wants_prefix
        assert module.prefix == "Fred"
        assert module.wants_overwrite_without_warning

    def test_load_v11():
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20130503182624
ModuleCount:1
HasImagePlaneDetails:False

ExportToSpreadsheet:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select or enter the column delimiter:Comma (",")
    Add image metadata columns to your object data file?:No
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:No
    Output file location:Elsewhere...\x7C/imaging/analysis/2005Projects
    Create a GenePattern GCT file?:No
    Select source of sample row name:Image filename
    Select the image to use as the identifier:GFP
    Select the metadata to use as the identifier:Metadata_GeneName
    Export all measurements, using default file names?:Yes
    Press button to select measurements to export:
    Representation of Nan/Inf:Null
    Add a prefix to file names?:Yes
    Filename prefix\x7c:Fred
    Overwrite existing files?:No
    Data to export:Nuclei
    Combine these object measurements with those of the previous object?:No
    File name:Output.csv
    Use the object name for the file name?:Yes

"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert isinstance(module, E.ExportToSpreadsheet)
        assert module.delimiter == E.DELIMITER_COMMA
        assert not module.add_metadata
        assert not module.wants_aggregate_means
        assert not module.wants_aggregate_medians
        assert module.directory.dir_choice == E.ABSOLUTE_FOLDER_NAME
        assert module.directory.custom_path == "/imaging/analysis/2005Projects"
        assert not module.wants_genepattern_file
        assert module.how_to_specify_gene_name == E.GP_NAME_FILENAME
        assert module.use_which_image_for_gene_name == "GFP"
        assert module.gene_name_column == "Metadata_GeneName"
        assert module.wants_everything
        assert module.nan_representation == E.NANS_AS_NULLS
        assert module.object_groups[0].name == "Nuclei"
        assert not module.object_groups[0].previous_file
        assert module.object_groups[0].file_name == "Output.csv"
        assert module.object_groups[0].wants_automatic_file_name
        assert module.wants_prefix
        assert module.prefix == "Fred"
        assert not module.wants_overwrite_without_warning

    def test_no_measurements():
        """Test an image set with objects but no measurements"""
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.object_groups[0].name.value = "my_object"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.wants_prefix.value = False
        m = cpmeas.Measurements()
        m.add_measurement("my_object", "my_measurement", np.zeros((0,)))
        m.add_image_measurement("Count_my_object", 0)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_object")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        fd = open(path, "r")
        try:
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 3
            assert header[2] == "my_measurement"
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()
            del m

    def test_experiment_measurement():
        """Test writing one experiment measurement"""
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.EXPERIMENT
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        m.add_experiment_measurement("my_measurement", "Hello, world")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 2
            assert header[0] == E.EH_KEY
            assert header[1] == E.EH_VALUE
            row = next(reader)
            assert len(row) == 2
            assert row[0] == "my_measurement"
            assert row[1] == "Hello, world"
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            del m
            fd.close()

    def test_two_experiment_measurements():
        """Test writing two experiment measurements"""
        path = os.path.join(output_dir, "%s.csv" % cpmeas.EXPERIMENT)
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.directory.dir_choice = E.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = output_dir
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.EXPERIMENT
        module.object_groups[0].file_name.value = "badfile"
        module.object_groups[0].wants_automatic_file_name.value = True
        m = cpmeas.Measurements(mode="memory")
        m.add_experiment_measurement("my_measurement", "Hello, world")
        m.add_experiment_measurement("my_other_measurement", "Goodbye")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        fd = open(path, "r")
        try:
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            row = next(reader)
            assert len(row) == 2
            assert row[0] == "my_measurement"
            assert row[1] == "Hello, world"
            row = next(reader)
            assert len(row) == 2
            assert row[0] == "my_other_measurement"
            assert row[1] == "Goodbye"
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_img_887_no_experiment_file():
        """Regression test of IMG-887: spirious experiment file

        ExportToSpreadsheet shouldn't generate an experiment file if
        the only measurements are Exit_Status or Complete.
        """
        np.random.seed(14887)
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.directory.dir_choice = E.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = output_dir
        module.wants_everything.value = True
        m = cpmeas.Measurements()
        m.add_experiment_measurement("Exit_Status", "Complete")
        image_measurements = np.random.uniform(size=4)
        m.add_all_measurements(cpmeas.IMAGE, "my_measurement", image_measurements)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        path = os.path.join(output_dir, "Experiment.csv")
        assert not os.path.exists(path)
        path = os.path.join(output_dir, "Image.csv")
        assert os.path.exists(path)

    def test_prefix():
        # Use a prefix, check that file name exists
        prefix = "Foo_"
        np.random.seed(14887)
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = True
        module.prefix.value = prefix
        module.directory.dir_choice = E.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = output_dir
        module.wants_everything.value = True
        m = cpmeas.Measurements()
        image_measurements = np.random.uniform(size=4)
        m.add_all_measurements(cpmeas.IMAGE, "my_measurement", image_measurements)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        path = os.path.join(output_dir, prefix + "Image.csv")
        assert os.path.exists(path)

    def test_image_measurement():
        """Test writing an image measurement"""
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        m.add_image_measurement("my_measurement", "Hello, world")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        fd = open(path, "r")
        try:
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 2
            assert header[0] == "ImageNumber"
            assert header[1] == "my_measurement"
            row = next(reader)
            assert row[0] == "1"
            assert row[1] == "Hello, world"
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_three_by_two_image_measurements():
        """Test writing three image measurements over two image sets"""
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        image_sets = [image_set_list.get_image_set(i) for i in range(2)]
        for i in range(2):
            if i:
                m.next_image_set()
            for j in range(3):
                m.add_image_measurement("measurement_%d" % j, "%d:%d" % (i, j))
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_sets[i],
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        fd = open(path, "r")
        try:
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 4
            assert header[0] == "ImageNumber"
            for i in range(3):
                assert header[i + 1] == "measurement_%d" % i
            for i in range(2):
                row = next(reader)
                assert len(row) == 4
                for j in range(3):
                    assert row[j + 1] == "%d:%d" % (i, j)
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_object_measurement():
        """Test getting a single object measurement"""
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = "my_object"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        np.random.seed(0)
        mvalues = np.random.uniform(size=(1,))
        m.add_measurement("my_object", "my_measurement", mvalues)
        m.add_image_measurement("Count_my_object", 1)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 3
            assert header[0] == "ImageNumber"
            assert header[1] == "ObjectNumber"
            assert header[2] == "my_measurement"
            row = next(reader)
            assert len(row) == 3
            assert round(abs(float(row[2]) - mvalues[0]), 4) == 0
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_three_by_two_object_measurements():
        """Test getting three measurements from two objects"""
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = "my_object"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements(mode="memory")
        np.random.seed(0)
        mvalues = np.random.uniform(size=(2, 3))
        for i in range(3):
            m.add_measurement("my_object", "measurement_%d" % i, mvalues[:, i])
        m.add_image_measurement("Count_my_object", 2)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 5
            assert header[0] == "ImageNumber"
            assert header[1] == "ObjectNumber"
            for i in range(3):
                assert header[i + 2] == "measurement_%d" % i
            for i in range(2):
                row = next(reader)
                assert len(row) == 5
                assert int(row[0]) == 1
                assert int(row[1]) == i + 1
                for j in range(3):
                    assert round(abs(float(row[j + 2]) - mvalues[i, j]), 7) == 0
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_get_measurements_from_two_objects():
        """Get three measurements from four cells and two objects"""
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.add_object_group()
        module.object_groups[0].name.value = "object_0"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.object_groups[1].previous_file.value = True
        module.object_groups[1].name.value = "object_1"
        module.object_groups[1].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        np.random.seed(0)
        # cell, measurement, object
        mvalues = np.random.uniform(size=(4, 3, 2))
        for oidx in range(2):
            for i in range(3):
                m.add_measurement(
                    "object_%d" % oidx, "measurement_%d" % i, mvalues[:, i, oidx]
                )
        m.add_image_measurement("Count_object_0", 4)
        m.add_image_measurement("Count_object_1", 4)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "object_0")
        object_set.add_objects(cpo.Objects(), "object_1")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 8
            for oidx in range(2):
                for i in range(3):
                    assert header[i + oidx * 3 + 2] == "object_%d" % oidx
            header = next(reader)
            assert len(header) == 8
            assert header[0] == "ImageNumber"
            assert header[1] == "ObjectNumber"
            for oidx in range(2):
                for i in range(3):
                    assert header[i + oidx * 3 + 2] == "measurement_%d" % i

            for i in range(4):
                row = next(reader)
                assert len(row) == 8
                assert int(row[0]) == 1
                assert int(row[1]) == i + 1
                for j in range(3):
                    for k in range(2):
                        assert (
                            round(abs(float(row[k * 3 + j + 2]) - mvalues[i, j, k]), 7)
                            == 0
                        )
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_nan_measurements():
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = "my_object"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.nan_representation.value = E.NANS_AS_NANS
        m = cpmeas.Measurements()
        np.random.seed(0)
        mvalues = np.random.uniform(size=(2,))
        mvalues[1] = np.NaN
        m.add_measurement("my_object", "my_measurement", mvalues)
        m.add_image_measurement("Count_my_object", 2)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 3
            assert header[0] == "ImageNumber"
            assert header[1] == "ObjectNumber"
            assert header[2] == "my_measurement"
            row = next(reader)
            assert len(row) == 3
            assert round(abs(float(row[2]) - mvalues[0]), 4) == 0
            row = next(reader)
            assert len(row) == 3
            assert row[2] == str(np.NaN)
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_null_measurements():
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = "my_object"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.nan_representation.value = E.NANS_AS_NULLS
        m = cpmeas.Measurements()
        np.random.seed(0)
        mvalues = np.random.uniform(size=(2,))
        mvalues[1] = np.NaN
        m.add_measurement("my_object", "my_measurement", mvalues)
        m.add_image_measurement("Count_my_object", 2)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 3
            assert header[0] == "ImageNumber"
            assert header[1] == "ObjectNumber"
            assert header[2] == "my_measurement"
            row = next(reader)
            assert len(row) == 3
            assert round(abs(float(row[2]) - mvalues[0]), 4) == 0
            row = next(reader)
            assert len(row) == 3
            assert len(row[2]) == 0
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_nan_image_measurements():
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.wants_aggregate_means.value = True
        module.nan_representation.value = E.NANS_AS_NANS
        m = cpmeas.Measurements()
        m.add_measurement(
            cpmeas.IMAGE,
            "my_image_measurement",
            13,
            image_set_number=1,
            data_type=np.float64,
        )
        mvalues = np.array([np.NaN, np.NaN])
        m.add_measurement(
            OBJECTS_NAME, OBJ_MEAS, mvalues, image_set_number=1, data_type=np.float64
        )
        m.add_measurement(
            cpmeas.IMAGE, "Count_%s" % OBJECTS_NAME, 2, image_set_number=1
        )
        m.add_measurement(
            cpmeas.IMAGE, IMG_MEAS, np.NaN, image_set_number=2, data_type=np.float64
        )
        m.add_measurement(
            cpmeas.IMAGE, "Count_%s" % OBJECTS_NAME, 0, image_set_number=2
        )
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), OBJECTS_NAME)
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        with open(path, "r") as fd:
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            d = dict([(h, i) for i, h in enumerate(header)])
            agg_meas = "Mean_%s_%s" % (OBJECTS_NAME, OBJ_MEAS)
            assert agg_meas in d
            assert IMG_MEAS in d
            row = next(reader)
            value = row[d[agg_meas]]
            assert value == str(np.NaN), "Expected nan %s measurement, got %s" % (
                agg_meas,
                value,
            )
            assert float(row[d[IMG_MEAS]]) == 13
            row = next(reader)
            for meas in agg_meas, IMG_MEAS:
                value = row[d[meas]]
                assert value == str(np.NaN), "Expected nan %s measurement, got %s" % (
                    meas,
                    value,
                )
            with pytest.raises(StopIteration):
                reader.__next__()

    def test_null_image_measurements():
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.wants_aggregate_means.value = True
        module.nan_representation.value = E.NANS_AS_NULLS
        m = cpmeas.Measurements()
        m.add_measurement(
            cpmeas.IMAGE,
            "my_image_measurement",
            13,
            image_set_number=1,
            data_type=np.float64,
        )
        mvalues = np.array([np.NaN, np.NaN])
        m.add_measurement(
            OBJECTS_NAME, OBJ_MEAS, mvalues, image_set_number=1, data_type=np.float64
        )
        m.add_measurement(
            cpmeas.IMAGE, "Count_%s" % OBJECTS_NAME, 2, image_set_number=1
        )
        m.add_measurement(
            cpmeas.IMAGE, IMG_MEAS, np.NaN, image_set_number=2, data_type=np.float64
        )
        m.add_measurement(
            cpmeas.IMAGE, "Count_%s" % OBJECTS_NAME, 0, image_set_number=2
        )
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), OBJECTS_NAME)
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        with open(path, "r") as fd:
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            d = dict([(h, i) for i, h in enumerate(header)])
            agg_meas = "Mean_%s_%s" % (OBJECTS_NAME, OBJ_MEAS)
            assert agg_meas in d
            assert IMG_MEAS in d
            row = next(reader)
            value = row[d[agg_meas]]
            assert len(value) == 0, "Expected null %s measurement, got %s" % (
                agg_meas,
                value,
            )
            assert float(row[d[IMG_MEAS]]) == 13
            row = next(reader)
            for meas in agg_meas, IMG_MEAS:
                value = row[d[meas]]
                assert len(value) == 0, "Expected null %s measurement, got %s" % (
                    meas,
                    value,
                )
            with pytest.raises(StopIteration):
                reader.__next__()

    def test_blob_image_measurements():
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.wants_aggregate_means.value = False
        m = cpmeas.Measurements()
        r = np.random.RandomState()
        r.seed(38)
        my_blob = r.randint(0, 256, 100).astype(np.uint8)
        m.add_measurement(
            cpmeas.IMAGE, IMG_MEAS, my_blob, image_set_number=1, data_type=np.uint8
        )
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        with open(path, "r") as fd:
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            d = dict([(h, i) for i, h in enumerate(header)])
            assert IMG_MEAS in d
            row = next(reader)
            data = base64.b64decode(row[d[IMG_MEAS]])
            value = np.frombuffer(data, np.uint8)
            np.testing.assert_array_equal(value, my_blob)

    def test_blob_experiment_measurements():
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.EXPERIMENT
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.wants_aggregate_means.value = False
        m = cpmeas.Measurements()
        r = np.random.RandomState()
        r.seed(38)
        my_blob = r.randint(0, 256, 100).astype(np.uint8)
        m.add_measurement(
            cpmeas.EXPERIMENT, IMG_MEAS, my_blob, image_set_number=1, data_type=np.uint8
        )
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        with open(path, "r") as fd:
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            for feature, value in reader:
                if feature == IMG_MEAS:
                    data = base64.b64decode(value)
                    value = np.frombuffer(data, np.uint8)
                    np.testing.assert_array_equal(value, my_blob)
                    break
            else:
                fail("Could not find %s in experiment CSV" % IMG_MEAS)

    def test_01_object_with_metadata():
        """Test writing objects with 2 pairs of 2 image sets w same metadata"""
        # +++backslash+++ here because Windows and join don't do well
        # if you have the raw backslash
        path = os.path.join(output_dir, "+++backslash+++g<tag>.csv")
        path = path.replace("\\", "\\\\")
        path = path.replace("+++backslash+++", "\\")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = "my_object"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        np.random.seed(0)
        mvalues = np.random.uniform(size=(4,))
        image_set_list = cpi.ImageSetList()
        for index, measurement, metadata in zip(
            list(range(4)), mvalues, ("foo", "bar", "bar", "foo")
        ):
            image_set = image_set_list.get_image_set(index)
            m.add_measurement("my_object", "my_measurement", np.array([measurement]))
            m.add_image_measurement("Metadata_tag", metadata)
            m.add_image_measurement("Count_my_object", 1)
            if index < 3:
                m.next_image_set()
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        for i in range(4):
            module.post_run(workspace)
        for file_name, value_indexes in (("foo.csv", (0, 3)), ("bar.csv", (1, 2))):
            path = os.path.join(output_dir, file_name)
            fd = open(path, "r")
            try:
                reader = csv.reader(fd, delimiter=module.delimiter_char)
                header = next(reader)
                assert len(header) == 3
                assert header[0] == "ImageNumber"
                assert header[1] == "ObjectNumber"
                assert header[2] == "my_measurement"
                for value_index in value_indexes:
                    row = next(reader)
                    assert len(row) == 3
                    assert int(row[0]) == value_index + 1
                    assert int(row[1]) == 1
                    assert round(abs(float(row[2]) - mvalues[value_index]), 4) == 0
                with pytest.raises(StopIteration):
                    reader.__next__()
            finally:
                fd.close()

    def test_02_object_with_path_metadata():
        #
        # Regression test of issue #1142
        #
        # +++backslash+++ here because Windows and join don't do well
        # if you have the raw backslash
        path = os.path.join(output_dir, "+++backslash+++g<tag>")
        path = path.replace("\\", "\\\\")
        path = path.replace("+++backslash+++", "\\")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = True
        module.wants_prefix.value = False
        module.directory.dir_choice = E.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = path
        m = cpmeas.Measurements()
        np.random.seed(0)
        mvalues = np.random.uniform(size=(4,))
        image_set_list = cpi.ImageSetList()
        for index, measurement, metadata in zip(
            list(range(4)), mvalues, ("foo", "bar", "bar", "foo")
        ):
            image_set = image_set_list.get_image_set(index)
            m.add_measurement("my_object", "my_measurement", np.array([measurement]))
            m.add_image_measurement("Metadata_tag", metadata)
            m.add_image_measurement("Count_my_object", 1)
            if index < 3:
                m.next_image_set()
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        for dir_name, value_indexes in (("foo", (0, 3)), ("bar", (1, 2))):
            path = os.path.join(output_dir, dir_name, "my_object.csv")
            fd = open(path, "r")
            try:
                reader = csv.reader(fd, delimiter=module.delimiter_char)
                header = next(reader)
                assert len(header) == 3
                assert header[0] == "ImageNumber"
                assert header[1] == "ObjectNumber"
                assert header[2] == "my_measurement"
                for value_index in value_indexes:
                    row = next(reader)
                    assert len(row) == 3
                    assert int(row[0]) == value_index + 1
                    assert int(row[1]) == 1
                    assert round(abs(float(row[2]) - mvalues[value_index]), 4) == 0
                with pytest.raises(StopIteration):
                    reader.__next__()
            finally:
                fd.close()

    def test_image_with_metadata():
        """Test writing image data with 2 pairs of 2 image sets w same metadata"""
        path = os.path.join(output_dir, "+++backslash+++g<tag>.csv")
        path = path.replace("\\", "\\\\")
        path = path.replace("+++backslash+++", "\\")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        np.random.seed(0)
        mvalues = np.random.uniform(size=(4,))
        image_set_list = cpi.ImageSetList()
        for index, measurement, metadata in zip(
            list(range(4)), mvalues, ("foo", "bar", "bar", "foo")
        ):
            image_set = image_set_list.get_image_set(index)
            m.add_image_measurement("my_measurement", measurement)
            m.add_image_measurement("Metadata_tag", metadata)
            if index < 3:
                m.next_image_set()
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        for i in range(4):
            module.post_run(workspace)
        for file_name, value_indexes in (("foo.csv", (0, 3)), ("bar.csv", (1, 2))):
            path = os.path.join(output_dir, file_name)
            fd = open(path, "r")
            try:
                reader = csv.reader(fd, delimiter=module.delimiter_char)
                header = next(reader)
                assert len(header) == 3
                d = {}
                assert "ImageNumber" in header
                assert "my_measurement" in header
                assert "Metadata_tag" in header
                for caption, index in zip(header, list(range(3))):
                    d[caption] = index
                for value_index in value_indexes:
                    row = next(reader)
                    assert len(row) == 3
                    assert (
                        round(
                            abs(float(row[d["my_measurement"]]) - mvalues[value_index]),
                            4,
                        )
                        == 0
                    )
                with pytest.raises(StopIteration):
                    reader.__next__()
            finally:
                fd.close()

    def test_image_with_path_metadata():
        """Test writing image data with 2 pairs of 2 image sets w same metadata"""
        path = os.path.join(output_dir, "+++backslash+++g<tag>")
        path = path.replace("\\", "\\\\")
        path = path.replace("+++backslash+++", "\\")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.directory.dir_choice = E.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = path
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = "output.csv"
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        np.random.seed(0)
        mvalues = np.random.uniform(size=(4,))
        image_set_list = cpi.ImageSetList()
        metadata_values = ("foo", "bar", "bar", "foo")
        for index, (measurement, metadata) in enumerate(zip(mvalues, metadata_values)):
            image_set = image_set_list.get_image_set(index)
            m.add_image_measurement("my_measurement", measurement)
            m.add_image_measurement("Metadata_tag", metadata)
            if index < 3:
                m.next_image_set()
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        for path_name, value_indexes in (("foo", (0, 3)), ("bar", (1, 2))):
            path = os.path.join(output_dir, path_name, "output.csv")
            fd = open(path, "r")
            try:
                reader = csv.reader(fd, delimiter=module.delimiter_char)
                header = next(reader)
                assert len(header) == 3
                d = {}
                assert "ImageNumber" in header
                assert "my_measurement" in header
                assert "Metadata_tag" in header
                for caption, index in zip(header, list(range(3))):
                    d[caption] = index
                for value_index in value_indexes:
                    row = next(reader)
                    assert len(row) == 3
                    assert (
                        round(
                            abs(float(row[d["my_measurement"]]) - mvalues[value_index]),
                            4,
                        )
                        == 0
                    )
                with pytest.raises(StopIteration):
                    reader.__next__()
            finally:
                fd.close()

    def test_image_measurement_custom_directory():
        """Test writing an image measurement"""
        path = os.path.join(output_dir, "my_dir", "my_file.csv")
        cpprefs.set_headless()
        cpprefs.set_default_output_directory(output_dir)
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.directory.dir_choice = E.DEFAULT_OUTPUT_SUBFOLDER_NAME
        module.directory.custom_path = "./my_dir"
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = "my_file.csv"
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements(mode="memory")
        m.add_image_measurement("my_measurement", "Hello, world")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 2
            assert header[0] == "ImageNumber"
            assert header[1] == "my_measurement"
            row = next(reader)
            assert row[0] == "1"
            assert row[1] == "Hello, world"
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_unicode_image_metadata():
        """Write image measurements containing unicode characters"""
        path = os.path.join(output_dir, "my_dir", "my_file.csv")
        cpprefs.set_headless()
        cpprefs.set_default_output_directory(output_dir)
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.directory.dir_choice = E.DEFAULT_OUTPUT_SUBFOLDER_NAME
        module.directory.custom_path = "./my_dir"
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = "my_file.csv"
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements(mode="memory")
        metadata_value = "\\u2211(Hello, world)"
        m.add_image_measurement("my_measurement", metadata_value)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 2
            assert header[0] == "ImageNumber"
            assert header[1] == "my_measurement"
            row = next(reader)
            assert row[0] == "1"
            assert row[1].decode("utf8") == metadata_value
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_overwrite_files_everything():
        m = make_measurements()
        pipeline = make_measurements_pipeline(m)
        #
        # This will give ExportToSpreadsheet some objects to deal with
        #
        idp = identifyprimaryobjects.IdentifyPrimaryObjects()
        idp.module_num = 1
        idp.y_name.value = OBJECTS_NAME
        pipeline.add_module(idp)

        module = E.ExportToSpreadsheet()
        module.wants_everything.value = True
        module.directory.dir_choice = E.cps.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = output_dir
        module.set_module_num(2)
        pipeline.add_module(module)

        workspace = cpw.Workspace(pipeline, module, m, None, m, None)
        for object_name in (cpmeas.EXPERIMENT, cpmeas.IMAGE, OBJECTS_NAME):
            file_name = module.make_objects_file_name(object_name, workspace, 1)
            with open(file_name, "w") as fd:
                fd.write("Hello, world.")
            module.wants_overwrite_without_warning.value = True
            assert module.prepare_run(workspace)
            module.wants_overwrite_without_warning.value = False
            assert not module.prepare_run(workspace)
            os.remove(file_name)
            assert module.prepare_run(workspace)

    def test_overwrite_files_group():
        m = make_measurements(dict(Metadata_tag=["foo", "bar"]))
        pipeline = make_measurements_pipeline(m)
        #
        # This will give ExportToSpreadsheet some objects to deal with
        #
        idp = identifyprimaryobjects.IdentifyPrimaryObjects()
        idp.module_num = 1
        idp.y_name.value = OBJECTS_NAME
        pipeline.add_module(idp)

        module = E.ExportToSpreadsheet()
        module.wants_everything.value = False
        module.directory.dir_choice = E.cps.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = output_dir
        g = module.object_groups[0]
        g.name.value = OBJECTS_NAME
        g.wants_automatic_file_name.value = False
        g.file_name.value = "\\g<tag>.csv"
        module.set_module_num(2)
        pipeline.add_module(module)

        workspace = cpw.Workspace(pipeline, module, m, None, m, None)

        for image_number in m.get_image_numbers():
            file_name = module.make_objects_file_name(
                OBJECTS_NAME, workspace, image_number, g
            )
            with open(file_name, "w") as fd:
                fd.write("Hello, world.")
            module.wants_overwrite_without_warning.value = True
            assert module.prepare_run(workspace)
            module.wants_overwrite_without_warning.value = False
            assert not module.prepare_run(workspace)
            os.remove(file_name)
            assert module.prepare_run(workspace)

    def test_aggregate_image_columns():
        """Test output of aggregate object data for images"""
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.wants_aggregate_means.value = True
        module.wants_aggregate_medians.value = True
        module.wants_aggregate_std.value = True
        m = cpmeas.Measurements()
        m.add_image_measurement("Count_my_objects", 6)
        np.random.seed(0)
        data = np.random.uniform(size=(6,))
        m.add_measurement("my_objects", "my_measurement", data)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        fd = open(path, "r")
        try:
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == len(cpmeas.AGG_NAMES) + 2
            d = {}
            for index, caption in enumerate(header):
                d[caption] = index

            row = next(reader)
            assert row[d["Count_my_objects"]] == "6"
            for agg in cpmeas.AGG_NAMES:
                value = (
                    np.mean(data)
                    if agg == cpmeas.AGG_MEAN
                    else np.std(data)
                    if agg == cpmeas.AGG_STD_DEV
                    else np.median(data)
                    if agg == cpmeas.AGG_MEDIAN
                    else np.NAN
                )
                assert (
                    round(
                        abs(
                            float(row[d["%s_my_objects_my_measurement" % agg]]) - value
                        ),
                        7,
                    )
                    == 0
                )
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_no_aggregate_image_columns():
        """Test output of aggregate object data for images"""
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.wants_aggregate_means.value = False
        module.wants_aggregate_medians.value = False
        module.wants_aggregate_std.value = False
        m = cpmeas.Measurements(mode="memory")
        m.add_image_measurement("Count_my_objects", 6)
        np.random.seed(0)
        data = np.random.uniform(size=(6,))
        m.add_measurement("my_objects", "my_measurement", data)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 2
            d = {}
            for index, caption in enumerate(header):
                d[caption] = index
            row = next(reader)
            assert row[d["Count_my_objects"]] == "6"
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            del m
            fd.close()

    def test_aggregate_and_filtered():
        """Regression test of IMG-987

        A bug in ExportToSpreadsheet caused it to fail to write any
        aggregate object measurements if measurements were filtered by
        pick_columns.
        """
        image_path = os.path.join(output_dir, "my_image_file.csv")
        object_path = os.path.join(output_dir, "my_object_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = image_path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.add_object_group()
        module.object_groups[1].name.value = "my_objects"
        module.object_groups[1].file_name.value = object_path
        module.object_groups[1].wants_automatic_file_name.value = False
        module.wants_aggregate_means.value = True
        module.wants_aggregate_medians.value = False
        module.wants_aggregate_std.value = False
        module.pick_columns.value = True
        columns = [
            module.columns.make_measurement_choice(ob, feature)
            for ob, feature in (
                (cpmeas.IMAGE, "ImageNumber"),
                (cpmeas.IMAGE, "Count_my_objects"),
                (cpmeas.IMAGE, "first_measurement"),
                ("my_objects", "my_measurement"),
                ("my_objects", "ImageNumber"),
                ("my_objects", "Number_Object_Number"),
            )
        ]
        module.columns.value = module.columns.get_value_string(columns)

        m = cpmeas.Measurements()
        np.random.seed(0)
        data = np.random.uniform(size=(6,))
        m.add_image_measurement("Count_my_objects", 6)
        m.add_image_measurement("first_measurement", np.sum(data))
        m.add_image_measurement("another_measurement", 43.2)
        m.add_measurement("my_objects", "Number_Object_Number", np.arange(1, 7))
        m.add_measurement("my_objects", "my_measurement", data)
        m.add_measurement(
            "my_objects", "my_filtered_measurement", np.random.uniform(size=(6,))
        )
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(image_path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 4
            expected_image_columns = (
                "ImageNumber",
                "Count_my_objects",
                "first_measurement",
                "Mean_my_objects_my_measurement",
            )
            d = {}
            for index, caption in enumerate(header):
                assert caption in expected_image_columns
                d[caption] = index
            row = next(reader)
            assert row[d["ImageNumber"]] == "1"
            assert row[d["Count_my_objects"]] == "6"
            assert round(abs(float(row[d["first_measurement"]]) - np.sum(data)), 7) == 0
            assert (
                round(
                    abs(
                        float(row[d["Mean_my_objects_my_measurement"]]) - np.mean(data)
                    ),
                    7,
                )
                == 0
            )
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()
        try:
            fd = open(object_path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 4
            expected_object_columns = (
                "ImageNumber",
                "ObjectNumber",
                "Number_Object_Number",
                "my_measurement",
            )
            d = {}
            for index, caption in enumerate(header):
                assert caption in expected_object_columns
                d[caption] = index
            for index, row in enumerate(reader):
                assert row[d["ImageNumber"]] == "1"
                assert int(row[d["ObjectNumber"]]) == index + 1
                # all object values get written as floats
                assert int(float(row[d["Number_Object_Number"]])) == index + 1
                assert round(abs(float(row[d["my_measurement"]]) - data[index]), 7) == 0
        finally:
            fd.close()

    def test_image_number():
        # Regression test of issue #1139
        # Always output the ImageNumber column in Image.csv

        image_path = os.path.join(output_dir, "my_image_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = image_path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.wants_aggregate_means.value = False
        module.wants_aggregate_medians.value = False
        module.wants_aggregate_std.value = False
        module.pick_columns.value = True
        columns = [
            module.columns.make_measurement_choice(ob, feature)
            for ob, feature in ((cpmeas.IMAGE, "first_measurement"),)
        ]
        module.columns.value = module.columns.get_value_string(columns)

        m = cpmeas.Measurements()
        np.random.seed(0)
        data = np.random.uniform(size=(6,))
        m.add_image_measurement("first_measurement", np.sum(data))
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(image_path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 2
            expected_image_columns = ("ImageNumber", "first_measurement")
            d = {}
            for index, caption in enumerate(header):
                assert caption in expected_image_columns
                d[caption] = index
        finally:
            fd.close()

    def test_image_index_columns():
        """Test presence of index column"""
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        data = (
            "The reverse side also has a reverse side. (Japanese proverb)",
            "When I was younger, I could remember anything, whether it had happened or not. (Mark Twain)",
            "A thing worth having is a thing worth cheating for. (W.C. Fields)",
        )
        for i in range(len(data)):
            image_set = image_set_list.get_image_set(i)
            m.add_image_measurement("quotation", data[i])
            if i < len(data) - 1:
                m.next_image_set()
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 2
            assert header[0] == E.IMAGE_NUMBER
            assert header[1] == "quotation"
            for i in range(len(data)):
                row = next(reader)
                assert int(row[0]) == i + 1
                assert row[1] == data[i]
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_object_index_columns():
        """Test presence of image and object index columns"""
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = "my_objects"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        np.random.seed(0)
        # Three images with four objects each
        mvalues = np.random.uniform(size=(3, 4))
        for image_idx in range(mvalues.shape[0]):
            if image_idx:
                m.next_image_set()
            m.add_image_measurement("Count_my_objects", mvalues.shape[1])
            m.add_measurement("my_objects", "my_measurement", mvalues[image_idx, :])
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 3
            assert header[0] == E.IMAGE_NUMBER
            assert header[1] == E.OBJECT_NUMBER
            assert header[2] == "my_measurement"
            for image_idx in range(mvalues.shape[0]):
                for object_idx in range(mvalues.shape[1]):
                    row = next(reader)
                    assert len(row) == 3
                    assert int(row[0]) == image_idx + 1
                    assert int(row[1]) == object_idx + 1
                    assert (
                        round(abs(float(row[2]) - mvalues[image_idx, object_idx]), 4)
                        == 0
                    )
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_object_metadata_columns():
        """Test addition of image metadata columns to an object metadata file"""
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = "my_objects"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.add_metadata.value = True
        m = cpmeas.Measurements()
        np.random.seed(0)
        # Three images with four objects each
        mvalues = np.random.uniform(size=(3, 4))
        for image_idx in range(mvalues.shape[0]):
            if image_idx:
                m.next_image_set()
            m.add_image_measurement("Count_my_objects", mvalues.shape[1])
            m.add_image_measurement("Metadata_Plate", "P-X9TRG")
            m.add_image_measurement("Metadata_Well", "C0%d" % (image_idx + 1))
            m.add_measurement("my_objects", "my_measurement", mvalues[image_idx, :])
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 5
            d = {}
            for index, column in enumerate(header):
                d[column] = index
            assert "Metadata_Plate" in d
            assert "Metadata_Well" in d
            assert "my_measurement" in d
            for image_idx in range(mvalues.shape[0]):
                for object_idx in range(mvalues.shape[1]):
                    row = next(reader)
                    assert len(row) == 5
                    assert row[d["Metadata_Plate"]] == "P-X9TRG"
                    assert row[d["Metadata_Well"]] == "C0%d" % (image_idx + 1)
                    assert (
                        round(
                            abs(
                                float(row[d["my_measurement"]])
                                - mvalues[image_idx, object_idx]
                            ),
                            4,
                        )
                        == 0
                    )
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_missing_measurements():
        """Make sure ExportToSpreadsheet can continue when measurements are missing

        Regression test of IMG-361
        Take measurements for 3 image sets, some measurements missing
        from the middle one.
        """
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = "my_objects"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.add_metadata.value = True
        m = cpmeas.Measurements()
        np.random.seed(0)
        # Three images with four objects each
        mvalues = np.random.uniform(size=(3, 4))
        for image_idx in range(mvalues.shape[0]):
            if image_idx:
                m.next_image_set()
            m.add_image_measurement("Count_my_objects", mvalues.shape[1])
            if image_idx != 1:
                m.add_image_measurement("my_measurement", 100)
                m.add_measurement("my_objects", "my_measurement", mvalues[image_idx, :])
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            make_measurements_pipeline(m),
            module,
            image_set,
            object_set,
            m,
            image_set_list,
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            assert len(header) == 3
            d = {}
            for index, column in enumerate(header):
                d[column] = index
            assert "my_measurement" in d
            for image_idx in range(3):
                for object_idx in range(mvalues.shape[1]):
                    row = next(reader)
                    assert len(row) == 3
                    if image_idx == 1:
                        assert row[d["my_measurement"]] == str(np.NAN)
                    else:
                        assert (
                            round(
                                abs(
                                    float(row[d["my_measurement"]])
                                    - mvalues[image_idx, object_idx]
                                ),
                                4,
                            )
                            == 0
                        )
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def test_missing_column_measurements():
        # Regression test of issue 1293:
        # pipeline.get_column_measurements reports a measurement
        # The measurement isn't made (e.g., FlagImages)
        # ExportToSpreadsheet should put a column of all NaNs, even if
        # no image set makes the measurement
        #
        path = os.path.join(output_dir, "my_file.csv")
        pipeline = cpp.Pipeline()
        module = identifyprimaryobjects.IdentifyPrimaryObjects()
        module.set_module_num(1)
        pipeline.add_module(module)
        module.x_name.value = "my_image"
        module.y_name.value = OBJECTS_NAME
        module = E.ExportToSpreadsheet()
        module.set_module_num(2)
        pipeline.add_module(module)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.nan_representation.value = E.NANS_AS_NANS
        module.object_groups[0].name.value = OBJECTS_NAME
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.add_metadata.value = False
        m = cpmeas.Measurements()
        m[cpmeas.IMAGE, cpmeas.GROUP_NUMBER, 1] = 1
        m[cpmeas.IMAGE, cpmeas.GROUP_INDEX, 1] = 1
        m[cpmeas.IMAGE, "_".join((C_COUNT, OBJECTS_NAME)), 1] = 3
        m[OBJECTS_NAME, M_LOCATION_CENTER_X, 1] = np.array([1, 4, 9], float)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(
            pipeline, module, image_set, object_set, m, image_set_list
        )
        module.post_run(workspace)
        try:
            fd = open(path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = next(reader)
            d = {}
            for index, column in enumerate(header):
                d[column] = index
            assert M_LOCATION_CENTER_X in d
            assert M_LOCATION_CENTER_Y in d
            for i in range(3):
                row = next(reader)
                x = row[d[M_LOCATION_CENTER_X]]
                assert float(x) == (i + 1) ** 2
                y = row[d[M_LOCATION_CENTER_Y]]
                assert y.lower() == "nan"
            with pytest.raises(StopIteration):
                reader.__next__()
        finally:
            fd.close()

    def make_pipeline(csv_text):
        import cellprofiler.modules.loaddata as L

        handle, name = tempfile.mkstemp("csv")
        fd = os.fdopen(handle, "w")
        fd.write(csv_text)
        fd.close()
        csv_path, csv_file = os.path.split(name)
        module = L.LoadText()
        module.csv_directory.dir_choice = cellprofiler.setting.ABSOLUTE_FOLDER_NAME
        module.csv_directory.custom_path = csv_path
        module.csv_file_name.value = csv_file
        module.set_module_num(1)
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)

        def error_callback(caller, event):
            assert not isinstance(event, cpp.RunExceptionEvent)

        pipeline.add_listener(error_callback)
        return pipeline, module, name

    def make_measurements_pipeline(m):
        """Pipeline reports measurements via get_measurement_columns"""
        assert isinstance(m, cpmeas.Measurements)
        columns = []
        if len(m.get_image_numbers()) > 0:
            image_number = m.get_image_numbers()[0]
        else:
            image_number = None
        for object_name in m.get_object_names():
            for feature in m.get_feature_names(object_name):
                if object_name == cpmeas.EXPERIMENT:
                    columns.append((object_name, feature, cpmeas.COLTYPE_VARCHAR))
                elif image_number is not None:
                    data = m[object_name, feature, image_number]
                    if isinstance(data, six.string_types):
                        columns.append((object_name, feature, cpmeas.COLTYPE_VARCHAR))
                    else:
                        columns.append((object_name, feature, cpmeas.COLTYPE_FLOAT))

        class MPipeline(cpp.Pipeline):
            def get_measurement_columns(terminating_module=None):
                return columns

        return MPipeline()

    def make_measurements(d=None):
        """Make a measurements object

        d - a dictionary whose keywords are the measurement names and whose
            values are sequences of measurement values per image set
        """
        if d is None:
            d = {cpmeas.GROUP_NUMBER: [0], cpmeas.GROUP_INDEX: [0]}
        m = cpmeas.Measurements()
        for k, v in list(d.items()):
            m[cpmeas.IMAGE, k, np.arange(len(v)) + 1] = v
        image_numbers = m.get_image_numbers()
        if cpmeas.GROUP_NUMBER not in d:
            m[cpmeas.IMAGE, cpmeas.GROUP_NUMBER, image_numbers] = [0] * len(
                image_numbers
            )
        if cpmeas.GROUP_INDEX not in d:
            m[cpmeas.IMAGE, cpmeas.GROUP_INDEX, image_numbers] = np.arange(
                len(image_numbers)
            )
        return m

    def add_gct_settings(output_csv_filename):
        module = E.ExportToSpreadsheet()
        module.set_module_num(2)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = output_csv_filename
        module.object_groups[0].wants_automatic_file_name.value = False
        module.wants_aggregate_means.value = False
        module.wants_aggregate_medians.value = False
        module.wants_aggregate_std.value = False
        module.wants_genepattern_file.value = True
        return module

    def test_basic_gct_check():
        # LoadData with data
        maybe_download_sbs()
        input_dir = os.path.join(example_images_directory(), "ExampleSBSImages")
        metadata_name = "Metadata_Bar"
        info = (
            "Image_FileName_Foo",
            "Image_PathName_Foo",
            metadata_name,
            input_dir,
            input_dir,
        )
        csv_text = (
            """"%s","%s","%s"
    "Channel1-01-A-01.tif","%s","Hi"
    "Channel1-02-A-02.tif","%s","Hello"
    """
            % info
        )
        pipeline, module, input_filename = make_pipeline(csv_text)

        output_csv_filename = os.path.join(tempfile.mkdtemp(), "my_file.csv")

        # ExportToSpreadsheet
        module = add_gct_settings(output_csv_filename)
        module.how_to_specify_gene_name.value = "Image filename"
        module.use_which_image_for_gene_name.value = "Foo"
        pipeline.add_module(module)

        try:
            m = pipeline.run()
            assert isinstance(m, cpmeas.Measurements)
            p, n = os.path.splitext(output_csv_filename)
            output_gct_filename = p + ".gct"
            fd = open(output_gct_filename, "r")
            reader = csv.reader(fd, delimiter="\t")
            row = next(reader)
            assert len(row) == 1
            assert row[0] == "#1.2"
            row = next(reader)
            assert len(row) == 2
            assert row[0] == "2"
            assert row[1] == "1"
            row = next(reader)
            assert len(row) == 3
            assert row[0].lower() == "name"
            assert row[1].lower() == "description"
            assert row[2] == metadata_name
            row = next(reader)
            assert row[1] == input_dir
        finally:
            try:
                os.remove(input_filename)
                os.remove(output_csv_filename)
            except:
                print("Failed to clean up files")

    def test_make_gct_file_with_filename():
        maybe_download_sbs()
        # LoadData with data
        input_dir = os.path.join(example_images_directory(), "ExampleSBSImages")
        metadata_name = "Metadata_Bar"
        info = (
            "Image_FileName_Foo",
            "Image_PathName_Foo",
            metadata_name,
            input_dir,
            input_dir,
        )
        csv_text = (
            """"%s","%s","%s"
    "Channel1-01-A-01.tif","%s","Hi"
    "Channel1-02-A-02.tif","%s","Hello"
    """
            % info
        )
        pipeline, module, input_filename = make_pipeline(csv_text)

        output_csv_filename = os.path.join(tempfile.mkdtemp(), "my_file.csv")

        # ExportToSpreadsheet
        module = add_gct_settings(output_csv_filename)
        module.how_to_specify_gene_name.value = "Image filename"
        module.use_which_image_for_gene_name.value = "Foo"
        pipeline.add_module(module)

        try:
            m = pipeline.run()
            assert isinstance(m, cpmeas.Measurements)
            p, n = os.path.splitext(output_csv_filename)
            output_gct_filename = p + ".gct"
            fd = open(output_gct_filename, "r")
            reader = csv.reader(fd, delimiter="\t")
            row = next(reader)
            row = next(reader)
            row = next(reader)
            row = next(reader)
            assert row[0] == "Channel1-01-A-01.tif"
            row = next(reader)
            assert row[0] == "Channel1-02-A-02.tif"
            fd.close()
        finally:
            os.remove(input_filename)
            os.remove(output_csv_filename)

    def test_make_gct_file_with_metadata():
        maybe_download_sbs()

        # LoadData with data
        input_dir = os.path.join(example_images_directory(), "ExampleSBSImages")
        metadata_name = "Metadata_Bar"
        info = (
            "Image_FileName_Foo",
            "Image_PathName_Foo",
            metadata_name,
            input_dir,
            input_dir,
        )
        csv_text = (
            """"%s","%s","%s"
    "Channel1-01-A-01.tif","%s","Hi"
    "Channel1-02-A-02.tif","%s","Hello"
    """
            % info
        )
        pipeline, module, input_filename = make_pipeline(csv_text)

        output_csv_filename = os.path.join(tempfile.mkdtemp(), "my_file.csv")

        # ExportToSpreadsheet
        module = add_gct_settings(output_csv_filename)
        module.how_to_specify_gene_name.value = "Metadata"
        module.gene_name_column.value = "Metadata_Bar"
        pipeline.add_module(module)

        try:
            m = pipeline.run()
            assert isinstance(m, cpmeas.Measurements)
            p, n = os.path.splitext(output_csv_filename)
            output_gct_filename = p + ".gct"
            fd = open(output_gct_filename, "r")
            reader = csv.reader(fd, delimiter="\t")
            row = next(reader)
            row = next(reader)
            row = next(reader)
            row = next(reader)
            assert row[0] == "Hi"
            row = next(reader)
            assert row[0] == "Hello"
            fd.close()
        finally:
            os.remove(input_filename)
            os.remove(output_csv_filename)

    def test_test_overwrite_gct_file():
        output_csv_filename = os.path.join(output_dir, "%s.gct" % cpmeas.IMAGE)
        m = make_measurements()
        pipeline = make_measurements_pipeline(m)
        module = E.ExportToSpreadsheet()
        module.wants_genepattern_file.value = True
        module.directory.dir_choice = E.cps.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = output_dir
        module.wants_prefix.value = False
        module.set_module_num(1)
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, m, None, m, None)
        assert output_csv_filename == module.make_gct_file_name(workspace, 1)

        assert module.prepare_run(workspace)
        with open(output_csv_filename, "w") as fd:
            fd.write("Hello, world.\n")
        module.wants_overwrite_without_warning.value = False
        assert not module.prepare_run(workspace)

    def test_relationships_file():
        r = np.random.RandomState()
        r.seed(91)
        path = os.path.join(output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.set_module_num(1)
        module.wants_everything.value = False
        module.wants_prefix.value = False
        module.object_groups[0].name.value = E.OBJECT_RELATIONSHIPS
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        for i in range(0, 10):
            image_set = image_set_list.get_image_set(i)
            m.add_image_measurement(cpp.IMAGE_NUMBER, i + 1)
            m.add_image_measurement(cpp.GROUP_NUMBER, 1)
            m.add_image_measurement(cpp.GROUP_INDEX, i + 1)
            if i < 9:
                m.next_image_set()
        my_relationship = "BlahBlah"
        my_object_name1 = "ABC"
        my_object_name2 = "DEF"
        my_image_numbers1 = r.randint(1, 10, size=10)
        my_object_numbers1 = r.randint(1, 10, size=10)
        my_image_numbers2 = r.randint(1, 10, size=10)
        my_object_numbers2 = r.randint(1, 10, size=10)
        m.add_relate_measurement(
            1,
            my_relationship,
            my_object_name1,
            my_object_name2,
            my_image_numbers1,
            my_object_numbers1,
            my_image_numbers2,
            my_object_numbers2,
        )
        pipeline = make_measurements_pipeline(m)
        pipeline.add_module(module)
        workspace = cpw.Workspace(
            pipeline, module, image_set, cpo.ObjectSet(), m, image_set_list
        )
        fd = None
        try:
            module.post_run(workspace)
            fd = open(path, "rb")
            rdr = csv.reader(fd)
            header = next(rdr)
            for heading, expected in zip(
                header,
                [
                    "Module",
                    "Module Number",
                    "Relationship",
                    "First Object Name",
                    "First Image Number",
                    "First Object Number",
                    "Second Object Name",
                    "Second Image Number",
                    "Second Object Number",
                ],
            ):
                assert heading == expected
            for i in range(len(my_image_numbers1)):
                (
                    module_name,
                    module_number,
                    relationship,
                    object_name_1,
                    image_number_1,
                    object_number_1,
                    object_name_2,
                    image_number_2,
                    object_number_2,
                ) = next(rdr)
                assert module_name == module.module_name
                assert int(module_number) == module.module_num
                assert relationship == my_relationship
                assert object_name_1 == my_object_name1
                assert int(image_number_1) == my_image_numbers1[i]
                assert int(object_number_1) == my_object_numbers1[i]
                assert object_name_2 == my_object_name2
                assert int(image_number_2) == my_image_numbers2[i]
                assert int(object_number_2) == my_object_numbers2[i]
        finally:
            try:
                if fd is not None:
                    fd.close()
                os.remove(path)
            except:
                pass

    def test_test_overwrite_relationships_file():
        output_csv_filename = os.path.join(output_dir, "my_file.csv")
        m = make_measurements()
        pipeline = make_measurements_pipeline(m)
        module = E.ExportToSpreadsheet()
        module.directory.dir_choice = E.cps.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = output_dir
        module.wants_prefix.value = False
        module.wants_everything.value = False
        g = module.object_groups[0]
        g.name.value = E.OBJECT_RELATIONSHIPS
        g.wants_automatic_file_name.value = False
        g.file_name.value = "my_file.csv"
        module.set_module_num(1)
        pipeline.add_module(module)

        workspace = cpw.Workspace(pipeline, module, m, None, m, None)
        assert module.prepare_run(workspace)
        with open(output_csv_filename, "w") as fd:
            fd.write("Hello, world.\n")
        assert not module.prepare_run(workspace)
