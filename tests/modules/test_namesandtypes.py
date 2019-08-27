import hashlib
import io
import os
import tempfile
import urllib.request

import bioformats
import javabridge
import numpy
import pytest
import six

import cellprofiler.measurement
import cellprofiler.modules.createbatchfiles
import cellprofiler.modules.namesandtypes
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.workspace
import tests.modules

M0, M1, M2, M3, M4, M5, M6 = ["MetadataKey%d" % i for i in range(7)]
C0, C1, C2, C3, C4, C5, C6 = ["Column%d" % i for i in range(7)]

IMAGE_NAME = "imagename"
ALT_IMAGE_NAME = "altimagename"
OBJECTS_NAME = "objectsname"
ALT_OBJECTS_NAME = "altobjectsname"
OUTLINES_NAME = "outlines"


def md(keys_and_counts):
    """Generate metadata dictionaries for the given metadata shape

    keys_and_counts - a collection of metadata keys and the dimension of
                        their extent. For instance [(M0, 2), (M1, 3)] generates
                        six dictionaries with two unique values of M0 and
                        three for M1
    """
    keys = [k for k, c in keys_and_counts]
    counts = numpy.array([c for k, c in keys_and_counts])
    divisors = numpy.hstack([[1], numpy.cumprod(counts[:-1])])

    return [
        dict(
            [(k, "k" + str(int(i / d) % c)) for k, d, c in zip(keys, divisors, counts)]
        )
        for i in range(numpy.prod(counts))
    ]


def test_load_v1():
    with open("./tests/resources/modules/namesandtypes/v1.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 3
    module = pipeline.modules()[2]
    assert isinstance(module, cellprofiler.modules.namesandtypes.NamesAndTypes)
    assert module.assignment_method == cellprofiler.modules.namesandtypes.ASSIGN_RULES
    assert (
        module.single_load_as_choice
        == cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE
    )
    assert module.single_image_provider.value == "PI"
    assert module.matching_choice == cellprofiler.modules.namesandtypes.MATCH_BY_ORDER
    assert module.assignments_count.value == 5
    aa = module.assignments
    for assignment, rule, image_name, objects_name, load_as in (
        (
            aa[0],
            'or (metadata does ChannelNumber "0")',
            "DNA",
            "Nuclei",
            cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE,
        ),
        (
            aa[1],
            'or (image does ismonochrome) (metadata does ChannelNumber "1") (extension does istif)',
            "Actin",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE,
        ),
        (
            aa[2],
            'or (metadata does ChannelNumber "2")',
            "GFP",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_MASK,
        ),
        (
            aa[3],
            'or (metadata does ChannelNumber "2")',
            "Foo",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS,
        ),
        (
            aa[4],
            'or (metadata does ChannelNumber "2")',
            "Illum",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_ILLUMINATION_FUNCTION,
        ),
    ):
        assert assignment.rule_filter.value == rule
        assert assignment.image_name == image_name
        assert assignment.object_name == objects_name
        assert assignment.load_as_choice == load_as


def test_load_v2():
    with open("./tests/resources/modules/namesandtypes/v2.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 3
    module = pipeline.modules()[2]
    assert isinstance(module, cellprofiler.modules.namesandtypes.NamesAndTypes)
    assert module.assignment_method == cellprofiler.modules.namesandtypes.ASSIGN_RULES
    assert (
        module.single_load_as_choice
        == cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE
    )
    assert module.single_image_provider.value == "PI"
    assert module.matching_choice == cellprofiler.modules.namesandtypes.MATCH_BY_ORDER
    assert module.assignments_count.value == 5
    aa = module.assignments
    for assignment, rule, image_name, objects_name, load_as in (
        (
            aa[0],
            'or (metadata does ChannelNumber "0")',
            "DNA",
            "Nuclei",
            cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE,
        ),
        (
            aa[1],
            'or (image does ismonochrome) (metadata does ChannelNumber "1") (extension does istif)',
            "Actin",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE,
        ),
        (
            aa[2],
            'or (metadata does ChannelNumber "2")',
            "GFP",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_MASK,
        ),
        (
            aa[3],
            'or (metadata does ChannelNumber "2")',
            "Foo",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS,
        ),
        (
            aa[4],
            'or (metadata does ChannelNumber "2")',
            "Illum",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_ILLUMINATION_FUNCTION,
        ),
    ):
        assert assignment.rule_filter.value == rule
        assert assignment.image_name == image_name
        assert assignment.object_name == objects_name
        assert assignment.load_as_choice == load_as


def test_load_v3():
    with open("./tests/resources/modules/namesandtypes/v3.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 3
    module = pipeline.modules()[2]
    assert isinstance(module, cellprofiler.modules.namesandtypes.NamesAndTypes)
    assert module.assignment_method == cellprofiler.modules.namesandtypes.ASSIGN_RULES
    assert (
        module.single_load_as_choice
        == cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE
    )
    assert module.single_image_provider.value == "PI"
    assert (
        module.single_rescale
        == cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_DATATYPE
    )
    assert module.matching_choice == cellprofiler.modules.namesandtypes.MATCH_BY_ORDER
    assert module.assignments_count.value == 5
    aa = module.assignments
    for assignment, rule, image_name, objects_name, load_as, rescale in (
        (
            aa[0],
            'or (metadata does ChannelNumber "0")',
            "DNA",
            "Nuclei",
            cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE,
            cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_METADATA,
        ),
        (
            aa[1],
            'or (image does ismonochrome) (metadata does ChannelNumber "1") (extension does istif)',
            "Actin",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE,
            cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_DATATYPE,
        ),
        (
            aa[2],
            'or (metadata does ChannelNumber "2")',
            "GFP",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_MASK,
            cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_METADATA,
        ),
        (
            aa[3],
            'or (metadata does ChannelNumber "2")',
            "Foo",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS,
            cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_DATATYPE,
        ),
        (
            aa[4],
            'or (metadata does ChannelNumber "2")',
            "Illum",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_ILLUMINATION_FUNCTION,
            cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_METADATA,
        ),
    ):
        assert assignment.rule_filter.value == rule
        assert assignment.image_name == image_name
        assert assignment.object_name == objects_name
        assert assignment.load_as_choice == load_as


def test_load_v4():
    with open("./tests/resources/modules/namesandtypes/v4.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 3
    module = pipeline.modules()[2]
    assert isinstance(module, cellprofiler.modules.namesandtypes.NamesAndTypes)
    assert module.assignment_method == cellprofiler.modules.namesandtypes.ASSIGN_RULES
    assert (
        module.single_load_as_choice
        == cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE
    )
    assert module.single_image_provider.value == "PI"
    assert (
        module.single_rescale
        == cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_DATATYPE
    )
    assert module.matching_choice == cellprofiler.modules.namesandtypes.MATCH_BY_ORDER
    assert module.assignments_count.value == 5
    aa = module.assignments
    for assignment, rule, image_name, objects_name, load_as, rescale in (
        (
            aa[0],
            'or (metadata does ChannelNumber "0")',
            "DNA",
            "Nuclei",
            cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE,
            cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_METADATA,
        ),
        (
            aa[1],
            'or (image does ismonochrome) (metadata does ChannelNumber "1") (extension does istif)',
            "Actin",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE,
            cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_DATATYPE,
        ),
        (
            aa[2],
            'or (metadata does ChannelNumber "2")',
            "GFP",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_MASK,
            cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_METADATA,
        ),
        (
            aa[3],
            'or (metadata does ChannelNumber "2")',
            "Foo",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS,
            cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_DATATYPE,
        ),
        (
            aa[4],
            'or (metadata does ChannelNumber "2")',
            "Illum",
            "Cells",
            cellprofiler.modules.namesandtypes.LOAD_AS_ILLUMINATION_FUNCTION,
            cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_METADATA,
        ),
    ):
        assert assignment.rule_filter.value == rule
        assert assignment.image_name.value == image_name
        assert assignment.object_name.value == objects_name
        assert assignment.load_as_choice.value == load_as
        assert assignment.rescale.value == rescale
        assert (
            assignment.manual_rescale.value
            == cellprofiler.modules.namesandtypes.DEFAULT_MANUAL_RESCALE
        )
    assert len(module.single_images) == 0


#     def test_load_v5():
#             with open("./tests/resources/modules/align/load_v2.pipeline", "r") as fd:
# data = fd.read()

# foo = r"""CellProfiler Pipeline: http://www.cellprofiler.org
# Version:3
# DateRevision:20130730112304
# ModuleCount:3
# HasImagePlaneDetails:False
#
# Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
#     :{"ShowFiltered"\x3A false}
#     Filter based on rules:Yes
#     Filter:or (extension does istif)
#
# Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
#     Extract metadata?:Yes
#     Extraction method count:1
#     Extraction method:Manual
#     Source:From file name
#     Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})f(?P<Site>\x5B0-9\x5D{2})d(?P<ChannelNumber>\x5B0-9\x5D)
#     Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
#     Filter images:All images
#     :or (file does contain "")
#     Metadata file location\x3A:
#     Match file and image metadata:\x5B\x5D
#
# NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
#     Assign a name to:Images matching rules
#     Select the image type:Color image
#     Name to assign these images:PI
#     :\x5B{u\'Illum\'\x3A u\'Plate\', u\'DNA\'\x3A u\'Plate\', \'Cells\'\x3A u\'Plate\', u\'Actin\'\x3A u\'Plate\', u\'GFP\'\x3A u\'Plate\'}, {u\'Illum\'\x3A u\'Well\', u\'DNA\'\x3A u\'Well\', \'Cells\'\x3A u\'Well\', u\'Actin\'\x3A u\'Well\', u\'GFP\'\x3A u\'Well\'}, {u\'Illum\'\x3A u\'Site\', u\'DNA\'\x3A u\'Site\', \'Cells\'\x3A u\'Site\', u\'Actin\'\x3A u\'Site\', u\'GFP\'\x3A u\'Site\'}\x5D
#     Channel matching method:Order
#     Set intensity range from:Image bit-depth
#     Assignments count:1
#     Single images count:5
#     Select the rule criteria:or (metadata does ChannelNumber "0")
#     Name to assign these images:DNA
#     Name to assign these objects:Nuclei
#     Select the image type:Grayscale image
#     Set intensity range from:Image metadata
#     Retain object outlines?:No
#     Name the outline image:LoadedOutlines
#     Single image:file\x3A///foo/bar
#     Name to assign these images:sDNA
#     Name to assign these objects:sNuclei
#     Select the image type:Grayscale image
#     Set intensity range from:Image metadata
#     Retain object outlines?:No
#     Name the outline image:LoadedOutlines
#     Single image:file\x3A///foo/bar 1 2 3
#     Name to assign these images:Actin
#     Name to assign these objects:Cells
#     Select the image type:Color image
#     Set intensity range from:Image bit-depth
#     Retain object outlines?:No
#     Name the outline image:LoadedOutlines
#     Single image:file\x3A///foo/bar 1 2 3
#     Name to assign these images:GFP
#     Name to assign these objects:Cells
#     Select the image type:Mask
#     Set intensity range from:Image metadata
#     Retain object outlines?:No
#     Name the outline image:LoadedOutlines
#     Single image:file\x3A///foo/bar 1 2 3
#     Name to assign these images:Foo
#     Name to assign these objects:Cells
#     Select the image type:Objects
#     Set intensity range from:Image bit-depth
#     Retain object outlines?:Yes
#     Name the outline image:MyCellOutlines
#     Single image:file\x3A///foo/bar 1 2 3
#     Name to assign these images:Illum
#     Name to assign these objects:Cells
#     Select the image type:Illumination function
#     Set intensity range from:Image metadata
#     Retain object outlines?:No
#     Name the outline image:LoadedOutlines
# """
#             pipeline = cpp.Pipeline()
#             def callback(caller, event):
#                 assertFalse(isinstance(event, cpp.LoadExceptionEvent))
#             pipeline.add_listener(callback)
#             pipeline.load(StringIO(data))
#             assertEqual(len(pipeline.modules()), 3)
#             module = pipeline.modules()[2]
#             assertTrue(isinstance(module, N.NamesAndTypes))
#             assertEqual(module.assignment_method, N.ASSIGN_RULES)
#             assertEqual(module.single_load_as_choice, N.LOAD_AS_COLOR_IMAGE)
#             assertEqual(module.single_image_provider.value, "PI")
#             assertEqual(module.single_rescale, N.INTENSITY_RESCALING_BY_DATATYPE)
#             assertEqual(module.matching_choice, N.MATCH_BY_ORDER)
#             assertEqual(module.assignments_count.value, 1)
#             assertEqual(module.single_images_count.value, 5)
#             assignment = module.assignments[0]
#             assertEqual(assignment.rule_filter,
#                              'or (metadata does ChannelNumber "0")')
#             assertEqual(assignment.image_name, "DNA")
#             assertEqual(assignment.object_name, "Nuclei")
#             assertEqual(assignment.load_as_choice, N.LOAD_AS_GRAYSCALE_IMAGE)
#             assertEqual(assignment.rescale, N.INTENSITY_RESCALING_BY_METADATA)
#             assertEqual(assignment.should_save_outlines, False)
#             assertEqual(assignment.save_outlines, "LoadedOutlines")
#             aa = module.single_images
#             first = True
#             for assignment, image_name, objects_name, load_as, \
#                 rescale, should_save_outlines, outlines_name in (
#                 (aa[0], "sDNA", "sNuclei", N.LOAD_AS_GRAYSCALE_IMAGE, N.INTENSITY_RESCALING_BY_METADATA, False, "LoadedOutlines"),
#                 (aa[1], "Actin", "Cells", N.LOAD_AS_COLOR_IMAGE, N.INTENSITY_RESCALING_BY_DATATYPE, False, "LoadedOutlines"),
#                 (aa[2], "GFP", "Cells", N.LOAD_AS_MASK, N.INTENSITY_RESCALING_BY_METADATA, False, "LoadedOutlines"),
#                 (aa[3], "Foo", "Cells", N.LOAD_AS_OBJECTS, N.INTENSITY_RESCALING_BY_DATATYPE, True, "MyCellOutlines"),
#                 (aa[4], "Illum", "Cells", N.LOAD_AS_ILLUMINATION_FUNCTION, N.INTENSITY_RESCALING_BY_METADATA, False, "LoadedOutlines")):
#                 ipd = assignment.image_plane
#                 assertEqual(ipd.url, "file:///foo/bar")
#                 if first:
#                     assertTrue(all([
#                         x is None for x in ipd.series, ipd.index, ipd.channel]))
#                 else:
#                     assertEqual(ipd.series, 1)
#                     assertEqual(ipd.index, 2)
#                     assertEqual(ipd.channel, 3)
#                 assertEqual(assignment.image_name.value, image_name)
#                 assertEqual(assignment.object_name.value, objects_name)
#                 assertEqual(assignment.load_as_choice.value, load_as)
#                 assertEqual(assignment.rescale.value, rescale)
#                 assertEqual(assignment.should_save_outlines.value, should_save_outlines)
#                 assertEqual(assignment.save_outlines.value, outlines_name)
#                 first = False
#
#     def test_load_v6():
#         with open("./tests/resources/modules/align/load_v2.pipeline", "r") as fd:
# data = fd.read()

# foo = r"""CellProfiler Pipeline: http://www.cellprofiler.org
# Version:3
# DateRevision:20141031194728
# GitHash:49bd1a0
# ModuleCount:3
# HasImagePlaneDetails:False
#
# Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
#     :{"ShowFiltered"\x3A false}
#     Filter images?:Custom
#     Select the rule criteria:or (extension does istif)
#
# Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
#     Extract metadata?:Yes
#     Metadata data type:Text
#     Metadata types:{}
#     Extraction method count:1
#     Metadata extraction method:Extract from file/folder names
#     Metadata source:File name
#     Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})f(?P<Site>\x5B0-9\x5D{2})d(?P<ChannelNumber>\x5B0-9\x5D)
#     Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
#     Extract metadata from:All images
#     Select the filtering criteria:or (file does contain "")
#     Metadata file location:
#     Match file and image metadata:\x5B\x5D
#     Use case insensitive matching?:No
#
# NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:6|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
#     Assign a name to:Images matching rules
#     Select the image type:Color image
#     Name to assign these images:PI
#     Match metadata:\x5B{u\'Illum\'\x3A u\'Plate\', u\'DNA\'\x3A u\'Plate\', \'Cells\'\x3A u\'Plate\', u\'Actin\'\x3A u\'Plate\', u\'GFP\'\x3A u\'Plate\'}, {u\'Illum\'\x3A u\'Well\', u\'DNA\'\x3A u\'Well\', \'Cells\'\x3A u\'Well\', u\'Actin\'\x3A u\'Well\', u\'GFP\'\x3A u\'Well\'}, {u\'Illum\'\x3A u\'Site\', u\'DNA\'\x3A u\'Site\', \'Cells\'\x3A u\'Site\', u\'Actin\'\x3A u\'Site\', u\'GFP\'\x3A u\'Site\'}\x5D
#     Image set matching method:Order
#     Set intensity range from:Image bit-depth
#     Assignments count:1
#     Single images count:5
#     Maximum intensity:100
#     Select the rule criteria:or (metadata does ChannelNumber "0")
#     Name to assign these images:DNA
#     Name to assign these objects:Nuclei
#     Select the image type:Grayscale image
#     Set intensity range from:Image metadata
#     Retain outlines of loaded objects?:No
#     Name the outline image:LoadedOutlines
#     Maximum intensity:200
#     Single image location:file\x3A///foo/bar
#     Name to assign this image:sDNA
#     Name to assign these objects:sNuclei
#     Select the image type:Grayscale image
#     Set intensity range from:Image metadata
#     Retain object outlines?:No
#     Name the outline image:LoadedOutlines
#     Maximum intensity:300
#     Single image location:file\x3A///foo/bar 1 2 3
#     Name to assign this image:Actin
#     Name to assign these objects:Cells
#     Select the image type:Color image
#     Set intensity range from:Image bit-depth
#     Retain object outlines?:No
#     Name the outline image:LoadedOutlines
#     Maximum intensity:400
#     Single image location:file\x3A///foo/bar 1 2 3
#     Name to assign this image:GFP
#     Name to assign these objects:Cells
#     Select the image type:Binary mask
#     Set intensity range from:Image metadata
#     Retain object outlines?:No
#     Name the outline image:LoadedOutlines
#     Maximum intensity:500
#     Single image location:file\x3A///foo/bar 1 2 3
#     Name to assign this image:Foo
#     Name to assign these objects:Cells
#     Select the image type:Objects
#     Set intensity range from:Image bit-depth
#     Retain object outlines?:Yes
#     Name the outline image:MyCellOutlines
#     Maximum intensity:600
#     Single image location:file\x3A///foo/bar 1 2 3
#     Name to assign this image:Illum
#     Name to assign these objects:Cells
#     Select the image type:Illumination function
#     Set intensity range from:Image metadata
#     Retain object outlines?:No
#     Name the outline image:LoadedOutlines
#     Maximum intensity:700
# """
#         pipeline = cpp.Pipeline()
#         def callback(caller, event):
#             assertFalse(isinstance(event, cpp.LoadExceptionEvent))
#         pipeline.add_listener(callback)
#         pipeline.load(StringIO(data))
#         assertEqual(len(pipeline.modules()), 3)
#         module = pipeline.modules()[2]
#         assertTrue(isinstance(module, N.NamesAndTypes))
#         assertEqual(module.assignment_method, N.ASSIGN_RULES)
#         assertEqual(module.single_load_as_choice, N.LOAD_AS_COLOR_IMAGE)
#         assertEqual(module.single_image_provider.value, "PI")
#         assertEqual(module.single_rescale, N.INTENSITY_RESCALING_BY_DATATYPE)
#         assertEqual(module.matching_choice, N.MATCH_BY_ORDER)
#         assertEqual(module.assignments_count.value, 1)
#         assertEqual(module.single_images_count.value, 5)
#         assertEqual(module.manual_rescale.value, 100)
#         assignment = module.assignments[0]
#         assertEqual(assignment.rule_filter,
#                          'or (metadata does ChannelNumber "0")')
#         assertEqual(assignment.image_name, "DNA")
#         assertEqual(assignment.object_name, "Nuclei")
#         assertEqual(assignment.load_as_choice, N.LOAD_AS_GRAYSCALE_IMAGE)
#         assertEqual(assignment.rescale, N.INTENSITY_RESCALING_BY_METADATA)
#         assertEqual(assignment.should_save_outlines, False)
#         assertEqual(assignment.save_outlines, "LoadedOutlines")
#         assertEqual(assignment.manual_rescale.value, 200)
#         aa = module.single_images
#         first = True
#         for assignment, image_name, objects_name, load_as, \
#             rescale, should_save_outlines, outlines_name, manual_rescale in (
#             (aa[0], "sDNA", "sNuclei", N.LOAD_AS_GRAYSCALE_IMAGE, N.INTENSITY_RESCALING_BY_METADATA, False, "LoadedOutlines", 300),
#             (aa[1], "Actin", "Cells", N.LOAD_AS_COLOR_IMAGE, N.INTENSITY_RESCALING_BY_DATATYPE, False, "LoadedOutlines", 400),
#             (aa[2], "GFP", "Cells", N.LOAD_AS_MASK, N.INTENSITY_RESCALING_BY_METADATA, False, "LoadedOutlines", 500),
#             (aa[3], "Foo", "Cells", N.LOAD_AS_OBJECTS, N.INTENSITY_RESCALING_BY_DATATYPE, True, "MyCellOutlines", 600),
#             (aa[4], "Illum", "Cells", N.LOAD_AS_ILLUMINATION_FUNCTION, N.INTENSITY_RESCALING_BY_METADATA, False, "LoadedOutlines", 700)):
#             ipd = assignment.image_plane
#             assertEqual(ipd.url, "file:///foo/bar")
#             if first:
#                 assertTrue(all([
#                     x is None for x in ipd.series, ipd.index, ipd.channel]))
#             else:
#                 assertEqual(ipd.series, 1)
#                 assertEqual(ipd.index, 2)
#                 assertEqual(ipd.channel, 3)
#             assertEqual(assignment.image_name.value, image_name)
#             assertEqual(assignment.object_name.value, objects_name)
#             assertEqual(assignment.load_as_choice.value, load_as)
#             assertEqual(assignment.rescale.value, rescale)
#             assertEqual(assignment.should_save_outlines.value, should_save_outlines)
#             assertEqual(assignment.save_outlines.value, outlines_name)
#             assertEqual(assignment.manual_rescale, manual_rescale)
#             first = False

url_root = "file:" + urllib.request.pathname2url(os.path.abspath(os.path.curdir))


def do_teest(module, channels, expected_tags, expected_metadata, additional=None):
    """Ensure that NamesAndTypes recreates the column layout when run

    module - instance of NamesAndTypes, set up for the test

    channels - a dictionary of channel name to list of "ipds" for that
                channel where "ipd" is a tuple of URL and metadata dictionary
                Entries may appear multiple times (e.g., illumination function)
    expected_tags - the metadata tags that should have been generated
                by prepare_run.
    expected_metadata - a sequence of two-tuples of metadata key and
                        the channel from which the metadata is extracted.
    additional - if present, these are added as ImagePlaneDetails in order
                    to create errors. Format is same as a single channel of
                    channels.
    """
    ipds = []
    urls = set()
    channels = dict(channels)
    if additional is not None:
        channels["Additional"] = additional
    for channel_name in list(channels):
        channel_data = [
            (url_root + "/" + path, metadata)
            for path, metadata in channels[channel_name]
        ]
        channels[channel_name] = channel_data
        for url, metadata in channel_data:
            if url in urls:
                continue
            urls.add(url)
            ipd = make_ipd(url, metadata)
            ipds.append(ipd)
    if additional is not None:
        del channels["Additional"]
    ipds.sort(key=lambda x: x.url)
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.set_filtered_file_list(urls, module)
    pipeline.set_image_plane_details(ipds, list(metadata.keys()), module)
    module.set_module_num(1)
    pipeline.add_module(module)
    m = cellprofiler.measurement.Measurements()
    workspace = cellprofiler.workspace.Workspace(pipeline, module, m, None, m, None)
    assert module.prepare_run(workspace)
    tags = m.get_metadata_tags()
    assert len(tags) == len(expected_tags)
    for tag, expected_tag in zip(tags, expected_tags):
        for et in expected_tag:
            ftr = (
                et
                if et == cellprofiler.measurement.IMAGE_NUMBER
                else "_".join((cellprofiler.measurement.C_METADATA, et))
            )
            if ftr == tag:
                break
        else:
            pytest.fail("%s not in %s" % (tag, ",".join(expected_tag)))
    iscds = m.get_channel_descriptors()
    assert len(iscds) == len(channels)
    for channel_name in list(channels.keys()):
        iscd_match = [x for x in iscds if x.name == channel_name]
        assert len(iscd_match) == 1
        iscd = iscd_match[0]
        assert isinstance(
            iscd, cellprofiler.pipeline.Pipeline.ImageSetChannelDescriptor
        )
        for i, (expected_url, metadata) in enumerate(channels[channel_name]):
            image_number = i + 1
            if iscd.channel_type == iscd.CT_OBJECTS:
                url_ftr = "_".join(
                    (cellprofiler.measurement.C_OBJECTS_URL, channel_name)
                )
            else:
                url_ftr = "_".join((cellprofiler.measurement.C_URL, channel_name))
            assert (
                expected_url == m[cellprofiler.measurement.IMAGE, url_ftr, image_number]
            )
            for key, channel in expected_metadata:
                if channel != channel_name:
                    continue
                md_ftr = "_".join((cellprofiler.measurement.C_METADATA, key))
                assert (
                    metadata[key]
                    == m[cellprofiler.measurement.IMAGE, md_ftr, image_number]
                )
    return workspace


def make_ipd(url, metadata, series=0, index=0, channel=None):
    if channel is None:
        channel = "ALWAYS_MONOCHROME"
    if isinstance(channel, six.string_types):
        channel = javabridge.run_script(
            """
        importPackage(Packages.org.cellprofiler.imageset);
        ImagePlane.%s;"""
            % channel
        )
    jmetadata = javabridge.make_map(**metadata)
    jipd = javabridge.run_script(
        """
            importPackage(Packages.org.cellprofiler.imageset);
            importPackage(Packages.org.cellprofiler.imageset.filter);
            var imageFile=new ImageFile(new java.net.URI(url));
            var imageFileDetails = new ImageFileDetails(imageFile);
            var imageSeries=new ImageSeries(imageFile, series);
            var imageSeriesDetails = new ImageSeriesDetails(imageSeries, imageFileDetails);
            var imagePlane=new ImagePlane(imageSeries, index, channel);
            var ipd = new ImagePlaneDetails(imagePlane, imageSeriesDetails);
            for (var entry in Iterator(metadata.entrySet())) {
                ipd.put(entry.getKey(), entry.getValue());
            }
            ipd;
            """,
        dict(url=url, metadata=jmetadata, series=series, index=index, channel=channel),
    )
    return cellprofiler.pipeline.ImagePlaneDetails(jipd)


def test_01_all():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_ALL
    n.single_image_provider.value = C0
    data = {C0: [("images/1.jpg", {M0: "1"})]}
    do_teest(n, data, [(cellprofiler.measurement.IMAGE_NUMBER,)], [(M0, C0)])


def test_one():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.matching_choice.value = cellprofiler.modules.namesandtypes.MATCH_BY_METADATA
    n.assignments[0].image_name.value = C0
    n.join.build('[{"%s":"%s"}]' % (C0, M0))
    # It should match by order, even if match by metadata and the joiner
    # are set up.
    data = {
        C0: [
            ("images/1.jpg", {M0: "k1"}),
            ("images/2.jpg", {M0: "k3"}),
            ("images/3.jpg", {M0: "k2"}),
        ]
    }
    do_teest(n, data, [(cellprofiler.measurement.IMAGE_NUMBER,)], [(M0, C0)])


def test_match_one_same_key():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.matching_choice.value = cellprofiler.modules.namesandtypes.MATCH_BY_METADATA
    n.add_assignment()
    n.assignments[0].image_name.value = C0
    n.assignments[1].image_name.value = C1
    n.assignments[0].rule_filter.value = 'file does contain "1"'
    n.assignments[1].rule_filter.value = 'file doesnot contain "1"'
    n.join.build("[{'%s':'%s','%s':'%s'}]" % (C0, M0, C1, M0))
    data = {C0: [("images/1.jpg", {M0: "k1"})], C1: [("images/2.jpg", {M0: "k1"})]}
    do_teest(n, data, [(M0,)], [(M0, C0)])


def test_match_one_different_key():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.matching_choice.value = cellprofiler.modules.namesandtypes.MATCH_BY_METADATA
    n.add_assignment()
    n.assignments[0].image_name.value = C0
    n.assignments[1].image_name.value = C1
    n.assignments[0].rule_filter.value = 'file does contain "1"'
    n.assignments[1].rule_filter.value = 'file doesnot contain "1"'
    n.join.build("[{'%s':'%s','%s':'%s'}]" % (C0, M0, C1, M1))
    data = {C0: [("images/1.jpg", {M0: "k1"})], C1: [("images/2.jpg", {M1: "k1"})]}
    do_teest(n, data, [(M0, M1)], [(M0, C0), (M1, C1)])


def test_match_two_one_key():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.matching_choice.value = cellprofiler.modules.namesandtypes.MATCH_BY_METADATA
    n.add_assignment()
    n.assignments[0].image_name.value = C0
    n.assignments[1].image_name.value = C1
    n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
    n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
    n.join.build("[{'%s':'%s','%s':'%s'}]" % (C0, M0, C1, M1))
    data = {
        C0: [("%s%d" % (C0, i), m) for i, m in enumerate(md([(M0, 2)]))],
        C1: [("%s%d" % (C1, i), m) for i, m in enumerate(md([(M1, 2)]))],
    }
    do_teest(n, data, [(M0, M1)], [(M0, C0), (M1, C1)])


def test_match_two_and_two():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.matching_choice.value = cellprofiler.modules.namesandtypes.MATCH_BY_METADATA
    n.add_assignment()
    n.assignments[0].image_name.value = C0
    n.assignments[1].image_name.value = C1
    n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
    n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
    n.join.build(
        "[{'%s':'%s','%s':'%s'},{'%s':'%s','%s':'%s'}]"
        % (C0, M0, C1, M2, C0, M1, C1, M3)
    )
    data = {
        C0: [
            ("%s%s%s" % (C0, m[M0], m[M1]), m)
            for i, m in enumerate(md([(M1, 3), (M0, 2)]))
        ],
        C1: [
            ("%s%s%s" % (C1, m[M2], m[M3]), m)
            for i, m in enumerate(md([(M3, 3), (M2, 2)]))
        ],
    }
    do_teest(n, data, [(M0, M2), (M1, M3)], [(M0, C0), (M1, C0), (M2, C1), (M3, C1)])


def test_01_two_with_same_metadata():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.matching_choice.value = cellprofiler.modules.namesandtypes.MATCH_BY_METADATA
    n.add_assignment()
    n.assignments[0].image_name.value = C0
    n.assignments[1].image_name.value = C1
    n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
    n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
    n.join.build(
        "[{'%s':'%s','%s':'%s'},{'%s':'%s','%s':'%s'}]"
        % (C0, M0, C1, M2, C0, M1, C1, M3)
    )
    data = {
        C0: [
            ("%s%s%s" % (C0, m[M0], m[M1]), m)
            for i, m in enumerate(md([(M1, 3), (M0, 2)]))
        ],
        C1: [
            ("%s%s%s" % (C1, m[M2], m[M3]), m)
            for i, m in enumerate(md([(M3, 3), (M2, 2)]))
        ],
    }
    bad_row = 5
    # Steal the bad row's metadata
    additional = [
        ("%sBad" % C0, data[C0][bad_row][1]),
        data[C0][bad_row],
        data[C1][bad_row],
    ]
    del data[C0][bad_row]
    del data[C1][bad_row]
    do_teest(
        n,
        data,
        [(M0, M2), (M1, M3)],
        [(M0, C0), (M1, C0), (M2, C1), (M3, C1)],
        additional,
    )


def test_02_missing():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.matching_choice.value = cellprofiler.modules.namesandtypes.MATCH_BY_METADATA
    n.add_assignment()
    n.assignments[0].image_name.value = C0
    n.assignments[1].image_name.value = C1
    n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
    n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
    n.join.build(
        "[{'%s':'%s','%s':'%s'},{'%s':'%s','%s':'%s'}]"
        % (C0, M0, C1, M2, C0, M1, C1, M3)
    )
    data = {
        C0: [
            ("%s%s%s" % (C0, m[M0], m[M1]), m)
            for i, m in enumerate(md([(M1, 3), (M0, 2)]))
        ],
        C1: [
            ("%s%s%s" % (C1, m[M2], m[M3]), m)
            for i, m in enumerate(md([(M3, 3), (M2, 2)]))
        ],
    }
    bad_row = 3
    # Steal the bad row's metadata
    additional = [data[C1][bad_row]]
    del data[C0][bad_row]
    del data[C1][bad_row]
    do_teest(
        n,
        data,
        [(M0, M2), (M1, M3)],
        [(M0, C0), (M1, C0), (M2, C1), (M3, C1)],
        additional,
    )


def test_one_against_all():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.matching_choice.value = cellprofiler.modules.namesandtypes.MATCH_BY_METADATA
    n.add_assignment()
    n.assignments[0].image_name.value = C0
    n.assignments[1].image_name.value = C1
    n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
    n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
    n.join.build("[{'%s':None,'%s':'%s'}]" % (C0, C1, M0))
    data = {
        C0: [(C0, {})] * 3,
        C1: [("%s%d" % (C1, i), m) for i, m in enumerate(md([(M0, 3)]))],
    }
    do_teest(n, data, [(M0,)], [(M0, C1)])


def test_some_against_all():
    #
    # Permute both the order of the columns and the order of joins
    #

    joins = [{C0: M0, C1: M1}, {C0: None, C1: M2}]
    for cA, cB in ((C0, C1), (C1, C0)):
        for j0, j1 in ((0, 1), (1, 0)):
            n = cellprofiler.modules.namesandtypes.NamesAndTypes()
            n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
            n.matching_choice.value = (
                cellprofiler.modules.namesandtypes.MATCH_BY_METADATA
            )
            n.add_assignment()
            n.assignments[0].image_name.value = cA
            n.assignments[1].image_name.value = cB
            n.assignments[0].rule_filter.value = 'file does contain "%s"' % cA
            n.assignments[1].rule_filter.value = 'file does contain "%s"' % cB
            n.join.build(repr([joins[j0], joins[j1]]))
            mA0 = [M0, M3][j0]
            mB0 = [M0, M3][j1]
            mA1 = joins[j0][C1]
            mB1 = joins[j1][C1]
            data = {
                C0: [
                    ("%s%s" % (C0, m[M0]), m)
                    for i, m in enumerate(md([(mB0, 2), (mA0, 3)]))
                ],
                C1: [
                    ("%s%s%s" % (C1, m[M1], m[M2]), m)
                    for i, m in enumerate(md([(mB1, 2), (mA1, 3)]))
                ],
            }
            expected_keys = [[(M0, M1), (M2,)][i] for i in (j0, j1)]
            do_teest(n, data, expected_keys, [(C0, M0), (C0, M3), (C1, M1), (C1, M2)])


def test_by_order():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.matching_choice.value = cellprofiler.modules.namesandtypes.MATCH_BY_ORDER
    n.add_assignment()
    n.assignments[0].image_name.value = C0
    n.assignments[1].image_name.value = C1
    n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
    n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
    data = {
        C0: [("%s%d" % (C0, i + 1), m) for i, m in enumerate(md([(M0, 2)]))],
        C1: [("%s%d" % (C1, i + 1), m) for i, m in enumerate(md([(M1, 2)]))],
    }
    do_teest(n, data, [(cellprofiler.measurement.IMAGE_NUMBER,)], [(C0, M0), (C1, M1)])


def test_by_order_bad():
    # Regression test of issue #392: columns of different lengths
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.matching_choice.value = cellprofiler.modules.namesandtypes.MATCH_BY_ORDER
    n.add_assignment()
    n.assignments[0].image_name.value = C0
    n.assignments[1].image_name.value = C1
    n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
    n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
    n.ipd_columns = [
        [make_ipd("%s%d" % (C0, (3 - i)), m) for i, m in enumerate(md([(M0, 3)]))],
        [make_ipd("%s%d" % (C1, i + 1), m) for i, m in enumerate(md([(M1, 2)]))],
    ]
    data = {
        C0: [("%s%d" % (C0, i + 1), m) for i, m in enumerate(md([(M0, 2)]))],
        C1: [("%s%d" % (C1, i + 1), m) for i, m in enumerate(md([(M1, 2)]))],
    }
    additional = [("%sBad" % C0, {})]
    do_teest(n, data, [(cellprofiler.measurement.IMAGE_NUMBER,)], [(C0, M0), (C1, M1)])


def test_single_image_by_order():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.matching_choice.value = cellprofiler.modules.namesandtypes.MATCH_BY_ORDER
    n.add_assignment()
    n.assignments[0].image_name.value = C0
    n.assignments[1].image_name.value = C1
    n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
    n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
    n.add_single_image()
    si = n.single_images[0]
    si.image_plane.value = si.image_plane.build(url_root + "/illum.tif")
    si.image_name.value = C2
    si.load_as_choice.value = cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE
    data = {
        C0: [("%s%d" % (C0, i + 1), m) for i, m in enumerate(md([(M0, 2)]))],
        C1: [("%s%d" % (C1, i + 1), m) for i, m in enumerate(md([(M1, 2)]))],
        C2: [("illum.tif", {}) for i, m in enumerate(md([(M1, 2)]))],
    }
    workspace = do_teest(
        n, data, [(cellprofiler.measurement.IMAGE_NUMBER,)], [(C0, M0), (C1, M1)]
    )
    m = workspace.measurements
    image_numbers = m.get_image_numbers()
    filenames = m[
        cellprofiler.measurement.IMAGE,
        cellprofiler.measurement.C_FILE_NAME + "_" + C2,
        image_numbers,
    ]
    assert all([f == "illum.tif" for f in filenames])
    urls = m[
        cellprofiler.measurement.IMAGE,
        cellprofiler.measurement.C_URL + "_" + C2,
        image_numbers,
    ]
    assert all([url == si.image_plane.url for url in urls])


def test_single_image_by_metadata():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.matching_choice.value = cellprofiler.modules.namesandtypes.MATCH_BY_METADATA
    n.add_assignment()
    n.assignments[0].image_name.value = C0
    n.assignments[1].image_name.value = C1
    n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
    n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
    n.join.build(
        "[{'%s':'%s','%s':'%s'},{'%s':'%s','%s':'%s'}]"
        % (C0, M0, C1, M2, C0, M1, C1, M3)
    )
    n.add_single_image()
    si = n.single_images[0]
    si.image_plane.value = si.image_plane.build(url_root + "/illum.tif")
    si.image_name.value = C2
    si.load_as_choice.value = cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE
    data = {
        C0: [
            ("%s%s%s" % (C0, m[M0], m[M1]), m)
            for i, m in enumerate(md([(M1, 3), (M0, 2)]))
        ],
        C1: [
            ("%s%s%s" % (C1, m[M2], m[M3]), m)
            for i, m in enumerate(md([(M3, 3), (M2, 2)]))
        ],
        C2: [("illum.tif", {}) for i, m in enumerate(md([(M3, 3), (M2, 2)]))],
    }
    do_teest(n, data, [(M0, M2), (M1, M3)], [(M0, C0), (M1, C0), (M2, C1), (M3, C1)])


def test_prepare_to_create_batch_single():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.module_num = 1
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_ALL
    n.single_image_provider.value = IMAGE_NAME
    m = cellprofiler.measurement.Measurements(mode="memory")
    pathnames = ["foo", "fuu"]
    expected_pathnames = ["bar", "fuu"]
    filenames = ["boo", "foobar"]
    expected_filenames = ["boo", "barbar"]
    urlnames = ["file:/foo/bar", "http://foo/bar"]
    expected_urlnames = ["file:/bar/bar", "http://foo/bar"]

    m.add_all_measurements(
        cellprofiler.measurement.IMAGE,
        cellprofiler.measurement.C_FILE_NAME + "_" + IMAGE_NAME,
        filenames,
    )
    m.add_all_measurements(
        cellprofiler.measurement.IMAGE,
        cellprofiler.measurement.C_PATH_NAME + "_" + IMAGE_NAME,
        pathnames,
    )
    m.add_all_measurements(
        cellprofiler.measurement.IMAGE,
        cellprofiler.measurement.C_URL + "_" + IMAGE_NAME,
        urlnames,
    )
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.add_module(n)
    workspace = cellprofiler.workspace.Workspace(pipeline, n, m, None, m, None)
    n.prepare_to_create_batch(workspace, lambda x: x.replace("foo", "bar"))
    for feature, expected in (
        (cellprofiler.measurement.C_FILE_NAME, expected_filenames),
        (cellprofiler.measurement.C_PATH_NAME, expected_pathnames),
        (cellprofiler.measurement.C_URL, expected_urlnames),
    ):
        values = m.get_measurement(
            cellprofiler.measurement.IMAGE,
            feature + "_" + IMAGE_NAME,
            numpy.arange(len(expected)) + 1,
        )
        assert expected == list(values)


def test_prepare_to_create_batch_multiple():
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.module_num = 1
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.add_assignment()
    n.assignments[
        0
    ].load_as_choice.value = cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE
    n.assignments[0].image_name.value = IMAGE_NAME
    n.assignments[
        1
    ].load_as_choice.value = cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS
    n.assignments[1].object_name.value = OBJECTS_NAME
    m = cellprofiler.measurement.Measurements(mode="memory")
    pathnames = ["foo", "fuu"]
    expected_pathnames = ["bar", "fuu"]
    filenames = ["boo", "foobar"]
    expected_filenames = ["boo", "barbar"]
    urlnames = ["file:/foo/bar", "http://foo/bar"]
    expected_urlnames = ["file:/bar/bar", "http://foo/bar"]

    for feature, name, values in (
        (cellprofiler.measurement.C_FILE_NAME, IMAGE_NAME, filenames),
        (
            cellprofiler.measurement.C_OBJECTS_FILE_NAME,
            OBJECTS_NAME,
            reversed(filenames),
        ),
        (cellprofiler.measurement.C_PATH_NAME, IMAGE_NAME, pathnames),
        (
            cellprofiler.measurement.C_OBJECTS_PATH_NAME,
            OBJECTS_NAME,
            reversed(pathnames),
        ),
        (cellprofiler.measurement.C_URL, IMAGE_NAME, urlnames),
        (cellprofiler.measurement.C_OBJECTS_URL, OBJECTS_NAME, reversed(urlnames)),
    ):
        m.add_all_measurements(
            cellprofiler.measurement.IMAGE, feature + "_" + name, values
        )
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.add_module(n)
    workspace = cellprofiler.workspace.Workspace(pipeline, n, m, None, m, None)
    n.prepare_to_create_batch(workspace, lambda x: x.replace("foo", "bar"))
    for feature, name, expected in (
        (cellprofiler.measurement.C_FILE_NAME, IMAGE_NAME, expected_filenames),
        (
            cellprofiler.measurement.C_OBJECTS_FILE_NAME,
            OBJECTS_NAME,
            reversed(expected_filenames),
        ),
        (cellprofiler.measurement.C_PATH_NAME, IMAGE_NAME, expected_pathnames),
        (
            cellprofiler.measurement.C_OBJECTS_PATH_NAME,
            OBJECTS_NAME,
            reversed(expected_pathnames),
        ),
        (cellprofiler.measurement.C_URL, IMAGE_NAME, expected_urlnames),
        (
            cellprofiler.measurement.C_OBJECTS_URL,
            OBJECTS_NAME,
            reversed(expected_urlnames),
        ),
    ):
        values = m.get_measurement(
            cellprofiler.measurement.IMAGE, feature + "_" + name, numpy.arange(1, 3)
        )
        assert list(expected) == list(values)


def test_prepare_to_create_batch_single_image():
    si_names = ["si1", "si2"]
    pathnames = ["foo", "fuu"]
    expected_pathnames = ["bar", "fuu"]
    filenames = ["boo", "foobar"]
    expected_filenames = ["boo", "barbar"]
    urlnames = ["file:/foo/bar", "http://foo/bar"]
    expected_urlnames = ["file:/bar/bar", "http://foo/bar"]

    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.module_num = 1
    n.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    n.assignments[
        0
    ].load_as_choice.value = cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE
    n.assignments[0].image_name.value = IMAGE_NAME
    for name, url in zip(si_names, urlnames):
        n.add_single_image()
        si = n.single_images[-1]
        si.image_plane.value = si.image_plane.build(url)
        si.load_as_choice.value = (
            cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE
        )
        si.image_name.value = name

    m = cellprofiler.measurement.Measurements(mode="memory")

    for feature, name, values in (
        (cellprofiler.measurement.C_FILE_NAME, IMAGE_NAME, filenames),
        (cellprofiler.measurement.C_FILE_NAME, si_names[0], filenames[:1] * 2),
        (cellprofiler.measurement.C_FILE_NAME, si_names[1], filenames[1:] * 2),
        (cellprofiler.measurement.C_PATH_NAME, IMAGE_NAME, pathnames),
        (cellprofiler.measurement.C_PATH_NAME, si_names[0], pathnames[:1] * 2),
        (cellprofiler.measurement.C_PATH_NAME, si_names[1], pathnames[1:] * 2),
        (cellprofiler.measurement.C_URL, IMAGE_NAME, urlnames),
        (cellprofiler.measurement.C_URL, si_names[0], urlnames[:1] * 2),
        (cellprofiler.measurement.C_URL, si_names[1], urlnames[1:] * 2),
    ):
        m.add_all_measurements(
            cellprofiler.measurement.IMAGE, feature + "_" + name, values
        )
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.add_module(n)
    workspace = cellprofiler.workspace.Workspace(pipeline, n, m, None, m, None)
    n.prepare_to_create_batch(workspace, lambda x: x.replace("foo", "bar"))
    for feature, name, expected in (
        (cellprofiler.measurement.C_FILE_NAME, IMAGE_NAME, expected_filenames),
        (cellprofiler.measurement.C_FILE_NAME, si_names[0], expected_filenames[:1] * 2),
        (cellprofiler.measurement.C_FILE_NAME, si_names[1], expected_filenames[1:] * 2),
        (cellprofiler.measurement.C_PATH_NAME, IMAGE_NAME, expected_pathnames),
        (cellprofiler.measurement.C_PATH_NAME, si_names[0], expected_pathnames[:1] * 2),
        (cellprofiler.measurement.C_PATH_NAME, si_names[1], expected_pathnames[1:] * 2),
        (cellprofiler.measurement.C_URL, IMAGE_NAME, expected_urlnames),
        (cellprofiler.measurement.C_URL, si_names[0], expected_urlnames[:1] * 2),
        (cellprofiler.measurement.C_URL, si_names[1], expected_urlnames[1:] * 2),
    ):
        values = m.get_measurement(
            cellprofiler.measurement.IMAGE, feature + "_" + name, numpy.arange(1, 3)
        )
        assert list(expected) == list(values)


# def test_create_batch_files_imagesets():
#     # Regression test of issue 1129
#     # Once imagesets are pickled in the M_IMAGE_SET measurement,
#     # their path names are smoewhat inaccessible, yet need conversion.
#     #
#     folders = ["ExampleAllModulesPipeline", "Images"]
#     aoi = "all_ones_image.tif"
#     ooi = "one_object_00_A.tif"
#     path = maybe_download_example_images(folders, [aoi, ooi])
#     aoi_path = os.path.join(path, aoi)
#     ooi_path = os.path.join(path, ooi)
#     m = cpmeas.Measurements()
#     pipeline = cpp.Pipeline()
#     pipeline.init_modules()
#     images_module, metadata_module, module, groups_module = \
#         pipeline.modules()
#     assert isinstance(module, N.NamesAndTypes)
#     cbf_module = CreateBatchFiles()
#     cbf_module.module_num = groups_module.module_num + 1
#     pipeline.add_module(cbf_module)
#     workspace = cpw.Workspace(
#         pipeline, images_module, m,
#         cpo.ObjectSet(), m, None)
#     #
#     # Add two files
#     #
#     current_path = os.path.abspath(os.curdir)
#     target_path = os.path.join(example_images_directory(), *folders)
#     img_url = pathname2url(os.path.join(current_path, aoi))
#     objects_url = pathname2url(os.path.join(current_path, ooi))
#     pipeline.add_urls([img_url, objects_url])
#     workspace.file_list.add_files_to_filelist([img_url, objects_url])
#     #
#     # Set up NamesAndTypes to read 1 image and 1 object
#     #
#     module.assignment_method.value = N.ASSIGN_RULES
#     module.add_assignment()
#     a = module.assignments[0]
#     a.rule_filter.value = 'and (file does contain "_image")'
#     a.image_name.value = IMAGE_NAME
#     a.load_as_choice.value = N.LOAD_AS_GRAYSCALE_IMAGE
#     a = module.assignments[1]
#     a.rule_filter.value = 'and (file does contain "_object_")'
#     a.load_as_choice.value = N.LOAD_AS_OBJECTS
#     a.object_name.value = OBJECTS_NAME
#     #
#     # Set up CreateBatchFiles to change names.
#     #
#     tempdir = tempfile.mkdtemp()
#     batch_data_filename = os.path.join(tempdir, F_BATCH_DATA_H5)
#     try:
#         cbf_module.wants_default_output_directory.value = False
#         cbf_module.custom_output_directory.value = tempdir
#         cbf_module.mappings[0].local_directory.value = current_path
#         cbf_module.mappings[0].remote_directory.value = target_path
#         cbf_module.go_to_website.value = False
#         pipeline.prepare_run(workspace)
#         assertTrue(os.path.exists(batch_data_filename))
#         #
#         # Load Batch_data.h5
#         #
#         workspace.load(batch_data_filename, True)
#         module = pipeline.modules()[2]
#         assertTrue(isinstance(module, N.NamesAndTypes))
#         workspace.set_module(module)
#         module.run(workspace)
#         img = workspace.image_set.get_image(IMAGE_NAME)
#         target = load_image(aoi_path)
#         assertEquals(tuple(img.pixel_data.shape),
#                           tuple(target.shape))
#         objs = workspace.object_set.get_objects(OBJECTS_NAME)
#         target = load_image(ooi_path, rescale = False)
#         n_objects = np.max(target)
#         assertEquals(objs.count, n_objects)
#     finally:
#         import gc
#         gc.collect()
#         os.remove(batch_data_filename)
#         os.rmdir(tempdir)


def run_workspace(
    path,
    load_as_type,
    series=None,
    index=None,
    channel=None,
    single=False,
    rescaled=cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_METADATA,
    lsi=[],
    volume=False,
    spacing=None,
):
    """Run a workspace to load a file

    path - path to the file
    load_as_type - one of the LOAD_AS... constants
    series, index, channel - pick a plane from within a file
    single - use ASSIGN_ALL to assign all images to a single channel
    rescaled - rescale the image if True
    lsi - a list of single images to load. Each list item is a dictionary
            describing the single image. Format of the dictionary is:
            "path": <path-to-file>
            "load_as_type": <how to load single image>
            "name": <image or object name>
            "rescaled": True or False (defaults to True)

    returns the workspace after running
    """
    if isinstance(rescaled, float):
        manual_rescale = rescaled
        rescaled = cellprofiler.modules.namesandtypes.INTENSITY_MANUAL
    else:
        manual_rescale = 255.0
    n = cellprofiler.modules.namesandtypes.NamesAndTypes()
    n.assignment_method.value = (
        cellprofiler.modules.namesandtypes.ASSIGN_ALL
        if single
        else cellprofiler.modules.namesandtypes.ASSIGN_RULES
    )
    n.single_image_provider.value = IMAGE_NAME
    n.single_load_as_choice.value = load_as_type
    n.single_rescale.value = rescaled
    n.manual_rescale.value = manual_rescale
    n.process_as_3d.value = volume
    if spacing is not None:
        z, x, y = spacing
        n.x.value = x
        n.y.value = y
        n.z.value = z
    n.assignments[0].image_name.value = IMAGE_NAME
    n.assignments[0].object_name.value = OBJECTS_NAME
    n.assignments[0].load_as_choice.value = load_as_type
    n.assignments[0].rescale.value = rescaled
    n.assignments[0].manual_rescale.value = manual_rescale
    n.module_num = 1
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.add_module(n)
    url = cellprofiler.modules.loadimages.pathname2url(path)
    pathname, filename = os.path.split(path)
    m = cellprofiler.measurement.Measurements()
    if load_as_type == cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS:
        url_feature = cellprofiler.measurement.C_OBJECTS_URL + "_" + OBJECTS_NAME
        path_feature = cellprofiler.measurement.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME
        file_feature = cellprofiler.measurement.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME
        series_feature = cellprofiler.measurement.C_OBJECTS_SERIES + "_" + OBJECTS_NAME
        frame_feature = cellprofiler.measurement.C_OBJECTS_FRAME + "_" + OBJECTS_NAME
        channel_feature = (
            cellprofiler.measurement.C_OBJECTS_CHANNEL + "_" + OBJECTS_NAME
        )
        names = javabridge.make_list([OBJECTS_NAME])
    else:
        url_feature = cellprofiler.measurement.C_URL + "_" + IMAGE_NAME
        path_feature = cellprofiler.measurement.C_PATH_NAME + "_" + IMAGE_NAME
        file_feature = cellprofiler.measurement.C_FILE_NAME + "_" + IMAGE_NAME
        series_feature = cellprofiler.measurement.C_SERIES + "_" + IMAGE_NAME
        frame_feature = cellprofiler.measurement.C_FRAME + "_" + IMAGE_NAME
        channel_feature = cellprofiler.measurement.C_CHANNEL + "_" + IMAGE_NAME
        names = javabridge.make_list([IMAGE_NAME])

    m.image_set_number = 1
    m.add_measurement(cellprofiler.measurement.IMAGE, url_feature, url)
    m.add_measurement(cellprofiler.measurement.IMAGE, path_feature, pathname)
    m.add_measurement(cellprofiler.measurement.IMAGE, file_feature, filename)
    if series is not None:
        m.add_measurement(cellprofiler.measurement.IMAGE, series_feature, series)
    if index is not None:
        m.add_measurement(cellprofiler.measurement.IMAGE, frame_feature, index)
    if channel is not None:
        m.add_measurement(cellprofiler.measurement.IMAGE, channel_feature, channel)
    m.add_measurement(
        cellprofiler.measurement.IMAGE, cellprofiler.measurement.GROUP_NUMBER, 1
    )
    m.add_measurement(
        cellprofiler.measurement.IMAGE, cellprofiler.measurement.GROUP_INDEX, 1
    )
    if load_as_type == cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE:
        stack = "Color"
        if channel is None:
            channel = "INTERLEAVED"
    elif load_as_type == cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS:
        stack = "Objects"
        if channel is None:
            channel = "OBJECT_PLANES"
    else:
        stack = "Monochrome"
        if channel is None:
            channel = "ALWAYS_MONOCHROME"
    ipds = javabridge.make_list(
        [make_ipd(url, {}, series or 0, index or 0, channel).jipd]
    )

    for d in lsi:
        path = d["path"]
        load_as_type = d["load_as_type"]
        name = d["name"]
        names.add(name)
        rescaled = d.get("rescaled", True)
        n.add_single_image()
        si = n.single_images[-1]
        si.image_name.value = name
        si.object_name.value = name
        si.load_as_choice.value = load_as_type
        si.rescale.value = rescaled
        si.manual_rescale.value = manual_rescale

        url = cellprofiler.modules.loadimages.pathname2url(path)
        pathname, filename = os.path.split(path)
        if load_as_type == cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS:
            url_feature = cellprofiler.measurement.C_OBJECTS_URL + "_" + name
            path_feature = cellprofiler.measurement.C_OBJECTS_PATH_NAME + "_" + name
            file_feature = cellprofiler.measurement.C_OBJECTS_FILE_NAME + "_" + name
            series_feature = cellprofiler.measurement.C_OBJECTS_SERIES + "_" + name
            frame_feature = cellprofiler.measurement.C_OBJECTS_FRAME + "_" + name
            channel_feature = cellprofiler.measurement.C_OBJECTS_CHANNEL + "_" + name
        else:
            url_feature = cellprofiler.measurement.C_URL + "_" + name
            path_feature = cellprofiler.measurement.C_PATH_NAME + "_" + name
            file_feature = cellprofiler.measurement.C_FILE_NAME + "_" + name
            series_feature = cellprofiler.measurement.C_SERIES + "_" + name
            frame_feature = cellprofiler.measurement.C_FRAME + "_" + name
            channel_feature = cellprofiler.measurement.C_CHANNEL + "_" + name
        m.add_measurement(cellprofiler.measurement.IMAGE, url_feature, url)
        m.add_measurement(cellprofiler.measurement.IMAGE, path_feature, pathname)
        m.add_measurement(cellprofiler.measurement.IMAGE, file_feature, filename)
        ipds.add(make_ipd(url, {}).jipd)

    script = (
        """
                    importPackage(Packages.org.cellprofiler.imageset);
                    var ls = new java.util.ArrayList();
                    for (var ipd in Iterator(ipds)) {
                        ls.add(ImagePlaneDetailsStack.make%sStack(ipd));
                    }
                    var kwlist = new java.util.ArrayList();
                    kwlist.add("ImageNumber");
                    var imageSet = new ImageSet(ls, kwlist);
                    imageSet.compress(names, null);
                    """
        % stack
    )
    blob = javabridge.run_script(script, dict(ipds=ipds.o, names=names.o))
    blob = javabridge.get_env().get_byte_array_elements(blob)
    m.add_measurement(
        cellprofiler.measurement.IMAGE,
        cellprofiler.modules.namesandtypes.M_IMAGE_SET,
        blob,
        data_type=numpy.uint8,
    )

    workspace = cellprofiler.workspace.Workspace(
        pipeline, n, m, cellprofiler.object.ObjectSet(), m, None
    )
    n.run(workspace)
    return workspace


def test_load_color():
    shape = (21, 31, 3)
    path = tests.modules.maybe_download_example_image(
        ["ExampleColorToGray"], "nt_03_01_color.tif", shape
    )
    with open(path, "rb") as fd:
        md5 = hashlib.md5(fd.read()).hexdigest()
    workspace = run_workspace(
        path, cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE
    )
    image = workspace.image_set.get_image(IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape == shape
    assert numpy.all(pixel_data >= 0)
    assert numpy.all(pixel_data <= 1)
    m = workspace.measurements
    assert (
        m[
            cellprofiler.measurement.IMAGE,
            cellprofiler.modules.loadimages.C_MD5_DIGEST + "_" + IMAGE_NAME,
        ]
        == md5
    )
    assert (
        m[
            cellprofiler.measurement.IMAGE,
            cellprofiler.modules.loadimages.C_HEIGHT + "_" + IMAGE_NAME,
        ]
        == 21
    )
    assert (
        m[
            cellprofiler.measurement.IMAGE,
            cellprofiler.modules.loadimages.C_WIDTH + "_" + IMAGE_NAME,
        ]
        == 31
    )


def test_load_monochrome_as_color():
    path = get_monochrome_image_path()
    target = bioformats.load_image(path)
    target_shape = (target.shape[0], target.shape[1], 3)
    workspace = run_workspace(
        path, cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE
    )
    image = workspace.image_set.get_image(IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape == target_shape
    assert numpy.all(pixel_data >= 0)
    assert numpy.all(pixel_data <= 1)
    numpy.testing.assert_equal(pixel_data[:, :, 0], pixel_data[:, :, 1])
    numpy.testing.assert_equal(pixel_data[:, :, 0], pixel_data[:, :, 2])


def test_load_color_frame():
    path = tests.modules.maybe_download_tesst_image("DrosophilaEmbryo_GFPHistone.avi")
    workspace = run_workspace(
        path, cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE, index=3
    )
    image = workspace.image_set.get_image(IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape == (264, 542, 3)
    assert numpy.all(pixel_data >= 0)
    assert numpy.all(pixel_data <= 1)
    assert numpy.any(pixel_data[:, :, 0] != pixel_data[:, :, 1])
    assert numpy.any(pixel_data[:, :, 0] != pixel_data[:, :, 2])


def get_monochrome_image_path():
    folder = "ExampleGrayToColor"
    file_name = "AS_09125_050116030001_D03f00d0.tif"
    return tests.modules.maybe_download_example_image([folder], file_name)


def test_load_monochrome():
    path = get_monochrome_image_path()
    target = bioformats.load_image(path)
    workspace = run_workspace(
        path, cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE
    )
    image = workspace.image_set.get_image(IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape == target.shape
    assert numpy.all(pixel_data >= 0)
    assert numpy.all(pixel_data <= 1)


def test_load_color_as_monochrome():
    shape = (21, 31, 3)
    path = tests.modules.maybe_download_example_image(
        ["ExampleColorToGray"], "nt_03_05_color.tif", shape
    )
    workspace = run_workspace(
        path, cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE
    )
    image = workspace.image_set.get_image(IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape == shape[:2]
    assert numpy.all(pixel_data >= 0)
    assert numpy.all(pixel_data <= 1)


def test_load_monochrome_plane():
    path = tests.modules.maybe_download_tesst_image("5channel.tif")

    for i in range(5):
        workspace = run_workspace(
            path, cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE, index=i
        )
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        assert pixel_data.shape == (64, 64)
        if i == 0:
            plane_0 = pixel_data.copy()
        else:
            assert numpy.any(pixel_data != plane_0)


def test_load_raw():
    folder = "namesandtypes_03_07"
    file_name = "1-162hrh2ax2.tif"
    path = tests.modules.make_12_bit_image(folder, file_name, (34, 19))
    workspace = run_workspace(
        path, cellprofiler.modules.namesandtypes.LOAD_AS_ILLUMINATION_FUNCTION
    )
    image = workspace.image_set.get_image(IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape == (34, 19)
    assert numpy.all(pixel_data >= 0)
    assert numpy.all(pixel_data <= 1.0 / 16.0)


def test_load_mask():
    path = tests.modules.maybe_download_example_image(
        ["ExampleSBSImages"], "Channel2-01-A-01.tif"
    )
    target = bioformats.load_image(path)
    workspace = run_workspace(path, cellprofiler.modules.namesandtypes.LOAD_AS_MASK)
    image = workspace.image_set.get_image(IMAGE_NAME)
    pixel_data = image.pixel_data
    assert pixel_data.shape == target.shape
    assert numpy.sum(~pixel_data) == numpy.sum(target == 0)


def test_load_objects():
    path = tests.modules.maybe_download_example_image(
        ["ExampleSBSImages"], "Channel2-01-A-01.tif"
    )
    target = bioformats.load_image(path, rescale=False)
    with open(path, "rb") as fd:
        md5 = hashlib.md5(fd.read()).hexdigest()
    workspace = run_workspace(path, cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS)
    o = workspace.object_set.get_objects(OBJECTS_NAME)
    assert isinstance(o, cellprofiler.object.Objects)
    areas = o.areas
    counts = numpy.bincount(target.flatten())
    assert areas[0] == counts[1]
    assert areas[1] == counts[2]
    assert areas[2] == counts[3]
    m = workspace.measurements
    assert (
        m[
            cellprofiler.measurement.IMAGE,
            cellprofiler.modules.loadimages.C_MD5_DIGEST + "_" + OBJECTS_NAME,
        ]
        == md5
    )
    assert (
        m[
            cellprofiler.measurement.IMAGE,
            cellprofiler.modules.loadimages.C_WIDTH + "_" + OBJECTS_NAME,
        ]
        == target.shape[1]
    )


def test_load_overlapped_objects():
    from .test_loadimages import overlapped_objects_data
    from .test_loadimages import overlapped_objects_data_masks

    fd, path = tempfile.mkstemp(".tif")
    f = os.fdopen(fd, "wb")
    f.write(overlapped_objects_data)
    f.close()
    try:
        workspace = run_workspace(
            path, cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS
        )
        o = workspace.object_set.get_objects(OBJECTS_NAME)
        assert isinstance(o, cellprofiler.object.Objects)
        assert o.count == 2
        mask = numpy.zeros(overlapped_objects_data_masks[0].shape, bool)
        expected_mask = (
            overlapped_objects_data_masks[0] | overlapped_objects_data_masks[1]
        )
        for i in range(2):
            expected = overlapped_objects_data_masks[i]
            i, j = o.ijv[o.ijv[:, 2] == i + 1, :2].transpose()
            assert numpy.all(expected[i, j])
            mask[i, j] = True
        assert not numpy.any(mask[~expected_mask])
    finally:
        try:
            os.unlink(path)
        except:
            pass


def test_load_rescaled():
    # Test all color/monochrome rescaled paths
    folder = "namesandtypes_03_11"
    file_name = "1-162hrh2ax2.tif"
    path = tests.modules.make_12_bit_image(folder, file_name, (34, 19))
    for single in (True, False):
        for rescaled in (
            cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_METADATA,
            cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_DATATYPE,
            float(2 ** 17),
        ):
            for load_as in (
                cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE,
                cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE,
            ):
                workspace = run_workspace(
                    path, load_as, single=single, rescaled=rescaled
                )
                image = workspace.image_set.get_image(IMAGE_NAME)
                pixel_data = image.pixel_data
                assert numpy.all(pixel_data >= 0)
                if (
                    rescaled
                    == cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_METADATA
                ):
                    assert numpy.any(pixel_data > 1.0 / 16.0)
                elif (
                    rescaled
                    == cellprofiler.modules.namesandtypes.INTENSITY_RESCALING_BY_DATATYPE
                ):
                    assert numpy.all(pixel_data <= 1.0 / 16.0)
                    assert numpy.any(pixel_data > 1.0 / 32.0)
                else:
                    assert numpy.all(pixel_data <= 1.0 / 32.0)


def test_load_single_image():
    # Test loading a pipeline whose image set loads a single image
    path = tests.modules.maybe_download_example_image(
        ["ExampleSBSImages"], "Channel1-01-A-01.tif"
    )
    lsi_path = tests.modules.maybe_download_example_image(
        ["ExampleGrayToColor"], "AS_09125_050116030001_D03f00d0.tif"
    )
    target = bioformats.load_image(lsi_path)
    workspace = run_workspace(
        path,
        cellprofiler.modules.namesandtypes.LOAD_AS_COLOR_IMAGE,
        lsi=[
            {
                "path": lsi_path,
                "load_as_type": cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE,
                "name": "lsi",
            }
        ],
    )
    image = workspace.image_set.get_image("lsi")
    pixel_data = image.pixel_data
    assert pixel_data.shape == target.shape


def test_load_single_object():
    path = tests.modules.maybe_download_example_image(
        ["ExampleSBSImages"], "Channel1-01-A-01.tif"
    )
    lsi_path = tests.modules.maybe_download_example_image(
        ["ExampleSBSImages"], "Channel2-01-A-01.tif"
    )
    target = bioformats.load_image(lsi_path, rescale=False)
    with open(lsi_path, "rb") as fd:
        md5 = hashlib.md5(fd.read()).hexdigest()
    workspace = run_workspace(
        path,
        cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE,
        lsi=[
            {
                "path": lsi_path,
                "load_as_type": cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS,
                "name": "lsi",
            }
        ],
    )
    o = workspace.object_set.get_objects("lsi")
    assert isinstance(o, cellprofiler.object.Objects)
    counts = numpy.bincount(target.flatten())
    areas = o.areas
    assert areas[0] == counts[1]
    assert areas[1] == counts[2]
    assert areas[2] == counts[3]
    m = workspace.measurements
    assert (
        m[
            cellprofiler.measurement.IMAGE,
            cellprofiler.modules.loadimages.C_MD5_DIGEST + "_lsi",
        ]
        == md5
    )
    assert (
        m[
            cellprofiler.measurement.IMAGE,
            cellprofiler.modules.loadimages.C_WIDTH + "_lsi",
        ]
        == target.shape[1]
    )


def test_get_measurement_columns():
    p = cellprofiler.pipeline.Pipeline()
    p.clear()
    nts = [
        m
        for m in p.modules()
        if isinstance(m, cellprofiler.modules.namesandtypes.NamesAndTypes)
    ]
    assert len(nts) == 1
    m = nts[0]
    m.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    m.add_assignment()
    m.assignments[0].image_name.value = IMAGE_NAME
    m.assignments[
        1
    ].load_as_choice.value = cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS
    m.assignments[1].object_name.value = OBJECTS_NAME

    columns = m.get_measurement_columns(p)

    for ftr in (
        cellprofiler.measurement.C_FILE_NAME,
        cellprofiler.measurement.C_PATH_NAME,
        cellprofiler.measurement.C_URL,
        cellprofiler.modules.loadimages.C_MD5_DIGEST,
        cellprofiler.modules.loadimages.C_SCALING,
        cellprofiler.modules.loadimages.C_HEIGHT,
        cellprofiler.modules.loadimages.C_WIDTH,
        cellprofiler.measurement.C_SERIES,
        cellprofiler.measurement.C_FRAME,
    ):
        mname = "_".join((ftr, IMAGE_NAME))
        assert any(
            [c[0] == cellprofiler.measurement.IMAGE and c[1] == mname for c in columns]
        )

    for ftr in (
        cellprofiler.measurement.C_OBJECTS_FILE_NAME,
        cellprofiler.measurement.C_OBJECTS_PATH_NAME,
        cellprofiler.modules.loadimages.C_MD5_DIGEST,
        cellprofiler.measurement.C_OBJECTS_URL,
        cellprofiler.modules.loadimages.C_HEIGHT,
        cellprofiler.modules.loadimages.C_WIDTH,
        cellprofiler.measurement.C_OBJECTS_SERIES,
        cellprofiler.measurement.C_OBJECTS_FRAME,
        cellprofiler.measurement.C_COUNT,
    ):
        mname = "_".join((ftr, OBJECTS_NAME))
        assert any(
            [c[0] == cellprofiler.measurement.IMAGE and c[1] == mname for c in columns]
        )

    for mname in (
        cellprofiler.measurement.M_LOCATION_CENTER_X,
        cellprofiler.measurement.M_LOCATION_CENTER_Y,
    ):
        assert any([c[0] == OBJECTS_NAME and c[1] == mname for c in columns])


def test_get_categories():
    p = cellprofiler.pipeline.Pipeline()
    p.clear()
    nts = [
        m
        for m in p.modules()
        if isinstance(m, cellprofiler.modules.namesandtypes.NamesAndTypes)
    ]
    assert len(nts) == 1
    m = nts[0]
    m.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    m.assignments[0].image_name.value = IMAGE_NAME
    categories = m.get_categories(p, cellprofiler.measurement.IMAGE)
    assert not (cellprofiler.measurement.C_OBJECTS_FILE_NAME in categories)
    assert not (cellprofiler.measurement.C_OBJECTS_PATH_NAME in categories)
    assert not (cellprofiler.measurement.C_OBJECTS_URL in categories)
    assert cellprofiler.measurement.C_FILE_NAME in categories
    assert cellprofiler.measurement.C_PATH_NAME in categories
    assert cellprofiler.measurement.C_URL in categories
    assert cellprofiler.modules.loadimages.C_MD5_DIGEST in categories
    assert cellprofiler.modules.loadimages.C_SCALING in categories
    assert cellprofiler.modules.loadimages.C_WIDTH in categories
    assert cellprofiler.modules.loadimages.C_HEIGHT in categories
    assert cellprofiler.measurement.C_SERIES in categories
    assert cellprofiler.measurement.C_FRAME in categories
    m.add_assignment()
    m.assignments[
        1
    ].load_as_choice.value = cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS
    m.assignments[1].object_name.value = OBJECTS_NAME
    categories = m.get_categories(p, cellprofiler.measurement.IMAGE)
    assert cellprofiler.measurement.C_OBJECTS_FILE_NAME in categories
    assert cellprofiler.measurement.C_OBJECTS_PATH_NAME in categories
    assert cellprofiler.measurement.C_OBJECTS_URL in categories
    categories = m.get_categories(p, OBJECTS_NAME)
    assert cellprofiler.measurement.C_LOCATION in categories


def test_get_measurements():
    p = cellprofiler.pipeline.Pipeline()
    p.clear()
    nts = [
        m
        for m in p.modules()
        if isinstance(m, cellprofiler.modules.namesandtypes.NamesAndTypes)
    ]
    assert len(nts) == 1
    m = nts[0]
    m.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
    m.assignments[0].image_name.value = IMAGE_NAME
    m.add_assignment()
    m.assignments[
        1
    ].load_as_choice.value = cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS
    m.assignments[1].object_name.value = OBJECTS_NAME
    for cname in (
        cellprofiler.measurement.C_FILE_NAME,
        cellprofiler.measurement.C_PATH_NAME,
        cellprofiler.measurement.C_URL,
    ):
        mnames = m.get_measurements(p, cellprofiler.measurement.IMAGE, cname)
        assert len(mnames) == 1
        assert mnames[0] == IMAGE_NAME

    for cname in (
        cellprofiler.measurement.C_OBJECTS_FILE_NAME,
        cellprofiler.measurement.C_OBJECTS_PATH_NAME,
        cellprofiler.measurement.C_OBJECTS_URL,
        cellprofiler.measurement.C_COUNT,
    ):
        mnames = m.get_measurements(p, cellprofiler.measurement.IMAGE, cname)
        assert len(mnames) == 1
        assert mnames[0] == OBJECTS_NAME

    for cname in (
        cellprofiler.modules.loadimages.C_MD5_DIGEST,
        cellprofiler.modules.loadimages.C_SCALING,
        cellprofiler.modules.loadimages.C_HEIGHT,
        cellprofiler.modules.loadimages.C_WIDTH,
        cellprofiler.measurement.C_SERIES,
        cellprofiler.measurement.C_FRAME,
    ):
        mnames = m.get_measurements(p, cellprofiler.measurement.IMAGE, cname)
        assert len(mnames) == 2
        assert all([x in mnames for x in (IMAGE_NAME, OBJECTS_NAME)])

    mnames = m.get_measurements(p, OBJECTS_NAME, cellprofiler.measurement.C_LOCATION)
    assert all(
        [
            x in mnames
            for x in (
                cellprofiler.measurement.FTR_CENTER_X,
                cellprofiler.measurement.FTR_CENTER_Y,
            )
        ]
    )


def test_validate_single_channel():
    # regression test for issue #1429
    #
    # Single column doesn't use MATCH_BY_METADATA

    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.init_modules()
    for module in pipeline.modules():
        if isinstance(module, cellprofiler.modules.namesandtypes.NamesAndTypes):
            module.assignment_method.value = (
                cellprofiler.modules.namesandtypes.ASSIGN_RULES
            )
            module.matching_choice.value = (
                cellprofiler.modules.namesandtypes.MATCH_BY_METADATA
            )
            module.assignments[0].image_name.value = IMAGE_NAME
            module.join.build([{IMAGE_NAME: None}])
            module.validate_module(pipeline)
            break
    else:
        fail()


def test_load_grayscale_volume():
    path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "../resources/ball.tif")
    )

    workspace = run_workspace(
        path, "Grayscale image", single=True, volume=True, spacing=(0.3, 0.7, 0.7)
    )

    image = workspace.image_set.get_image("imagename")

    assert 3 == image.dimensions

    assert (9, 9, 9) == image.pixel_data.shape

    assert (0.3 / 0.7, 1.0, 1.0) == image.spacing

    assert image.pixel_data.dtype.kind == "f"


def test_load_color_volume():
    path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "../resources/ball.tif")
    )

    workspace = run_workspace(
        path, "Color image", single=True, volume=True, spacing=(0.3, 0.7, 0.7)
    )

    image = workspace.image_set.get_image("imagename")

    assert 3 == image.dimensions

    assert (9, 9, 9, 3) == image.pixel_data.shape

    assert (0.3 / 0.7, 1.0, 1.0) == image.spacing

    assert image.pixel_data.dtype.kind == "f"


def test_load_binary_mask():
    path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "../resources/ball.tif")
    )

    workspace = run_workspace(
        path, "Binary mask", single=True, volume=True, spacing=(0.3, 0.7, 0.7)
    )

    image = workspace.image_set.get_image("imagename")

    assert 3 == image.dimensions

    assert (9, 9, 9) == image.pixel_data.shape

    assert (0.3 / 0.7, 1.0, 1.0) == image.spacing

    assert image.pixel_data.dtype.kind == "b"


def test_load_illumination_function():
    path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "../resources/ball.tif")
    )

    workspace = run_workspace(
        path, "Illumination function", volume=True, spacing=(0.3, 0.7, 0.7)
    )

    image = workspace.image_set.get_image("imagename")

    assert 3 == image.dimensions

    assert (9, 9, 9) == image.pixel_data.shape

    assert (0.3 / 0.7, 1.0, 1.0) == image.spacing

    assert image.pixel_data.dtype.kind == "f"
