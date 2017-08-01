# Python-bioformats is distributed under the GNU General Public
# License, but this file is licensed under the more permissive BSD
# license.  See the accompanying file LICENSE for details.
#
# Copyright (c) 2009-2014 Broad Institute
# All rights reserved.

"""test_omexml.py read and write OME xml

"""

from __future__ import absolute_import, unicode_literals

import datetime
import os
import unittest
import urllib
import xml.dom

import bioformats.omexml as O

def read_fully(filename):
    path = os.path.join(os.path.split(__file__)[0], filename)
    fd = open(path)
    return fd.read()

class TestOMEXML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.GROUPFILES_XML = read_fully("groupfiles.xml")

    def test_00_00_init(self):
        o = O.OMEXML()
        self.assertEquals(o.root_node.tag, O.qn(o.get_ns("ome"), "OME"))
        self.assertEquals(o.image_count, 1)

    def test_01_01_read(self):
        for xml in (self.GROUPFILES_XML, TIFF_XML):
            o = O.OMEXML(xml)

    def test_02_01_iter_children(self):
        o = O.OMEXML(TIFF_XML)
        for node, expected_tag in zip(
            o.root_node,
            (O.qn(o.get_ns("ome"), "Image"), O.qn(o.get_ns("sa"), "StructuredAnnotations"))):
            self.assertEqual(node.tag, expected_tag)

    def test_02_02_get_text(self):
        o = O.OMEXML(TIFF_XML)
        ad = o.root_node.find(
            "/".join([O.qn(o.get_ns('ome'), x) for x in ("Image", "AcquisitionDate")]))
        self.assertEqual(O.get_text(ad), "2008-02-05T17:24:46")

    def test_02_04_set_text(self):
        o = O.OMEXML(TIFF_XML)
        ad = o.root_node.find("/".join(
            [O.qn(o.get_ns('ome'), x) for x in ("Image", "AcquisitionDate")]))
        im = o.root_node.find(O.qn(o.get_ns("ome"), "Image"))
        O.set_text(im, "Foo")
        self.assertEqual(O.get_text(im), "Foo")
        O.set_text(ad, "Bar")
        self.assertEqual(O.get_text(ad), "Bar")

    def test_03_01_get_image_count(self):
        for xml, count in ((self.GROUPFILES_XML, 576), (TIFF_XML,1)):
            o = O.OMEXML(xml)
            self.assertEqual(o.image_count, count)

    def test_03_02_set_image_count(self):
        o = O.OMEXML(TIFF_XML)
        o.image_count = 2
        self.assertEqual(len(o.root_node.findall(O.qn(o.get_ns("ome"), "Image"))), 2)

    def test_03_03_image(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        self.assertEquals(o.image_count, 576)
        for i in range(576):
            im = o.image(i)
            self.assertEqual(im.node.get("ID"), "Image:%d" % i)

    def test_03_04_structured_annotations(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.structured_annotations.node.tag,
                         O.qn(o.get_ns("sa"), "StructuredAnnotations"))

    def test_04_01_image_get_id(self):
        o =  O.OMEXML(TIFF_XML)
        self.assertEquals(o.image(0).ID, "Image:0")

    def test_04_02_image_set_id(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).ID = "Foo"
        self.assertEquals(o.image(0).node.get("ID"), "Foo")

    def test_04_03_image_get_name(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEquals(o.image(0).Name, "Channel1-01-A-01.tif")

    def test_04_04_image_set_name(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).Name = "Foo"
        self.assertEquals(o.image(0).node.get("Name"), "Foo")

    def test_04_05_image_get_acquisition_date(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).AcquisitionDate, "2008-02-05T17:24:46")

    def test_04_06_image_set_acquisition_date(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).AcquisitionDate = "2011-12-21T11:04:14.903000"
        self.assertEqual(o.image(0).AcquisitionDate, "2011-12-21T11:04:14.903000")

    def test_04_07_image_1_acquisition_date(self):
        # regression test of #38
        o = O.OMEXML()
        o.set_image_count(2)
        date_1 = "2011-12-21T11:04:14.903000"
        date_2 = "2015-10-13T09:57:00.000000"
        o.image(0).AcquisitionDate = date_1
        o.image(1).AcquisitionDate = date_2
        self.assertEqual(o.image(0).AcquisitionDate, date_1)
        self.assertEqual(o.image(1).AcquisitionDate, date_2)

    def test_05_01_pixels_get_id(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).Pixels.ID, "Pixels:0")

    def test_05_02_pixels_set_id(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).Pixels.ID = "Foo"
        self.assertEqual(o.image(0).Pixels.ID, "Foo")

    def test_05_03_pixels_get_dimension_order(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).Pixels.DimensionOrder, O.DO_XYCZT)

    def test_05_04_pixels_set_dimension_order(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).Pixels.DimensionOrder = O.DO_XYZCT
        self.assertEqual(o.image(0).Pixels.DimensionOrder, O.DO_XYZCT)

    def test_05_05_pixels_get_pixel_type(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).Pixels.PixelType, O.PT_UINT8)

    def test_05_06_pixels_set_pixel_type(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).Pixels.PixelType = O.PT_FLOAT
        self.assertEqual(o.image(0).Pixels.PixelType, O.PT_FLOAT)

    def test_05_07_pixels_get_size_x(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).Pixels.SizeX, 640)

    def test_05_08_pixels_set_size_x(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).Pixels.SizeX = 480
        self.assertEqual(o.image(0).Pixels.SizeX, 480)

    def test_05_09_pixels_get_size_y(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).Pixels.SizeY, 512)

    def test_05_10_pixels_set_size_y(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).Pixels.SizeY = 480
        self.assertEqual(o.image(0).Pixels.SizeY, 480)

    def test_05_11_pixels_get_size_z(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).Pixels.SizeZ, 1)

    def test_05_12_pixels_set_size_z(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).Pixels.SizeZ = 2
        self.assertEqual(o.image(0).Pixels.SizeZ, 2)

    def test_05_13_pixels_get_size_c(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).Pixels.SizeC, 2)

    def test_05_14_pixels_set_size_c(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).Pixels.SizeC = 3
        self.assertEqual(o.image(0).Pixels.SizeC, 3)

    def test_05_15_pixels_get_size_t(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).Pixels.SizeT, 3)

    def test_05_16_pixels_set_size_t(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).Pixels.SizeT = 1
        self.assertEqual(o.image(0).Pixels.SizeT, 1)

    def test_05_17_pixels_get_channel_count(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).Pixels.channel_count, 1)

    def test_05_18_pixels_set_channel_count(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).Pixels.channel_count = 2
        self.assertEqual(
            len(o.image(0).Pixels.node.findall(O.qn(o.get_ns("ome"), "Channel"))), 2)

    def test_06_01_channel_get_id(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).Pixels.Channel(0).ID, "Channel:0:0")

    def test_06_02_channel_set_id(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).Pixels.Channel(0).ID = "Red"
        self.assertEqual(o.image(0).Pixels.Channel(0).ID, "Red")

    def test_06_03_channel_get_name(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).Pixels.Channel(0).Name, "Actin")

    def test_06_04_channel_set_Name(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).Pixels.Channel(0).Name = "PI"
        self.assertEqual(o.image(0).Pixels.Channel(0).Name, "PI")

    def test_06_04_channel_get_samples_per_pixel(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).Pixels.Channel(0).SamplesPerPixel, 1)

    def test_06_04_channel_set_samples_per_pixel(self):
        o = O.OMEXML(TIFF_XML)
        o.image(0).Pixels.Channel(0).SamplesPerPixel = 3
        self.assertEqual(o.image(0).Pixels.Channel(0).SamplesPerPixel, 3)

    def test_07_01_sa_get_item(self):
        o = O.OMEXML(TIFF_XML)
        a = o.structured_annotations["Annotation:4"]
        self.assertEqual(a.tag, O.qn(o.get_ns("sa"), "XMLAnnotation"))
        values = a.findall(O.qn(o.get_ns("sa"), "Value"))
        self.assertEqual(len(values), 1)
        oms = values[0].findall(O.qn(O.NS_ORIGINAL_METADATA, "OriginalMetadata"))
        self.assertEqual(len(oms), 1)
        keys = oms[0].findall(O.qn(O.NS_ORIGINAL_METADATA, "Key"))
        self.assertEqual(len(keys), 1)
        self.assertEqual(O.get_text(keys[0]), "XResolution")
        values = oms[0].findall(O.qn(O.NS_ORIGINAL_METADATA, "Value"))
        self.assertEqual(len(values), 1)
        self.assertEqual(O.get_text(values[0]), "72")

    def test_07_02_01_sa_keys(self):
        keys = O.OMEXML(TIFF_XML).structured_annotations.keys()
        for i in range(21):
            self.assertTrue("Annotation:%d" %i in keys)

    def test_07_02_02_sa_has_key(self):
        o = O.OMEXML(TIFF_XML)
        for i in range(20):
            self.assertTrue("Annotation:%d" %i in o.structured_annotations)
        self.assertFalse("Foo" in o.structured_annotations)

    def test_07_03_om_getitem(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.structured_annotations.OriginalMetadata["MetaMorph"], "no")

    def test_07_04_01_om_keys(self):
        o = O.OMEXML(TIFF_XML)
        keys = o.structured_annotations.OriginalMetadata.keys()
        self.assertEqual(len(keys), 21)
        for k in ("DateTime", "Software", "YResolution"):
            self.assertTrue(k in keys)

    def test_07_04_02_om_has_key(self):
        o = O.OMEXML(TIFF_XML)
        om = o.structured_annotations.OriginalMetadata
        for k in ("DateTime", "Software", "YResolution"):
            self.assertTrue(k in om)
        self.assertFalse("Foo" in om)

    def test_07_05_om_setitem(self):
        o = O.OMEXML()
        o.structured_annotations.OriginalMetadata["Foo"] = "Bar"
        sa = o.structured_annotations.node
        a = sa.findall(O.qn(o.get_ns("sa"), "XMLAnnotation"))
        self.assertEqual(len(a), 1)
        vs = a[0].findall(O.qn(o.get_ns("sa"), "Value"))
        self.assertEqual(len(vs), 1)
        om = vs[0].findall(O.qn(O.NS_ORIGINAL_METADATA, "OriginalMetadata"))
        self.assertEqual(len(om), 1)
        k = om[0].findall(O.qn(O.NS_ORIGINAL_METADATA, "Key"))
        self.assertEqual(len(k), 1)
        self.assertEqual(O.get_text(k[0]), "Foo")
        v = om[0].findall(O.qn(O.NS_ORIGINAL_METADATA, "Value"))
        self.assertEqual(len(v), 1)
        self.assertEqual(O.get_text(v[0]), "Bar")

    def test_08_01_get_plate(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        plate = o.plates[0]
        self.assertEqual(plate.ID, "Plate:0")

    def test_08_02_get_plate_count(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        self.assertEqual(len(o.plates), 1)

    def test_08_02_new_plate(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        o.plates.newPlate("MyPlate", "Plate:1")
        self.assertEqual(o.plates[1].ID, "Plate:1")
        self.assertEqual(o.plates[1].Name, "MyPlate")

    def test_08_03_plate_iter(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        nplates = 5
        for i in range(1, nplates):
            o.plates.newPlate("MyPlate%d" %i, "Plate:%d" % i)
        for i, plate in enumerate(o.plates):
            self.assertEqual(plate.ID, "Plate:%d" %i)

    def test_08_04_plate_slice(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        for i in range(1, 5):
            o.plates.newPlate("MyPlate%d" %i, "Plate:%d" % i)
        plates = o.plates[2:-1]
        self.assertEqual(len(plates), 2)
        self.assertTrue(all([plate.ID == "Plate:%d" % (i+2)
                             for i, plate in enumerate(plates)]))

        plates = o.plates[-4:4]
        self.assertEqual(len(plates), 3)
        self.assertTrue(all([plate.ID == "Plate:%d" % (i+1)
                             for i, plate in enumerate(plates)]))

    def test_09_01_plate_get_name(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        self.assertEqual(o.plates[0].Name, "TimePoint_1")

    def test_09_02_plate_set_status(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.Status = "Gronked"
        self.assertEqual(plate.node.get("Status"),"Gronked")

    def test_09_03_plate_get_status(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.node.set("Status", "Gronked")
        self.assertEqual(plate.Status,"Gronked")

    def test_09_04_plate_get_external_identifier(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.node.set("ExternalIdentifier", "xyz")
        self.assertEqual(plate.ExternalIdentifier, "xyz")

    def test_09_05_plate_set_external_identifier(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.ExternalIdentifier = "xyz"
        self.assertEqual(plate.node.get("ExternalIdentifier"), "xyz")

    def test_09_06_plate_get_column_naming_convention(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.node.set("ColumnNamingConvention", O.NC_LETTER)
        self.assertEqual(plate.ColumnNamingConvention, O.NC_LETTER)

    def test_09_07_plate_set_column_naming_convention(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.ColumnNamingConvention = O.NC_NUMBER
        self.assertEqual(plate.ColumnNamingConvention, O.NC_NUMBER)

    def test_09_08_plate_get_row_naming_convention(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.node.set("RowNamingConvention", O.NC_LETTER)
        self.assertEqual(plate.RowNamingConvention, O.NC_LETTER)

    def test_09_09_plate_set_row_naming_convention(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.RowNamingConvention = O.NC_NUMBER
        self.assertEqual(plate.RowNamingConvention, O.NC_NUMBER)

    def test_09_10_plate_get_well_origin_x(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.node.set("WellOriginX", "4.8")
        self.assertEqual(plate.WellOriginX, 4.8)

    def test_09_11_plate_set_well_origin_x(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.WellOriginX = 3.5
        self.assertEqual(plate.node.get("WellOriginX"), "3.5")

    def test_09_12_plate_get_well_origin_y(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.node.set("WellOriginY", "5.8")
        self.assertEqual(plate.WellOriginY, 5.8)

    def test_09_13_plate_set_well_origin_y(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.WellOriginY = 3.5
        self.assertEqual(plate.node.get("WellOriginY"), "3.5")

    def test_09_14_plate_get_rows(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.node.set("Rows", "8")
        self.assertEqual(plate.Rows, 8)

    def test_09_15_plate_set_rows(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.Rows = 16
        self.assertEqual(plate.node.get("Rows"), "16")

    def test_09_16_plate_get_columns(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.node.set("Columns", "12")
        self.assertEqual(plate.Columns, 12)

    def test_09_15_plate_set_columns(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo", "Bar")
        plate.Columns = 24
        self.assertEqual(plate.node.get("Columns"), "24")

    def test_10_01_wells_len(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        self.assertEqual(len(o.plates[0].Well), 96)

    def test_10_02_wells_by_row_and_column(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        w = o.plates[0].Well[1,3]
        self.assertEqual(w.ID, "Well:0:15")

    def test_10_03_wells_by_index(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        w = o.plates[0].Well[2]
        self.assertEqual(w.ID, "Well:0:2")

    def test_10_04_wells_by_name(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        w = o.plates[0].Well["C05"]
        self.assertEqual(w.Row, 2)
        self.assertEqual(w.Column, 4)

    def test_10_05_wells_by_id(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        w = o.plates[0].Well["Well:0:3"]
        self.assertEqual(w.Row, 0)
        self.assertEqual(w.Column, 3)

    def test_10_06_wells_by_slice(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        for i, w in enumerate(o.plates[0].Well[1::12]):
            self.assertEqual(w.Column, 1)
            self.assertEqual(w.Row, i)

    def test_10_07_iter_wells(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        for i, name in enumerate(o.plates[0].Well):
            row = int(i / 12)
            column = i % 12
            self.assertEqual(name, "ABCDEFGH"[row] + "%02d" % (column+1))
        self.assertEqual(name, "H12")

    def test_10_08_new_well(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("Foo","Bar")
        plate.Well.new(4,5, "xyz")
        w = plate.Well[0]
        self.assertEqual(w.node.get("Row"), "4")
        self.assertEqual(w.node.get("Column"), "5")
        self.assertEqual(w.node.get("ID"), "xyz")

    def test_11_01_get_Column(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        self.assertEqual(o.plates[0].Well["B05"].Column, 4)

    def test_11_02_get_Row(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        self.assertEqual(o.plates[0].Well["B05"].Row, 1)

    def test_11_03_get_external_description(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("foo","bar")
        w = plate.Well.new(4,5, "xyz")
        w.node.set("ExternalDescription", "ijk")
        self.assertEqual(w.ExternalDescription, "ijk")

    def test_11_04_set_external_description(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("foo","bar")
        w = plate.Well.new(4,5, "xyz")
        w.ExternalDescription = "LMO"
        self.assertEqual(w.node.get("ExternalDescription"), "LMO")

    def test_11_05_get_external_identifier(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("foo","bar")
        w = plate.Well.new(4,5, "xyz")
        w.node.set("ExternalIdentifier", "ijk")
        self.assertEqual(w.ExternalIdentifier, "ijk")

    def test_11_06_set_external_identifier(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("foo","bar")
        w = plate.Well.new(4,5, "xyz")
        w.ExternalIdentifier = "LMO"
        self.assertEqual(w.node.get("ExternalIdentifier"), "LMO")

    def test_12_01_get_sample_len(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        self.assertEqual(len(o.plates[0].Well[0].Sample), 6)

    def test_12_02_get_sample_item(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        s = o.plates[0].Well[0].Sample[2]
        self.assertEqual(s.node.get("ID"), "WellSample:0:0:2")

    def test_12_03_get_sample_item_slice(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        for i, s in enumerate(o.plates[0].Well[0].Sample[1::2]):
            self.assertEqual(s.node.get("ID"), "WellSample:0:0:%d" % (i * 2 + 1))

    def test_12_04_iter_sample_item(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        for i, s in enumerate(o.plates[0].Well[0].Sample):
            self.assertEqual(s.node.get("ID"), "WellSample:0:0:%d" % i)

    def test_12_05_new_sample_item(self):
        o = O.OMEXML()
        plate = o.plates.newPlate("foo","bar")
        w = plate.Well.new(4,5, "xyz")
        w.Sample.new("ooo")
        w.Sample.new("ppp")
        sample_nodes = w.node.findall(O.qn(o.get_ns("spw"), "WellSample"))
        self.assertEqual(len(sample_nodes), 2)
        self.assertEqual(sample_nodes[0].get("ID"), "ooo")
        self.assertEqual(sample_nodes[1].get("ID"), "ppp")
        self.assertEqual(sample_nodes[0].get("Index"), "0")
        self.assertEqual(sample_nodes[1].get("Index"), "1")

    def test_13_01_get_sample_id(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        self.assertEqual(o.plates[0].Well['A02'].Sample[3].ID, "WellSample:0:1:3")

    def test_13_02_set_sample_id(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        ws = o.plates[0].Well['A02'].Sample[3]
        ws.ID = "Foo"
        self.assertEqual(ws.node.get("ID"), "Foo")

    def test_13_03_get_position_x(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        self.assertEqual(o.plates[0].Well['A01'].Sample[4].PositionX, 402.5)

    def test_13_04_set_position_x(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        ws = o.plates[0].Well['A02'].Sample[3]
        ws.PositionX = 201.75
        self.assertEqual(ws.node.get("PositionX"), "201.75")

    def test_13_05_get_position_y(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        self.assertEqual(o.plates[0].Well['A01'].Sample[4].PositionY, 204.25)

    def test_13_06_set_position_y(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        ws = o.plates[0].Well['A02'].Sample[3]
        ws.PositionY = 14.5
        self.assertEqual(ws.node.get("PositionY"), "14.5")

    def test_13_07_get_timepoint(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        self.assertEqual(o.plates[0].Well['A01'].Sample[1].Timepoint, '2011-12-27T08:24:29.960000')

    def test_13_08_set_timepoint(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        ws = o.plates[0].Well['A02'].Sample[3]
        now = datetime.datetime.now()
        now_string = now.isoformat()
        ws.Timepoint = now
        self.assertEqual(ws.node.get("Timepoint"), now_string)
        ws = o.plates[0].Well['A03'].Sample[4]
        ws.Timepoint = now_string
        self.assertEqual(ws.node.get("Timepoint"), now_string)

    def test_13_09_get_index(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        ws = o.plates[0].Well['A02'].Sample[3]
        self.assertEqual(ws.Index, 9)

    def test_13_10_set_index(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        ws = o.plates[0].Well['A02'].Sample[3]
        ws.Index = 301
        self.assertEqual(ws.Index, 301)

    def test_13_11_get_image_ref(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        ws = o.plates[0].Well['A02'].Sample[3]
        self.assertEqual(ws.ImageRef, "Image:9")
        ref = ws.node.find(O.qn(o.get_ns("spw"), "ImageRef"))
        ws.node.remove(ref)
        self.assertTrue(ws.ImageRef is None)

    def test_13_12_set_image_ref(self):
        o = O.OMEXML(self.GROUPFILES_XML)
        ws = o.plates[0].Well['A02'].Sample[3]
        ws.ImageRef = "Foo"
        self.assertEqual(ws.node.find(O.qn(o.get_ns("spw"), "ImageRef")).get("ID"), "Foo")

    def test_14_01_get_plane_count(self):
        o = O.OMEXML(TIFF_XML)
        self.assertEqual(o.image(0).Pixels.plane_count, 1)

    def test_14_02_set_plane_count(self):
        o = O.OMEXML()
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        pixels.plane_count = 2
        self.assertEqual(len(pixels.node.findall(O.qn(o.get_ns('ome'), "Plane"))), 2)

    def test_14_03_get_the_c(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        self.assertTrue(isinstance(plane, O.OMEXML.Plane))
        plane.node.set("TheC", "15")
        self.assertEqual(plane.TheC, 15)

    def test_14_04_get_the_z(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        self.assertTrue(isinstance(plane, O.OMEXML.Plane))
        plane.node.set("TheZ", "10")
        self.assertEqual(plane.TheZ, 10)

    def test_14_05_get_the_t(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        self.assertTrue(isinstance(plane, O.OMEXML.Plane))
        plane.node.set("TheT", "9")
        self.assertEqual(plane.TheT, 9)

    def test_14_06_set_the_c(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        plane.TheC = 5
        self.assertEqual(int(plane.node.get("TheC")), 5)

    def test_14_07_set_the_z(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        plane.TheZ = 6
        self.assertEqual(int(plane.node.get("TheZ")), 6)

    def test_14_08_set_the_t(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        plane.TheC = 7
        self.assertEqual(int(plane.node.get("TheC")), 7)

    def test_14_09_get_delta_t(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        self.assertEqual(plane.DeltaT, 1.25)

    def test_14_10_get_exposure_time(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        self.assertEqual(plane.ExposureTime, 0.25)

    def test_14_11_get_position_x(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        self.assertEqual(plane.PositionX, 3.5)

    def test_14_12_get_position_y(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        self.assertEqual(plane.PositionY, 4.75)

    def test_14_13_get_position_z(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        self.assertEqual(plane.PositionZ, 2.25)

    def test_14_14_set_delta_t(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        plane.DeltaT = 1.25
        self.assertEqual(float(plane.node.get("DeltaT")), 1.25)

    def test_14_15_set_position_x(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        plane.PositionX = 5.5
        self.assertEqual(float(plane.node.get("PositionX")), 5.5)

    def test_14_16_set_position_y(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        plane.PositionY = 6.5
        self.assertEqual(float(plane.node.get("PositionY")), 6.5)

    def test_14_17_set_position_z(self):
        o = O.OMEXML(TIFF_XML)
        pixels = o.image(0).Pixels
        self.assertTrue(isinstance(pixels, O.OMEXML.Pixels))
        plane = pixels.Plane(0)
        plane.PositionZ = 7.5
        self.assertEqual(float(plane.node.get("PositionZ")), 7.5)

TIFF_XML = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2013-06"
	xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2013-06 http://www.openmicroscopy.org/Schemas/OME/2013-06/ome.xsd">
	<Image ID="Image:0" Name="Channel1-01-A-01.tif">
		<AcquisitionDate>2008-02-05T17:24:46</AcquisitionDate>
		<Pixels DimensionOrder="XYCZT" ID="Pixels:0" PhysicalSizeX="352.77777777777777"
			PhysicalSizeY="352.77777777777777" SizeC="2" SizeT="3" SizeX="640"
			SizeY="512" SizeZ="1" Type="uint8">
			<Channel ID="Channel:0:0" SamplesPerPixel="1" Name="Actin">
				<LightPath />
			</Channel>
			<BinData xmlns="http://www.openmicroscopy.org/Schemas/BinaryFile/2013-06"
				BigEndian="true" Length="0" />
                        <Plane TheC="0" TheT="0" TheZ="0" DeltaT="1.25" ExposureTime="0.25" PositionX="3.5" PositionY="4.75" PositionZ="2.25"/>
		</Pixels>
	</Image>
	<StructuredAnnotations
		xmlns="http://www.openmicroscopy.org/Schemas/SA/2013-06">
		<XMLAnnotation ID="Annotation:0">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>BIG_TIFF</Key>
					<Value>false</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:1">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>ImageLength</Key>
					<Value>640</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:2">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>MetaDataPhotometricInterpretation</Key>
					<Value>Monochrome</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:3">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>PhotometricInterpretation</Key>
					<Value>Palette</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:4">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>XResolution</Key>
					<Value>72</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:5">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>NewSubfileType</Key>
					<Value>0</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:6">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>DateTime</Key>
					<Value>2008:02:05 17:24:46</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:7">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>MetaMorph</Key>
					<Value>no</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:8">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>Software</Key>
					<Value>Adobe Photoshop CS2 Macintosh</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:9">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>YResolution</Key>
					<Value>72</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:10">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>PIXEL_X_DIMENSION</Key>
					<Value>640</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:11">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>ResolutionUnit</Key>
					<Value>Inch</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:12">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>PIXEL_Y_DIMENSION</Key>
					<Value>640</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:13">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>Compression</Key>
					<Value>Uncompressed</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:14">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>LITTLE_ENDIAN</Key>
					<Value>false</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:15">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>COLOR_SPACE</Key>
					<Value>1</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:16">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>BitsPerSample</Key>
					<Value>8</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:17">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>NumberOfChannels</Key>
					<Value>3</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:18">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>Orientation</Key>
					<Value>1st row - top; 1st column - left</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:19">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>SamplesPerPixel</Key>
					<Value>1</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
		<XMLAnnotation ID="Annotation:20">
			<Value>
				<OriginalMetadata xmlns="openmicroscopy.org/OriginalMetadata">
					<Key>ImageWidth</Key>
					<Value>640</Value>
				</OriginalMetadata>
			</Value>
		</XMLAnnotation>
	</StructuredAnnotations>
</OME>"""
