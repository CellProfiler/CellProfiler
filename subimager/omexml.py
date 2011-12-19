"""omexml.py read and write OME xml

"""
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2011
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org
#
import xml.dom
from xml.dom.minidom import parseString
import datetime
import uuid

def xsd_now():
    '''Return the current time in xsd:dateTime format'''
    return datetime.datetime.now().isoformat()

DEFAULT_NOW = xsd_now()
#
# The namespaces
#
NS_OME = "http://www.openmicroscopy.org/Schemas/OME/2011-06"
NS_BINARY_FILE = "http://www.openmicroscopy.org/Schemas/BinaryFile/2011-06"
NS_SA = "http://www.openmicroscopy.org/Schemas/SA/2011-06"

default_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="%(NS_OME)s" 
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2011-06 http://www.openmicroscopy.org/Schemas/OME/2011-06/ome.xsd">
  <Image ID="Image:0" Name="default.png">
    <AcquiredDate>%(DEFAULT_NOW)s</AcquiredDate>
    <Pixels DimensionOrder="XYCTZ" 
            ID="Pixels:0" 
            SizeC="1" 
            SizeT="1" 
            SizeX="512" 
            SizeY="512"
            SizeZ="1"
            Type="uint8">
<Channel ID="Channel:0:0" SamplesPerPixel="1">
        <LightPath/>
      </Channel>
      <BinData xmlns="%(NS_BINARY_FILE)s" 
       BigEndian="false" Length="0"/>
    </Pixels>
  </Image>
  <StructuredAnnotations xmlns="%(NS_SA)s"/>
</OME>
""" % globals()
#
# These are the OME-XML pixel types - not all supported by subimager
#
PT_INT8 = "int8"
PT_INT16 = "int16"
PT_INT32 = "int32"
PT_UINT8 = "uint8"
PT_UINT16 = "uint16"
PT_UINT32 = "uint32"
PT_FLOAT = "float"
PT_BIT = "bit"
PT_DOUBLE = "double"
PT_COMPLEX = "complex"
PT_DOUBLECOMPLEX = "double-complex"
#
# The allowed dimension types
#
DO_XYZCT = "XYZCT"
DO_XYZTC = "XYZTC"
DO_XYCTZ = "XYCTZ"
DO_XYCZT = "XYCZT"
DO_XYTCZ = "XYTCZ"
DO_XYTZC = "XYTZC"

def get_text(node):
    '''Get the contents of text nodes in a parent node'''
    node.normalize()
    child = node.firstChild
    while child is not None:
        if child.nodeType == xml.dom.Node.TEXT_NODE:
            return child.data
        child = child.nextSibling
    return None

def set_text(node, text):
    '''Set the text of a parent'''
    node.normalize()
    child = node.firstChild
    while child is not None:
        if child.nodeType == xml.dom.Node.TEXT_NODE:
            child.data = text
            return
        child = child.nextSibling
    dom = get_dom(node)
    node.appendChild(dom.createTextNode(text))

def get_dom(node):
    '''Get the document node given a rooted node'''
    while node.nodeType != xml.dom.Node.DOCUMENT_NODE:
        node = node.parentNode
    return node

def insert_after(prior, subsequent):
    '''Insert an unrooted node after a prior, rooted node'''
    if prior.nextSibling is None:
        prior.parentNode.appendChild(subsequent)
    else:
        prior.parentNode.insertBefore(subsequent, prior.nextSibling)

class OMEXML(object):
    def __init__(self, xml=default_xml):
        self.dom = parseString(xml)
        
    def __str__(self):
        return self.dom.toprettyxml()

    @property
    def root_node(self):
        return self.dom.getElementsByTagName("OME")[0]
    
    def get_image_count(self):
        '''The number of images (= series) specified by the XML'''
        return len(self.root_node.getElementsByTagName("Image"))
    
    def set_image_count(self, value):
        '''Add or remove image nodes as needed'''
        assert value > 0
        root = self.root_node
        while(self.image_count > value):
            last = self.root_node.getElementsByTagNameNS(NS_OME, "Image")[-1]
            root.removeChild(last)
            last.unlink()
        while(self.image_count < value):
            last_image = self.root_node.getElementsByTagNameNS(NS_OME,"Image")[-1]
            new_image = self.Image(self.dom.createElementNS(NS_OME, "Image"))
            insert_after(last_image, new_image.node)
            new_image.ID = str(uuid.uuid4())
            new_image.Name = "default.png"
            new_image.AcquiredDate = xsd_now()
            new_pixels = self.Pixels(self.dom.createElementNS(NS_OME, "Pixels"))
            new_image.node.appendChild(new_pixels.node)
            new_pixels.ID = str(uuid.uuid4())
            new_pixels.DimensionOrder = DO_XYCTZ
            new_pixels.PixelType = PT_UINT8
            new_pixels.SizeC = 1
            new_pixels.SizeT = 1
            new_pixels.SizeX = 512
            new_pixels.SizeY = 512
            new_pixels.SizeZ = 1
            new_channel = self.Channel(self.dom.createElementNS(NS_OME, "Channel"))
            new_pixels.node.appendChild(new_channel.node)
            new_channel.ID = "Channel%d:0" % self.image_count
            new_channel.Name = new_channel.ID
            new_channel.SamplesPerPixel = 1
            
    image_count = property(get_image_count, set_image_count)
    
    class Image(object):
        def __init__(self, node):
            self.node = node
            
        def get_ID(self):
            return self.node.getAttribute("ID")
        def set_ID(self, value):
            self.node.setAttribute("ID", value)
        ID = property(get_ID, set_ID)

        def get_Name(self):
            return self.node.getAttribute("Name")
        def set_Name(self, value):
            self.node.setAttribute("Name", value)
        Name = property(get_Name, set_Name)
        
        def get_AcquiredDate(self):
            acquired_dates = self.node.getElementsByTagNameNS(NS_OME, "AcquiredDate")
            if len(acquired_dates) == 0:
                return None
            return get_text(acquired_dates[0])
        
        def set_AcquiredDate(self, date):
            acquired_dates = self.node.getElementsByTagNameNS(NS_OME, "AcquiredDate")
            if len(acquired_dates) == 0:
                dom = get_dom(self.node)
                acquired_date = dom.createElementNS(NS_OME, "AcquiredDate")
                self.node.appendChild(acquired_date)
            else:
                acquired_date = acquired_dates[0]
            set_text(acquired_date, date)
        AcquiredDate = property(get_AcquiredDate, set_AcquiredDate)
            
        @property
        def Pixels(self):
            return self.Pixels(self.node.getElementsByTagNameNS(NS_OME, "Pixels")[0])
        
    def image(self, index=0):
        '''Return an image node by index'''
        return self.Image(self.root_node.getElementsByTagNameNS(NS_OME, "Image")[index])
    
    class Channel(object):
        def __init__(self, node):
            self.node = node
            
        def get_ID(self):
            return self.node.getAttribute("ID")
        def set_ID(self, value):
            self.node.setAttribute("ID", value)
        ID = property(get_ID, set_ID)
        
        def get_Name(self):
            return self.node.getAttribute("Name")
        def set_Name(self, value):
            self.node.setAttribute("Name", value)
        Name = property(get_Name, set_Name)
        
        def get_SamplesPerPixel(self):
            return int(self.node.getAttribute("SamplesPerPixel"))
        def set_SamplesPerPixel(self, value):
            self.node.setAttribute("SamplesPerPixel", str(value))
        SamplesPerPixel = property(get_SamplesPerPixel, set_SamplesPerPixel)
        
    class Pixels(object):
        def __init__(self, node):
            self.node = node
            
        def get_ID(self):
            return self.node.getAttribute("ID")
        def set_ID(self, value):
            self.node.setAttribute("ID", value)
        ID = property(get_ID, set_ID)
        
        def get_DimensionOrder(self):
            return self.node.getAttribute("DimensionOrder")
        def set_DimensionOrder(self, value):
            self.node.setAttribute("DimensionOrder", value)
        DimensionOrder = property(get_DimensionOrder, set_DimensionOrder)
        
        def get_PixelType(self):
            return self.node.getAttribute("Type")
        def set_PixelType(self, value):
            self.node.setAttribute("Type", value)
        PixelType = property(get_PixelType, set_PixelType)
        
        def get_SizeX(self):
            return int(self.node.getAttribute("SizeX"))
        def set_SizeX(self, value):
            self.node.setAttribute("SizeX", str(value))
        SizeX = property(get_SizeX, set_SizeX)
        
        def get_SizeY(self):
            return int(self.node.getAttribute("SizeY"))
        def set_SizeY(self, value):
            self.node.setAttribute("SizeY", str(value))
        SizeY = property(get_SizeX, set_SizeY)
        
        def get_SizeZ(self):
            return int(self.node.getAttribute("SizeZ"))
        def set_SizeZ(self, value):
            self.node.setAttribute("SizeZ", str(value))
        SizeZ = property(get_SizeZ, set_SizeZ)
        
        def get_SizeT(self):
            return int(self.node.getAttribute("SizeT"))
        def set_SizeT(self, value):
            self.node.setAttribute("SizeT", str(value))
        SizeT = property(get_SizeT, set_SizeT)
        
        def get_SizeC(self):
            return int(self.node.getAttribute("SizeC"))
        def set_SizeC(self, value):
            self.node.setAttribute("SizeC", str(value))
        SizeC = property(get_SizeC, set_SizeC)
        
        def get_channel_count(self):
            return len(self.node.getElementsByNameNS(NS_OME, "Channel"))
        
        def set_channel_count(self, value):
            assert value > 0
            dom = get_dom(self.node)
            while(self.channel_count > value):
                last = self.node.getElementsByTagNameNS(NS_OME, "Channel")[-1]
                self.node.removeChild(last)
                last.unlink()
            
            while(self.channel_count < value):
                last = self.node.getElementsByTagNameNS(NS_OME, "Channel")[-1]
                new_channel = OMEXML.Channel(dom.createElementNS(NS_OME, "Channel"))
                insert_after(last, new_channel.node)
                new_channel.ID = str(uuid.uuid4())
                new_channel.Name = new_channel.ID
                new_channel.SamplesPerPixel = 1
                
        channel_count = property(get_channel_count, set_channel_count)
