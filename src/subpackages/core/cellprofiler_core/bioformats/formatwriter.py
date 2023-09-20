import os
import scyjava
import numpy as np
import logging

from . import omexml
from ..utilities.java import jimport


LOGGER = logging.getLogger(__name__)

p2j = lambda v: scyjava.to_java(v)

def write_image(
    pathname,
    pixels,
    pixel_type,
    c,
    z,
    t,
    size_c,
    size_z,
    size_t,
    channel_names
):
    # https://www.javadoc.io/doc/org.openmicroscopy/ome-common/5.3.2/loci/common/services/ServiceFactory.html
    OMEXMLServiceFactory = jimport("loci.common.services.ServiceFactory")
    # https://javadoc.scijava.org/Bio-Formats/loci/formats/services/OMEXMLService.html
    OMEXMLService = jimport("loci.formats.services.OMEXMLService")
    # https://javadoc.scijava.org/Bio-Formats/loci/formats/ImageWriter.html
    ImageWriter = jimport("loci.formats.ImageWriter")
    # https://www.javadoc.io/static/org.openmicroscopy/ome-xml/6.3.1/ome/xml/meta/IMetadata.html
    IMetadata = jimport("loci.formats.meta.IMetadata")

    DimensionsOrder = jimport("ome.xml.model.enums.DimensionOrder")
    PixelType = jimport("ome.xml.model.enums.PixelType")
    PositiveInteger = jimport("ome.xml.model.primitives.PositiveInteger")

    omexml_service = OMEXMLServiceFactory().getInstance(OMEXMLService)
    # https://www.javadoc.io/static/org.openmicroscopy/ome-xml/6.3.1/ome/xml/meta/OMEXMLMetadata.html
    # https://www.javadoc.io/static/org.openmicroscopy/ome-xml/6.3.1/ome/xml/meta/MetadataStore.html
    metadata = omexml_service.createOMEXMLMetadata()
    metadata.createRoot()

    metadata.setImageName(os.path.split(pathname)[1], 0)
    metadata.setPixelsSizeX(PositiveInteger(p2j(pixels.shape[1])), 0)
    metadata.setPixelsSizeY(PositiveInteger(p2j(pixels.shape[0])), 0)
    metadata.setPixelsSizeC(PositiveInteger(p2j(size_c)), 0)
    metadata.setPixelsSizeZ(PositiveInteger(p2j(size_z)), 0)
    metadata.setPixelsSizeT(PositiveInteger(p2j(size_t)), 0)
    metadata.setPixelsBinDataBigEndian(True, 0, 0)
    metadata.setPixelsDimensionOrder(DimensionsOrder.XYCTZ, 0)
    metadata.setPixelsType(PixelType.fromString(pixel_type), 0)


    if pixels.ndim == 3:
        metadata.setPixelsSizeC(PositiveInteger(p2j(pixels.shape[2])), 0)
        metadata.setChannelSamplesPerPixel(PositiveInteger(p2j(pixels.shape[2])), 0, 0)
        omexml_service.populateOriginalMetadata(metadata, "SamplesPerPixel", str(pixels.shape[2]))
        # omexml.structured_annotations.add_original_metadata(
        #     ome.OM_SAMPLES_PER_PIXEL, str(pixels.shape[2]))
    elif size_c > 1:
        # meta.channel_count = size_c <- cant find
        metadata.setPixelsSizeC(PositiveInteger(p2j(pixels.shape[2])), 0)
        omexml_service.populateOriginalMetadata(metadata, "SamplesPerPixel", str(pixels.shape[2]))
    
    metadata.setImageID("Image:0", 0)
    metadata.setPixelsID("Pixels:0", 0)

    for i in range(size_c):
        metadata.setChannelID(f"Channel:0:{i}", 0, i)
        metadata.setChannelSamplesPerPixel(PositiveInteger(p2j(1)), 0, i)
    
    index = c + size_c * z + size_c * size_z * t
    pixel_buffer = convert_pixels_to_buffer(pixels, pixel_type)


    writer = ImageWriter()
    writer.setMetadataRetrieve(metadata)
    writer.setId(pathname)
    writer.setInterleaved(True)
    writer.saveBytes(index, pixel_buffer)
    writer.close()

def convert_pixels_to_buffer(pixels, pixel_type):
    '''Convert the pixels in the image into a buffer of the right pixel type

    pixels - a 2d monochrome or color image

    pixel_type - one of the OME pixel types

    returns a 1-d byte array
    '''
    if pixel_type == omexml.PT_UINT8:
        as_dtype = np.uint8
    elif pixel_type == omexml.PT_UINT16:
        as_dtype = "<u2"
    elif pixel_type == omexml.PT_FLOAT:
        as_dtype = "<f4"
    else:
        raise NotImplementedError("Unsupported pixel type: %d" % pixel_type)

    return np.frombuffer(np.ascontiguousarray(pixels, as_dtype).data, np.uint8)

