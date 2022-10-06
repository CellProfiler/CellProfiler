import scyjava

def write_image(
    filename,
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
    omexml = scyjava.jimport()
    omexml.image(0).Name = os.path.split(pathname)[1]
    p = omexml.image(0).Pixels
    assert isinstance(p, ome.OMEXML.Pixels)
    p.SizeX = pixels.shape[1]
    p.SizeY = pixels.shape[0]
    p.SizeC = size_c
    p.SizeT = size_t
    p.SizeZ = size_z
    p.DimensionOrder = ome.DO_XYCZT
    p.PixelType = pixel_type
    index = c + size_c * z + size_c * size_z * t
    if pixels.ndim == 3:
        p.SizeC = pixels.shape[2]
        p.Channel(0).SamplesPerPixel = pixels.shape[2]
        omexml.structured_annotations.add_original_metadata(
            ome.OM_SAMPLES_PER_PIXEL, str(pixels.shape[2]))
    elif size_c > 1:
        p.channel_count = size_c

    pixel_buffer = convert_pixels_to_buffer(pixels, pixel_type)
    xml = omexml.to_xml()
    script = """
    importClass(Packages.loci.formats.services.OMEXMLService,
                Packages.loci.common.services.ServiceFactory,
                Packages.loci.formats.ImageWriter);
    var service = new ServiceFactory().getInstance(OMEXMLService);
    var metadata = service.createOMEXMLMetadata(xml);
    var writer = new ImageWriter();
    writer.setMetadataRetrieve(metadata);
    writer.setId(path);
    writer.setInterleaved(true);
    writer.saveBytes(index, buffer);
    writer.close();
    """
