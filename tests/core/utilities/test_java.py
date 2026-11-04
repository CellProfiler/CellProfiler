from cellprofiler_core.utilities.java import BIOFORMATS_VERSION, jimport


def test_bioformats_readers_have_matching_dependencies(capfd):
    image_reader = jimport("loci.formats.ImageReader")
    tiff_reader = jimport("loci.formats.in.TiffReader")
    ionpath_reader = jimport("loci.formats.in.IonpathMIBITiffReader")

    for reader_class in (image_reader, tiff_reader, ionpath_reader):
        version = reader_class.class_.getPackage().getImplementationVersion()
        assert str(version) == BIOFORMATS_VERSION

    reader = image_reader()
    try:
        assert any(
            isinstance(candidate, ionpath_reader) for candidate in reader.getReaders()
        )
    finally:
        reader.close()

    output = capfd.readouterr()
    assert '"loci.formats.in.WlzReader" is not valid' not in output.err
    assert '"loci.formats.in.IonpathMIBITiffReader" is not valid' not in output.err
