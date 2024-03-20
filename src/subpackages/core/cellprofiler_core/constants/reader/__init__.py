import re

builtin_readers = {
    "imageio_reader": "ImageIOReader",
    "imageio_reader_v3": "ImageIOReaderV3",
    "ngff_reader": "NGFFReader",
    "bioformats_reader": "BioformatsReader",
    "gcs_reader": "GcsReader",
}
# All successfully loaded reader classes. Maps name:class
ALL_READERS = dict()
# Reader classes that failed to load. Maps name:exception str
BAD_READERS = dict()
# Active reader classes (ALL_READERS that aren't disabled by user). Maps name:class
AVAILABLE_READERS = dict()

ZARR_FILETYPE = re.compile(r"(?<=\.zarr)", flags=re.IGNORECASE)

