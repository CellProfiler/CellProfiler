UIC1_TAG = 33628
UIC2_TAG = 33629
UIC3_TAG = 33630
UIC4_TAG = 33631
C_MD5_DIGEST = "MD5Digest"
C_SCALING = "Scaling"
C_HEIGHT = "Height"
C_WIDTH = "Width"
MS_EXACT_MATCH = "Text-Exact match"
MS_REGEXP = "Text-Regular expressions"
MS_ORDER = "Order"
FF_INDIVIDUAL_IMAGES = "individual images"
FF_STK_MOVIES = "stk movies"
FF_AVI_MOVIES = "avi,mov movies"
FF_AVI_MOVIES_OLD = ["avi movies"]
FF_OTHER_MOVIES = "tif,tiff,flex,zvi movies"
FF_OTHER_MOVIES_OLD = ["tif,tiff,flex movies", "tif,tiff,flex movies, zvi movies"]
IO_IMAGES = "Images"
IO_OBJECTS = "Objects"
IO_ALL = (IO_IMAGES, IO_OBJECTS)
IMAGE_FOR_OBJECTS_F = "IMAGE_FOR_%s"
SUPPORTED_IMAGE_EXTENSIONS = {
    ".arg",
    ".bmp",
    ".bufr",
    ".bw",
    ".cur",
    ".dcx",
    ".dib",
    ".emf",
    ".eps",
    ".fit",
    ".fits",
    ".flc",
    ".fli",
    ".fpx",
    ".gbr",
    ".gif",
    ".grib",
    ".h5",
    ".hdf",
    ".icns",
    ".ico",
    ".iim",
    ".im",
    ".jfif",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".mic",
    ".mpeg",
    ".mpg",
    ".msp",
    ".palm",
    ".pbm",
    ".pcd",
    ".pcx",
    ".pdf",
    ".pgm",
    ".png",
    ".ppm",
    ".ps",
    ".psd",
    ".ras",
    ".rgb",
    ".rgba",
    ".sgi",
    ".tga",
    ".tif",
    ".tiff",
    ".wmf",
    ".xbm",
    ".xpm",
    ".zarr",
}
SUPPORTED_MOVIE_EXTENSIONS = {
    ".avi",
    ".flex",
    ".mov",
    ".mpeg",
    ".stk",
    ".tif",
    ".tiff",
    ".zvi",
}

# See https://docs.openmicroscopy.org/bio-formats/latest/supported-formats.html
ALL_BIOFORMATS_EXTENSIONS = (
    ".1sc", ".2fl", ".acff", ".afi", ".afm", ".aim", ".al3d", ".ali",
    ".am", ".amiramesh", ".apl", ".arf", ".avi", ".bif", ".bin", ".bip",
    ".bmp", ".btf", ".c01", ".cfg", ".ch5", ".cif", ".cr2", ".crw",
    ".cxd", ".czi", ".dat", ".dcm", ".dib", ".dicom", ".dm2", ".dm3",
    ".dm4", ".dti", ".dv", ".eps", ".epsi", ".exp", ".fdf", ".fff",
    ".ffr", ".fits", ".flex", ".fli", ".frm", ".gel", ".gif", ".grey",
    ".h5", ".hdf", ".hdr", ".hed", ".his", ".htd", ".html", ".hx", ".i2i",
    ".ics", ".ids", ".im3", ".img", ".ims", ".inr", ".ipl", ".ipm", ".ipw",
    ".j2k", ".jp2", ".jpeg", ".jpf", ".jpg", ".jpk", ".jpx", ".klb",
    ".l2d", ".labels", ".lei", ".lif", ".liff", ".lim", ".lms", ".lsm",
    ".map", ".mdb", ".mea", ".mnc", ".mng", ".mod", ".mov", ".mrc", ".mrcs",
    ".mrw", ".msr", ".mtb", ".mvd2", ".naf", ".nd", ".nd2", ".ndpi", ".ndpis",
    ".nef", ".nhdr", ".nii", ".nii.gz", ".nrrd", ".obf", ".obsep", ".oib",
    ".oif", ".oir", ".ome", ".ome.btf", ".ome.tf2", ".ome.tf8", ".ome.tif",
    ".ome.tiff", ".ome.xml", ".par", ".pbm", ".pcoraw", ".pcx", ".pds",
    ".pgm", ".pic", ".pict", ".png", ".pnl", ".ppm", ".pr3", ".ps", ".psd",
    ".qptiff", ".r3d", ".raw", ".rcpnl", ".rec", ".res", ".scn", ".sdt",
    ".seq", ".sif", ".sld", ".sm2", ".sm3", ".spc", ".spe", ".spi", ".st",
    ".stk", ".stp", ".svs", ".sxm", ".tc.", ".tf2", ".tf8", ".tfr", ".tga",
    ".tif", ".tiff", ".tnb", ".top", ".txt", ".v", ".vff", ".vms", ".vsi",
    ".vws", ".wat", ".wlz", ".wpi", ".xdce", ".xml", ".xqd", ".xqf", ".xv",
    ".xys", ".zfp", ".zfr", ".zvi", ".ets"
)

DISALLOWED_BIOFORMATS_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".eps",
    ".epsi",
    ".htm",
    ".html",
    ".inf",
    ".log",
    ".ps",
    ".txt",
    ".wav",
    ".xml",
    ".zip"
}

BIOFORMATS_IMAGE_EXTENSIONS = set(ALL_BIOFORMATS_EXTENSIONS) - DISALLOWED_BIOFORMATS_EXTENSIONS

ALL_IMAGE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS.union(BIOFORMATS_IMAGE_EXTENSIONS)


FF = [FF_INDIVIDUAL_IMAGES, FF_STK_MOVIES, FF_AVI_MOVIES, FF_OTHER_MOVIES]
M_NONE = "None"
M_FILE_NAME = "File name"
M_PATH = "Path"
M_BOTH = "Both"
M_Z = "Z"
M_T = "T"
C_SERIES = "Series"
C_FRAME = "Frame"
P_IMAGES = "LoadImagesImageProvider"
V_IMAGES = 1
P_MOVIES = "LoadImagesMovieProvider"
V_MOVIES = 2
P_FLEX = "LoadImagesFlexFrameProvider"
V_FLEX = 1
I_INTERLEAVED = "Interleaved"
I_SEPARATED = "Separated"
SUB_NONE = "None"
SUB_ALL = "All"
SUB_SOME = "Some"
FILE_SCHEME = "file:"
PASSTHROUGH_SCHEMES = ["http", "https", "ftp", "s3", "gs"]

CT_GRAYSCALE = "Grayscale"
CT_COLOR = "Color"
CT_MASK = "Mask"
CT_OBJECTS = "Objects"
CT_FUNCTION = "Function"

MD_PLANAR = "Planar"
MD_SIZE_S = "SizeS"
MD_SIZE_C = "SizeC"
MD_SIZE_Z = "SizeZ"
MD_SIZE_T = "SizeT"
MD_SIZE_X = "SizeX"
MD_SIZE_Y = "SizeY"
MD_SERIES_NAME = "SeriesNames"
MD_SIZE_KEYS = {MD_SIZE_S, MD_SIZE_X, MD_SIZE_Y, MD_SIZE_Z, MD_SIZE_C, MD_SIZE_T}

NO_RESCALE = (None, None)
