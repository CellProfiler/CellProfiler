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
    ".ppm",
    ".grib",
    ".im",
    ".rgba",
    ".rgb",
    ".pcd",
    ".h5",
    ".jpe",
    ".jfif",
    ".jpg",
    ".fli",
    ".sgi",
    ".gbr",
    ".pcx",
    ".mpeg",
    ".jpeg",
    ".ps",
    ".flc",
    ".tif",
    ".hdf",
    ".icns",
    ".gif",
    ".palm",
    ".mpg",
    ".fits",
    ".pgm",
    ".mic",
    ".fit",
    ".xbm",
    ".eps",
    ".emf",
    ".dcx",
    ".bmp",
    ".bw",
    ".pbm",
    ".dib",
    ".ras",
    ".cur",
    ".fpx",
    ".png",
    ".msp",
    ".iim",
    ".wmf",
    ".tga",
    ".bufr",
    ".ico",
    ".psd",
    ".xpm",
    ".arg",
    ".pdf",
    ".tiff",
}
SUPPORTED_MOVIE_EXTENSIONS = {
    ".avi",
    ".mpeg",
    ".stk",
    ".flex",
    ".mov",
    ".tif",
    ".tiff",
    ".zvi",
}
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
PASSTHROUGH_SCHEMES = ("http", "https", "ftp", "omero", "s3")
