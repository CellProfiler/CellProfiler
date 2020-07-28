import functools

import numpy

from cellprofiler_core.constants.measurement import (
    C_PATH_NAME,
    C_FILE_NAME,
    C_URL,
    C_OBJECTS_FILE_NAME,
    C_OBJECTS_PATH_NAME,
    C_OBJECTS_URL,
    COLTYPE_VARCHAR,
    COLTYPE_FLOAT,
    COLTYPE_INTEGER,
)


def header_to_column(field):
    """Convert the field name in the header to a column name

    This function converts Image_FileName to FileName and
    Image_PathName to PathName so that the output column names
    in the database will be Image_FileName and Image_PathName
    """
    for name in (
        C_PATH_NAME,
        C_FILE_NAME,
        C_URL,
        C_OBJECTS_FILE_NAME,
        C_OBJECTS_PATH_NAME,
        C_OBJECTS_URL,
    ):
        if field.startswith("Image" + "_" + name + "_"):
            return field[6:]
    return field


def is_path_name_feature(feature):
    """Return true if the feature name is a path name"""
    return feature.startswith(C_PATH_NAME + "_")


def is_file_name_feature(feature):
    """Return true if the feature name is a file name"""
    return feature.startswith(C_FILE_NAME + "_")


def is_url_name_feature(feature):
    return feature.startswith(C_URL + "_")


def is_objects_path_name_feature(feature):
    """Return true if the feature name is the path to a labels file"""
    return feature.startswith(C_OBJECTS_PATH_NAME + "_")


def is_objects_file_name_feature(feature):
    """Return true if the feature name is a labels file name"""
    return feature.startswith(C_OBJECTS_FILE_NAME + "_")


def is_objects_url_name_feature(feature):
    return feature.startswith(C_OBJECTS_URL + "_")


def get_image_name(feature):
    """Extract the image name from a feature name"""
    if is_path_name_feature(feature):
        return feature[len(C_PATH_NAME + "_") :]
    if is_file_name_feature(feature):
        return feature[len(C_FILE_NAME + "_") :]
    if is_url_name_feature(feature):
        return feature[len(C_URL + "_") :]
    raise ValueError('"%s" is not a path feature or file name feature' % feature)


def get_objects_name(feature):
    """Extract the objects name from a feature name"""
    if is_objects_path_name_feature(feature):
        return feature[len(C_OBJECTS_PATH_NAME + "_") :]
    if is_objects_file_name_feature(feature):
        return feature[len(C_OBJECTS_FILE_NAME + "_") :]
    if is_objects_url_name_feature(feature):
        return feature[len(C_OBJECTS_URL + "_") :]
    raise ValueError(
        '"%s" is not a objects path feature or file name feature' % feature
    )


def make_path_name_feature(image):
    """Return the path name feature, given an image name

    The path name feature is the name of the measurement that stores
    the image's path name.
    """
    return C_PATH_NAME + "_" + image


def make_file_name_feature(image):
    """Return the file name feature, given an image name

    The file name feature is the name of the measurement that stores
    the image's file name.
    """
    return C_FILE_NAME + "_" + image


def make_objects_path_name_feature(objects_name):
    """Return the path name feature, given an object name

    The path name feature is the name of the measurement that stores
    the objects file path name.
    """
    return C_OBJECTS_PATH_NAME + "_" + objects_name


def make_objects_file_name_feature(objects_name):
    """Return the file name feature, given an object name

    The file name feature is the name of the measurement that stores
    the objects file name.
    """
    return C_OBJECTS_FILE_NAME + "_" + objects_name


def best_cast(sequence, coltype=None):
    """Return the best cast (integer, float or string) of the sequence

    sequence - a sequence of strings

    Try casting all elements to integer and float, returning a numpy
    array of values. If all fail, return a numpy array of strings.
    """
    if isinstance(coltype, str) and coltype.startswith(COLTYPE_VARCHAR):
        # Cast columns already defined as strings as same
        return numpy.array(sequence)

    def fn(x, y):
        if COLTYPE_VARCHAR in (x, y):
            return COLTYPE_VARCHAR
        if COLTYPE_FLOAT in (x, y):
            return COLTYPE_FLOAT
        return COLTYPE_INTEGER

    ldtype = functools.reduce(
        fn, [get_loaddata_type(x) for x in sequence], COLTYPE_INTEGER,
    )
    if ldtype == COLTYPE_VARCHAR:
        return numpy.array(sequence)
    elif ldtype == COLTYPE_FLOAT:
        return numpy.array(sequence, numpy.float64)
    else:
        return numpy.array(sequence, numpy.int32)


def get_loaddata_type(x):
    """Return the type to use to represent x

    If x is a 32-bit integer, return cpmeas.COLTYPE_INTEGER.
    If x cannot be represented in 32 bits but is an integer,
    return cpmeas.COLTYPE_VARCHAR
    If x can be represented as a float, return COLTYPE_FLOAT
    """

    try:
        iv = int(x)
        if iv > numpy.iinfo(numpy.int32).max:
            return COLTYPE_VARCHAR
        if iv < numpy.iinfo(numpy.int32).min:
            return COLTYPE_VARCHAR
        return COLTYPE_INTEGER
    except:
        try:
            fv = float(x)
            return COLTYPE_FLOAT
        except:
            return COLTYPE_VARCHAR


def bad_sizes_warning(first_size, first_filename, second_size, second_filename):
    """Return a warning message about sizes being wrong

    first_size: tuple of height / width of first image
    first_filename: file name of first image
    second_size: tuple of height / width of second image
    second_filename: file name of second image
    """
    warning = (
        "Warning: loading image files of different dimensions.\n\n"
        "%s: width = %d, height = %d\n"
        "%s: width = %d, height = %d"
    ) % (
        first_filename,
        first_size[1],
        first_size[0],
        second_filename,
        second_size[1],
        second_size[0],
    )
    return warning
