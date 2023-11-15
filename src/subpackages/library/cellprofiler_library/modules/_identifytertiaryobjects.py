import numpy
from cellprofiler_library.functions.object_processing import outline

def identifytertiaryobjects(
        primary_objects: numpy.ndarray = None,
        secondary_objects: numpy.ndarray = None,
        shrink_primary: bool = True,
        return_cp_output: bool = False
):
    if primary_objects.shape != secondary_objects.shape:
        raise ValueError(
            f"""
            This module requires that the object sets have matching widths
            and matching heights. The primary and secondary objects do not
            ({primary_objects.shape} vs {secondary_objects.shape}). If they
            are paired correctly you may want to use the ResizeObjects
            module to make them the same size.
            """
        )
    #
    # Find the outlines of the primary image and use this to shrink the
    # primary image by one. This guarantees that there is something left
    # of the secondary image after subtraction
    #
    primary_outline = outline(primary_objects)
    tertiary_objects = secondary_objects.copy()

    if shrink_primary:
        primary_mask = numpy.logical_or(primary_objects == 0, primary_outline)
    else:
        primary_mask = primary_objects == 0
    
    tertiary_objects[primary_mask == False] = 0

    if return_cp_output:
        tertiary_outlines = outline(tertiary_objects) != 0
        return (
            tertiary_objects,
            tertiary_outlines
        )
    return tertiary_objects
