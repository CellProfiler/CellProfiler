import numpy
from typing import Annotated, Union, Tuple
from pydantic import validate_call, ConfigDict, Field
from cellprofiler_library.functions.object_processing import outline
from cellprofiler_library.types import ObjectSegmentation

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def identifytertiaryobjects(
        primary_labels:     Annotated[ObjectSegmentation, Field(description="Primary object segmentations")] = None,
        secondary_labels:   Annotated[ObjectSegmentation, Field(description="Secondary object segmentations")] = None,
        shrink_primary:     Annotated[bool, Field(description="Shrink the primary objects")] = True,
        return_cp_output:   Annotated[bool, Field(description="Return CellProfiler output")] = False
) -> Union[
    ObjectSegmentation,
    Tuple[
        ObjectSegmentation,
        ObjectSegmentation
    ]
]:
    # If size/shape differences were too extreme, raise an error.
    if primary_labels.shape != secondary_labels.shape:
        raise ValueError(
            f"""
            This module requires that the object sets have matching widths
            and matching heights. The primary and secondary objects do not
            ({primary_labels.shape} vs {secondary_labels.shape}). If they
            are paired correctly you may want to use the ResizeObjects
            module to make them the same size.
            """
        )
    #
    # Find the outlines of the primary image and use this to shrink the
    # primary image by one. This guarantees that there is something left
    # of the secondary image after subtraction
    #
    primary_outline = outline(primary_labels)
    tertiary_labels = secondary_labels.copy()
    if shrink_primary:
        primary_mask = numpy.logical_or(primary_labels == 0, primary_outline)
    else:
        primary_mask = primary_labels == 0
    tertiary_labels[primary_mask == False] = 0
     #
    # Check if a label was deleted as a result of the subtraction
    #
    secondary_unique_labels, secondary_unique_indices = numpy.unique(secondary_labels, return_index=True)
    tertiary_unique_labels = numpy.unique(tertiary_labels)
    missing_labels = numpy.setdiff1d(secondary_unique_labels, tertiary_unique_labels)
    for missing_label in missing_labels:
        # If a label was deleted, manually add a pixel to the tertiary_labels.
        # This workaround ensures that ghost objects do not get created by identifytertiaryobjects.
        
        # first non-zero (top-left) coodrinate of the secondary object is used to add a pixel to the tertiary_labels
        first_row, first_col = numpy.unravel_index(secondary_unique_indices[missing_label], secondary_labels.shape)
        tertiary_labels[first_row, first_col] = missing_label
    
    if return_cp_output:
        #
        # Get the outlines of the tertiary image
        #
        tertiary_outlines = outline(tertiary_labels) != 0
        return (
            tertiary_labels,
            tertiary_outlines
        )
    return tertiary_labels