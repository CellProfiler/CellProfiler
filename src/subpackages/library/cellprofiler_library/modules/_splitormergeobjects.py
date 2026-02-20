import numpy as np
from pydantic import validate_call, Field, ConfigDict
from typing import Annotated, Optional
from cellprofiler_library.types import ObjectSegmentation, Image2DGrayscale
from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.opts.splitormergeobjects import RelabelOption, MergeOption, MergingMethod, C_PARENT, ObjectIntensityMethod
from cellprofiler_library.functions.object_processing import filter_using_image, split_objects, merge_unify_distance, merge_unify_parent
from cellprofiler_library.functions.measurement import get_object_count_measurements, get_object_location_measurements, get_relate_object_measurements
from cellprofiler_library.functions.segmentation import convert_labels_to_ijv

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def split_or_merge_objects(
        labels:                     Annotated[ObjectSegmentation, Field(description="The input object segmentation.")],
        relabel_option:             Annotated[RelabelOption, Field(description="Whether to split separate objects sharing a label or merge adjacent objects.")],
        objects_name:               Annotated[str, Field(description="The name of the input objects.")], 
        parent_name:                Annotated[Optional[str], Field(description="The name of the parent object used to guide merging (if using 'Unify Parent').")], 
        merge_option:               Annotated[Optional[MergeOption], Field(description="The method used to merge objects (Distance or Parent).")],
        merging_method:             Annotated[Optional[MergingMethod], Field(description="Whether to keep merged objects as disconnected pieces or create a convex hull.")], 
        distance_threshold:         Annotated[Optional[float], Field(description="The maximum distance (in pixels) within which to merge objects.")], 
        image:                      Annotated[Optional[Image2DGrayscale], Field(description="The grayscale image used to guide merging. You must also pass the minimum intensity fraction and where algorithm if using this option.")],
        relaitonship_measurement:   Annotated[Optional[LibraryMeasurements], Field(description="Measurements containing the parent-child relationships.")], 
        merge_condition:            Annotated[Optional[ObjectIntensityMethod], Field(description="The algorithm used to evaluate intensity between objects (Centroids or Closest Point).")],
        minimum_intensity_fraction: Annotated[Optional[float], Field(description="The minimum intensity fraction required to merge objects when using an image.")],
        output_objects_name:        Annotated[Optional[str], Field(description="The name of the output objects. Only used if returning measurements.")],
        output_object_volumetric:   Annotated[Optional[bool], Field(description="Whether the output objects are volumetric. Only used if returning measurements.")],
        labels_ijv:                 Annotated[Optional[ObjectSegmentation], Field(description="The ijv representation of the input objects. Only used if returning measurements.")],
        return_measurements:        Annotated[bool, Field(description="Whether to return the relabeled objects and their measurements.")] = False
    ):
    if relabel_option == RelabelOption.SPLIT:
        output_labels = split_objects(labels)
    else:
        if merge_option == MergeOption.UNIFY_DISTANCE:
            assert distance_threshold is not None
            if image is not None:
                assert merge_condition is not None, "Merge condition must be provided when merge_using_image is True"
                assert minimum_intensity_fraction is not None, "Minimum intensity fraction must be provided when merge_using_image is True"
            output_labels = merge_unify_distance(
                labels,
                distance_threshold,
                image,
                merge_condition,
                minimum_intensity_fraction
            )
        elif merge_option == MergeOption.UNIFY_PARENT:
            assert parent_name is not None, "Parent name must be provided when merge_option is Unify Parent"
            assert merging_method is not None, "Merging method must be provided when merge_option is Unify Parent"
            assert relaitonship_measurement is not None, "Relationship measurement must be provided when merge_option is Unify Parent"
            parents_of = relaitonship_measurement.get_measurement(
                objects_name, "_".join((C_PARENT, parent_name))
            )
            output_labels = merge_unify_parent(
                labels,
                parents_of,
                merging_method,
            )
        else: 
            raise NotImplementedError(f"Unimplemented merging method: {merging_method}")
    if return_measurements:
        assert labels_ijv is not None, "labels_ijv must be provided if returning measurements"
        assert output_object_volumetric is not None, "output_object_volumetric must be provided if returning measurements"
        assert output_objects_name is not None, "output_objects_name must be provided if returning measurements"
    
        output_object_ijv = convert_labels_to_ijv(output_labels)
        
        lib_measurements_object_count = get_object_count_measurements(output_objects_name, np.max(output_labels))
        lib_mesaurements_object_location = get_object_location_measurements(output_objects_name, output_labels)
        lib_measurements_relate = get_relate_object_measurements(
        output_labels, output_object_volumetric, output_objects_name, output_object_ijv,
        objects_name, labels, labels_ijv
        )
        final_lib_measurements = lib_measurements_object_count.merge(lib_mesaurements_object_location).merge(lib_measurements_relate)
        return output_labels, final_lib_measurements
    return output_labels

