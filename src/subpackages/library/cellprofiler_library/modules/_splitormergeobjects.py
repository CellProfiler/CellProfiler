from pydantic import validate_call, Field, ConfigDict
from typing import Annotated, Optional
from cellprofiler_library.types import ObjectSegmentation, Image2DGrayscale
from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.opts.splitormergeobjects import RelabelOption, MergeOption, MergingMethod, C_PARENT, ObjectIntensityMethod
from cellprofiler_library.functions.object_processing import filter_using_image, split_objects, merge_unify_distance, merge_unify_parent

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def split_or_merge_objects(
        labels:             Annotated[ObjectSegmentation, Field(description="The input object label matrix to be split or merged.")],
        relabel_option:     Annotated[RelabelOption, Field(description="Whether to split separate objects sharing a label or merge adjacent objects.")],
        objects_name:       Annotated[str, Field(description="The name of the input objects.")], 
        parent_name:        Annotated[str, Field(description="The name of the parent object used to guide merging (if using 'Unify Parent').")], 
        merge_option:       Annotated[Optional[MergeOption], Field(description="The method used to merge objects (Distance or Parent).")],
        merging_method:     Annotated[Optional[MergingMethod], Field(description="Whether to keep merged objects as disconnected pieces or create a convex hull.")], 
        distance_threshold: Annotated[Optional[float], Field(description="The maximum distance (in pixels) within which to merge objects.")], 
        merge_using_image:  Annotated[bool, Field(description="Whether to use a grayscale image's intensity to guide merging.")], 
        image:              Annotated[Optional[Image2DGrayscale], Field(description="The grayscale image used to guide merging.")], 
        relaitonship_measurement: Annotated[LibraryMeasurements, Field(description="Measurements containing the parent-child relationships.")], 
        where_algorithm:    Annotated[ObjectIntensityMethod, Field(description="The algorithm used to evaluate intensity between objects (Centroids or Closest Point).")],
        minimum_intensity_fraction: Annotated[float, Field(description="The minimum intensity fraction required to merge objects when using an image.")]
    ):
    if relabel_option == RelabelOption.SPLIT:
        output_labels = split_objects(labels)
    else:
        if merge_option == MergeOption.UNIFY_DISTANCE:
            assert distance_threshold is not None
            if merge_using_image:
                assert image is not None, "Image must be provided when merge_using_image is True"
                assert where_algorithm is not None, "Where algorithm must be provided when merge_using_image is True"
                assert minimum_intensity_fraction is not None, "Minimum intensity fraction must be provided when merge_using_image is True"
            output_labels = merge_unify_distance(
                labels,
                distance_threshold,
                merge_using_image,
                image,
                where_algorithm,
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
    return output_labels

