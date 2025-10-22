from pydantic import validate_call, ConfigDict, Field
from typing import Annotated
from cellprofiler_library.types import ObjectSegmentation
from cellprofiler_library.functions.object_processing import find_centroids

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def shrink_to_object_centers(
    label_image: Annotated[ObjectSegmentation, Field(description="Input label image")]
    ) -> ObjectSegmentation:
    return find_centroids(label_image)