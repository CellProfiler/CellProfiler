import numpy
from numpy.typing import NDArray
from typing import Optional, Tuple, Annotated
from pydantic import Field, validate_call, ConfigDict

from cellprofiler_library.types import ObjectSegmentation
from cellprofiler_library.opts.measureobjectneighbors import DistanceMethod as NeighborsDistanceMethod
from cellprofiler_library.functions.measurement import measure_object_neighbors as _measure_object_neighbors

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_object_neighbors(
        labels:                 Annotated[ObjectSegmentation, Field(description="Input labeled objects. Remember to include any small objects or objects that are on the edge")],
        kept_labels:            Annotated[ObjectSegmentation, Field(description="Input labels of interest. Can ignore small objects or objects that are on the edge and need to be ignored in the final output")],
        neighbor_labels:        Annotated[ObjectSegmentation, Field(description="Input labels for neighboring objects. Can ignore small objects or objects that are on the edge and need to be ignored in the final output")],
        neighbor_kept_labels:   Annotated[ObjectSegmentation, Field(description="Input labels for neighboring objects of interest. Can ignore small objects or objects that are on the edge and need to be ignored in the final output")],
        neighbors_are_objects:  Annotated[bool, Field(description="Set to True if the neighbors are taken from the same object set as the input objects")],
        dimensions:             Annotated[int, Field(description="Use 2 for 2D and 3 for 3D")],
        distance_value:         Annotated[int, Field(description="Neighbor distance")],
        distance_method:        Annotated[NeighborsDistanceMethod, Field(description="Method to determine neighbors")], 
        wants_excluded_objects: Annotated[bool, Field(description="Consider objects discarded for touching image border?")]=True,
    ) -> Tuple[
        NDArray[numpy.float_],
        NDArray[numpy.int_],
        NDArray[numpy.int_],
        NDArray[numpy.float_],
        NDArray[numpy.float_],
        NDArray[numpy.float_],
        NDArray[numpy.float_],
        NDArray[numpy.int_],
        NDArray[numpy.int_],
        Optional[NDArray[numpy.int_]],
    ]:
    return _measure_object_neighbors(
        labels, 
        kept_labels,
        neighbor_labels, 
        neighbor_kept_labels,
        neighbors_are_objects,
        dimensions, 
        distance_value,
        distance_method, 
        wants_excluded_objects,
    )