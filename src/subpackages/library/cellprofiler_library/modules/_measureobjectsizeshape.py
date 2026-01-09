from typing import Tuple, Optional, Annotated, Dict, Any
import numpy
import centrosome
import centrosome.zernike
from pydantic import validate_call, ConfigDict, Field
from cellprofiler_library.types import ObjectLabelsDense
from cellprofiler_library.functions.measurement import (
    measure_object_size_shape_2d,
    measure_object_size_shape_3d
)
from cellprofiler_library.opts.objectsizeshapefeatures import (
    F_STD_2D, 
    F_STD_3D, 
    F_ADV_2D, 
    F_ADV_3D, 
    F_STANDARD, 
    ZERNIKE_N,
    ObjectSizeShapeFeatures
)
from cellprofiler_library.functions.segmentation import (
    _validate_dense,
    convert_dense_to_label_set,
)

DEFAULT_INVALID_VALUE_DTYPE = {
    numpy.float64: numpy.nan,
    numpy.float32: numpy.nan,
    numpy.float16: numpy.nan,
    numpy.uint8: 0,
    numpy.uint16: 0,
    numpy.uint32: 0,
    numpy.uint64: 0,
    numpy.int8: 0,
    numpy.int16: 0,
    numpy.int32: 0,
    numpy.int64: 0,
    numpy.bool_: False,
    numpy.object_: None,
    numpy.str_: "",
}

def measureobjectsizeshape(
    objects:            Annotated[ObjectLabelsDense, Field(description="Object labels in the dense format")],
    calculate_advanced: Annotated[bool, Field(description="Calculate advanced features?")] = True,
    calculate_zernikes: Annotated[bool, Field(description="Calculate zernike features?")] = True,
    volumetric:         Annotated[bool, Field(description="Are the objects volumetric?")] = False,
    spacing:            Annotated[Optional[Tuple], Field(description="Object spacing")] = None,
) -> Dict[str, Optional[numpy.float_]]:
    """
    Objects: dense, sparse, ijv, or label objects?
    For now, we will assume dense
    """
    # _validate_dense(objects)

    # Define the feature names
    feature_names = [i.value for i in F_STANDARD]
    if volumetric:
        feature_names += [i.value for i in F_STD_3D]
        if calculate_advanced:
            feature_names += [i.value for i in F_ADV_3D]
    else:
        feature_names += [i.value for i in F_STD_2D]
        if calculate_zernikes:
            feature_names += [
                f"Zernike_{index[0]}_{index[1]}"
                for index in centrosome.zernike.get_zernike_indexes(
                    ZERNIKE_N + 1
                )
            ]
        if calculate_advanced:
            feature_names += [i.value for i in F_ADV_2D]

    if len(objects[objects != 0]) == 0:
        data = dict(zip(feature_names, [None] * len(feature_names)))
        for ft in feature_names:
            data[ft] = numpy.zeros((0,))
        return data

    if not volumetric:
        desired_properties = [
            "label",
            "image",
            "area",
            "perimeter",
            "bbox",
            "bbox_area",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "centroid",
            "equivalent_diameter",
            "extent",
            "eccentricity",
            "convex_area",
            "solidity",
            "euler_number",
        ]
        if calculate_advanced:
            desired_properties += [
                "inertia_tensor",
                "inertia_tensor_eigvals",
                "moments",
                "moments_central",
                "moments_hu",
                "moments_normalized",
            ]
    else:
        desired_properties = [
            "label",
            "image",
            "area",
            "centroid",
            "bbox",
            "bbox_area",
            "major_axis_length",
            "minor_axis_length",
            "extent",
            "equivalent_diameter",
            "euler_number",
        ]
        if calculate_advanced:
            desired_properties += [
                "solidity",
            ]

    labels = convert_dense_to_label_set(objects, validate=False)
    labels = [i[0] for i in labels]  # Just need the labelmaps, not indices

    if len(labels) > 1:
        # Overlapping labels
        features_to_record = {}
        for labelmap in labels:
            buffer, measured_labels, nobjects = _measure_single_labelset(
                labelmap,
                desired_properties,
                calculate_zernikes,
                calculate_advanced,
                volumetric,
                spacing
            )
            for f, m in buffer.items():
                if f in features_to_record:
                    features_to_record[f] = numpy.concatenate(
                        (features_to_record[f], m)
                    )
                else:
                    features_to_record[f] = m
    else:
        features_to_record, measured_labels, nobjects = _measure_single_labelset(
            labels[0],
            desired_properties,
            calculate_zernikes,
            calculate_advanced,
            volumetric,
            spacing
        )

    # ensure that all objects (objects.indices) are represented in the
    # output, even if they are not present in the label matrix. Fill with nan if missing
    if len(measured_labels) < nobjects:
        for i in objects.indices:
            if i not in measured_labels:
                for f in features_to_record:
                    features_to_record[f] = numpy.insert(
                        features_to_record[f], i-1, DEFAULT_INVALID_VALUE_DTYPE.get(
                            features_to_record[f].dtype.type, numpy.nan
                        )
                    )


    return features_to_record

def _measure_single_labelset(
    labels: numpy.ndarray,
    desired_properties: list,
    calculate_zernikes: bool,
    calculate_advanced: bool,
    volumetric: bool,
    spacing: Optional[Tuple]
) -> Tuple[Dict[str, Any], Any, int]:
    
    if spacing is None:
        spacing = (1.0,) * labels.ndim

    if not volumetric:
        props, (formfactor, compactness), (max_r, mean_r, median_r), zf, (min_feret, max_feret) = measure_object_size_shape_2d(
            labels, desired_properties, calculate_zernikes, spacing
        )
        nobjects = len(props["label"])
        
        features_to_record = {
            ObjectSizeShapeFeatures.F_AREA.value: props["area"],
            ObjectSizeShapeFeatures.F_PERIMETER.value: props["perimeter"],
            ObjectSizeShapeFeatures.F_MAJOR_AXIS_LENGTH.value: props["major_axis_length"],
            ObjectSizeShapeFeatures.F_MINOR_AXIS_LENGTH.value: props["minor_axis_length"],
            ObjectSizeShapeFeatures.F_ECCENTRICITY.value: props["eccentricity"],
            ObjectSizeShapeFeatures.F_ORIENTATION.value: props["orientation"] * (180 / numpy.pi),
            ObjectSizeShapeFeatures.F_CENTER_X.value: props["centroid-1"],
            ObjectSizeShapeFeatures.F_CENTER_Y.value: props["centroid-0"],
            ObjectSizeShapeFeatures.F_BBOX_AREA.value: props["bbox_area"],
            ObjectSizeShapeFeatures.F_MIN_X.value: props["bbox-1"],
            ObjectSizeShapeFeatures.F_MAX_X.value: props["bbox-3"],
            ObjectSizeShapeFeatures.F_MIN_Y.value: props["bbox-0"],
            ObjectSizeShapeFeatures.F_MAX_Y.value: props["bbox-2"],
            ObjectSizeShapeFeatures.F_FORM_FACTOR.value: formfactor,
            ObjectSizeShapeFeatures.F_EXTENT.value: props["extent"],
            ObjectSizeShapeFeatures.F_SOLIDITY.value: props["solidity"],
            ObjectSizeShapeFeatures.F_COMPACTNESS.value: compactness,
            ObjectSizeShapeFeatures.F_EULER_NUMBER.value: props["euler_number"],
            ObjectSizeShapeFeatures.F_MAXIMUM_RADIUS.value: max_r,
            ObjectSizeShapeFeatures.F_MEAN_RADIUS.value: mean_r,
            ObjectSizeShapeFeatures.F_MEDIAN_RADIUS.value: median_r,
            ObjectSizeShapeFeatures.F_CONVEX_AREA.value: props["convex_area"],
            ObjectSizeShapeFeatures.F_MIN_FERET_DIAMETER.value: min_feret,
            ObjectSizeShapeFeatures.F_MAX_FERET_DIAMETER.value: max_feret,
            ObjectSizeShapeFeatures.F_EQUIVALENT_DIAMETER.value: props["equivalent_diameter"],
        }
        
        if calculate_advanced:
            features_to_record.update(
                {
                    ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_0.value: props["moments-0-0"],
                    ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_1.value: props["moments-0-1"],
                    ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_2.value: props["moments-0-2"],
                    ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_0_3.value: props["moments-0-3"],
                    ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_0.value: props["moments-1-0"],
                    ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_1.value: props["moments-1-1"],
                    ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_2.value: props["moments-1-2"],
                    ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_1_3.value: props["moments-1-3"],
                    ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_0.value: props["moments-2-0"],
                    ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_1.value: props["moments-2-1"],
                    ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_2.value: props["moments-2-2"],
                    ObjectSizeShapeFeatures.F_SPATIAL_MOMENT_2_3.value: props["moments-2-3"],
                    ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_0.value: props["moments_central-0-0"],
                    ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_1.value: props["moments_central-0-1"],
                    ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_2.value: props["moments_central-0-2"],
                    ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_0_3.value: props["moments_central-0-3"],
                    ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_0.value: props["moments_central-1-0"],
                    ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_1.value: props["moments_central-1-1"],
                    ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_2.value: props["moments_central-1-2"],
                    ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_1_3.value: props["moments_central-1-3"],
                    ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_0.value: props["moments_central-2-0"],
                    ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_1.value: props["moments_central-2-1"],
                    ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_2.value: props["moments_central-2-2"],
                    ObjectSizeShapeFeatures.F_CENTRAL_MOMENT_2_3.value: props["moments_central-2-3"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_0.value: props["moments_normalized-0-0"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_1.value: props["moments_normalized-0-1"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_2.value: props["moments_normalized-0-2"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_0_3.value: props["moments_normalized-0-3"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_0.value: props["moments_normalized-1-0"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_1.value: props["moments_normalized-1-1"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_2.value: props["moments_normalized-1-2"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_1_3.value: props["moments_normalized-1-3"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_0.value: props["moments_normalized-2-0"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_1.value: props["moments_normalized-2-1"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_2.value: props["moments_normalized-2-2"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_2_3.value: props["moments_normalized-2-3"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_0.value: props["moments_normalized-3-0"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_1.value: props["moments_normalized-3-1"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_2.value: props["moments_normalized-3-2"],
                    ObjectSizeShapeFeatures.F_NORMALIZED_MOMENT_3_3.value: props["moments_normalized-3-3"],
                    ObjectSizeShapeFeatures.F_HU_MOMENT_0.value: props["moments_hu-0"],
                    ObjectSizeShapeFeatures.F_HU_MOMENT_1.value: props["moments_hu-1"],
                    ObjectSizeShapeFeatures.F_HU_MOMENT_2.value: props["moments_hu-2"],
                    ObjectSizeShapeFeatures.F_HU_MOMENT_3.value: props["moments_hu-3"],
                    ObjectSizeShapeFeatures.F_HU_MOMENT_4.value: props["moments_hu-4"],
                    ObjectSizeShapeFeatures.F_HU_MOMENT_5.value: props["moments_hu-5"],
                    ObjectSizeShapeFeatures.F_HU_MOMENT_6.value: props["moments_hu-6"],
                    ObjectSizeShapeFeatures.F_INERTIA_TENSOR_0_0.value: props["inertia_tensor-0-0"],
                    ObjectSizeShapeFeatures.F_INERTIA_TENSOR_0_1.value: props["inertia_tensor-0-1"],
                    ObjectSizeShapeFeatures.F_INERTIA_TENSOR_1_0.value: props["inertia_tensor-1-0"],
                    ObjectSizeShapeFeatures.F_INERTIA_TENSOR_1_1.value: props["inertia_tensor-1-1"],
                    ObjectSizeShapeFeatures.F_INERTIA_TENSOR_EIGENVALUES_0.value: props[
                        "inertia_tensor_eigvals-0"
                    ],
                    ObjectSizeShapeFeatures.F_INERTIA_TENSOR_EIGENVALUES_1.value: props[
                        "inertia_tensor_eigvals-1"
                    ],
                }
            )

        if calculate_zernikes:
            features_to_record.update(
                {f"Zernike_{n}_{m}": zf[(n, m)] for n, m in zf}
            )

    else:
        # 3D
        props, surface_areas = measure_object_size_shape_3d(
            labels, desired_properties, spacing
        )
        nobjects = len(props["label"])
        
        features_to_record = {
            ObjectSizeShapeFeatures.F_VOLUME.value: props["area"],
            ObjectSizeShapeFeatures.F_SURFACE_AREA.value: surface_areas,
            ObjectSizeShapeFeatures.F_MAJOR_AXIS_LENGTH.value: props["major_axis_length"],
            ObjectSizeShapeFeatures.F_MINOR_AXIS_LENGTH.value: props["minor_axis_length"],
            ObjectSizeShapeFeatures.F_CENTER_X.value: props["centroid-2"],
            ObjectSizeShapeFeatures.F_CENTER_Y.value: props["centroid-1"],
            ObjectSizeShapeFeatures.F_CENTER_Z.value: props["centroid-0"],
            ObjectSizeShapeFeatures.F_BBOX_VOLUME.value: props["bbox_area"],
            ObjectSizeShapeFeatures.F_MIN_X.value: props["bbox-2"],
            ObjectSizeShapeFeatures.F_MAX_X.value: props["bbox-5"],
            ObjectSizeShapeFeatures.F_MIN_Y.value: props["bbox-1"],
            ObjectSizeShapeFeatures.F_MAX_Y.value: props["bbox-4"],
            ObjectSizeShapeFeatures.F_MIN_Z.value: props["bbox-0"],
            ObjectSizeShapeFeatures.F_MAX_Z.value: props["bbox-3"],
            ObjectSizeShapeFeatures.F_EXTENT.value: props["extent"],
            ObjectSizeShapeFeatures.F_EULER_NUMBER.value: props["euler_number"],
            ObjectSizeShapeFeatures.F_EQUIVALENT_DIAMETER.value: props["equivalent_diameter"],
        }
        if calculate_advanced:
            features_to_record[ObjectSizeShapeFeatures.F_SOLIDITY.value] = props["solidity"]

    return features_to_record, props["label"], nobjects
