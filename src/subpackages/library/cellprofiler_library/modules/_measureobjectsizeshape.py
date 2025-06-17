from typing import Tuple
import numpy
import skimage
import scipy

import centrosome
import centrosome.zernike

from cellprofiler_library.functions.measurement import measure_object_size_shape
from cellprofiler_library.opts.objectsizeshapefeatures import ObjectSizeShapeFeatures
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
    objects,
    calculate_advanced: bool = True,
    calculate_zernikes: bool = True,
    volumetric: bool = False,
    spacing: Tuple = None,
):
    """
    Objects: dense, sparse, ijv, or label objects?
    For now, we will assume dense
    """
    # _validate_dense(objects)

    # Define the feature names
    feature_names = list(ObjectSizeShapeFeatures.F_STANDARD.value)
    if volumetric:
        feature_names += list(ObjectSizeShapeFeatures.F_STD_3D.value)
        if calculate_advanced:
            feature_names += list(ObjectSizeShapeFeatures.F_ADV_3D.value)
    else:
        feature_names += list(ObjectSizeShapeFeatures.F_STD_2D.value)
        if calculate_zernikes:
            feature_names += [
                f"Zernike_{index[0]}_{index[1]}"
                for index in centrosome.zernike.get_zernike_indexes(
                    ObjectSizeShapeFeatures.ZERNIKE_N.value + 1
                )
            ]
        if calculate_advanced:
            feature_names += list(ObjectSizeShapeFeatures.F_ADV_2D.value)

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
            buffer, measured_labels, nobjects = measure_object_size_shape(
                labels=labelmap,
                desired_properties=desired_properties,
                calculate_zernikes=calculate_zernikes,
                calculate_advanced=calculate_advanced,
                spacing=spacing,
            )
            for f, m in buffer.items():
                if f in features_to_record:
                    features_to_record[f] = numpy.concatenate(
                        (features_to_record[f], m)
                    )
                else:
                    features_to_record[f] = m
    else:
        features_to_record, measured_labels, nobjects = measure_object_size_shape(
            labels=labels[0],
            desired_properties=desired_properties,
            calculate_zernikes=calculate_zernikes,
            calculate_advanced=calculate_advanced,
            spacing=spacing,
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
