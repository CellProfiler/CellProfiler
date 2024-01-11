from enum import Enum


class ObjectSizeShapeFeatures(Enum):
    """The category of the per-object measurements made by the MeasureObjectSizeShape module"""

    AREA_SHAPE = "AreaShape"

    ZERNIKE_N = 9

    F_AREA = "Area"
    F_PERIMETER = "Perimeter"
    F_VOLUME = "Volume"
    F_SURFACE_AREA = "SurfaceArea"
    F_ECCENTRICITY = "Eccentricity"
    F_SOLIDITY = "Solidity"
    F_CONVEX_AREA = "ConvexArea"
    F_EXTENT = "Extent"
    F_CENTER_X = "Center_X"
    F_CENTER_Y = "Center_Y"
    F_CENTER_Z = "Center_Z"
    F_BBOX_AREA = "BoundingBoxArea"
    F_BBOX_VOLUME = "BoundingBoxVolume"
    F_MIN_X = "BoundingBoxMinimum_X"
    F_MAX_X = "BoundingBoxMaximum_X"
    F_MIN_Y = "BoundingBoxMinimum_Y"
    F_MAX_Y = "BoundingBoxMaximum_Y"
    F_MIN_Z = "BoundingBoxMinimum_Z"
    F_MAX_Z = "BoundingBoxMaximum_Z"
    F_EULER_NUMBER = "EulerNumber"
    F_FORM_FACTOR = "FormFactor"
    F_MAJOR_AXIS_LENGTH = "MajorAxisLength"
    F_MINOR_AXIS_LENGTH = "MinorAxisLength"
    F_ORIENTATION = "Orientation"
    F_COMPACTNESS = "Compactness"
    F_INERTIA = "InertiaTensor"
    F_MAXIMUM_RADIUS = "MaximumRadius"
    F_MEDIAN_RADIUS = "MedianRadius"
    F_MEAN_RADIUS = "MeanRadius"
    F_MIN_FERET_DIAMETER = "MinFeretDiameter"
    F_MAX_FERET_DIAMETER = "MaxFeretDiameter"

    F_CENTRAL_MOMENT_0_0 = "CentralMoment_0_0"
    F_CENTRAL_MOMENT_0_1 = "CentralMoment_0_1"
    F_CENTRAL_MOMENT_0_2 = "CentralMoment_0_2"
    F_CENTRAL_MOMENT_0_3 = "CentralMoment_0_3"
    F_CENTRAL_MOMENT_1_0 = "CentralMoment_1_0"
    F_CENTRAL_MOMENT_1_1 = "CentralMoment_1_1"
    F_CENTRAL_MOMENT_1_2 = "CentralMoment_1_2"
    F_CENTRAL_MOMENT_1_3 = "CentralMoment_1_3"
    F_CENTRAL_MOMENT_2_0 = "CentralMoment_2_0"
    F_CENTRAL_MOMENT_2_1 = "CentralMoment_2_1"
    F_CENTRAL_MOMENT_2_2 = "CentralMoment_2_2"
    F_CENTRAL_MOMENT_2_3 = "CentralMoment_2_3"
    F_EQUIVALENT_DIAMETER = "EquivalentDiameter"
    F_HU_MOMENT_0 = "HuMoment_0"
    F_HU_MOMENT_1 = "HuMoment_1"
    F_HU_MOMENT_2 = "HuMoment_2"
    F_HU_MOMENT_3 = "HuMoment_3"
    F_HU_MOMENT_4 = "HuMoment_4"
    F_HU_MOMENT_5 = "HuMoment_5"
    F_HU_MOMENT_6 = "HuMoment_6"
    F_INERTIA_TENSOR_0_0 = "InertiaTensor_0_0"
    F_INERTIA_TENSOR_0_1 = "InertiaTensor_0_1"
    F_INERTIA_TENSOR_1_0 = "InertiaTensor_1_0"
    F_INERTIA_TENSOR_1_1 = "InertiaTensor_1_1"
    F_INERTIA_TENSOR_EIGENVALUES_0 = "InertiaTensorEigenvalues_0"
    F_INERTIA_TENSOR_EIGENVALUES_1 = "InertiaTensorEigenvalues_1"
    F_NORMALIZED_MOMENT_0_0 = "NormalizedMoment_0_0"
    F_NORMALIZED_MOMENT_0_1 = "NormalizedMoment_0_1"
    F_NORMALIZED_MOMENT_0_2 = "NormalizedMoment_0_2"
    F_NORMALIZED_MOMENT_0_3 = "NormalizedMoment_0_3"
    F_NORMALIZED_MOMENT_1_0 = "NormalizedMoment_1_0"
    F_NORMALIZED_MOMENT_1_1 = "NormalizedMoment_1_1"
    F_NORMALIZED_MOMENT_1_2 = "NormalizedMoment_1_2"
    F_NORMALIZED_MOMENT_1_3 = "NormalizedMoment_1_3"
    F_NORMALIZED_MOMENT_2_0 = "NormalizedMoment_2_0"
    F_NORMALIZED_MOMENT_2_1 = "NormalizedMoment_2_1"
    F_NORMALIZED_MOMENT_2_2 = "NormalizedMoment_2_2"
    F_NORMALIZED_MOMENT_2_3 = "NormalizedMoment_2_3"
    F_NORMALIZED_MOMENT_3_0 = "NormalizedMoment_3_0"
    F_NORMALIZED_MOMENT_3_1 = "NormalizedMoment_3_1"
    F_NORMALIZED_MOMENT_3_2 = "NormalizedMoment_3_2"
    F_NORMALIZED_MOMENT_3_3 = "NormalizedMoment_3_3"
    F_SPATIAL_MOMENT_0_0 = "SpatialMoment_0_0"
    F_SPATIAL_MOMENT_0_1 = "SpatialMoment_0_1"
    F_SPATIAL_MOMENT_0_2 = "SpatialMoment_0_2"
    F_SPATIAL_MOMENT_0_3 = "SpatialMoment_0_3"
    F_SPATIAL_MOMENT_1_0 = "SpatialMoment_1_0"
    F_SPATIAL_MOMENT_1_1 = "SpatialMoment_1_1"
    F_SPATIAL_MOMENT_1_2 = "SpatialMoment_1_2"
    F_SPATIAL_MOMENT_1_3 = "SpatialMoment_1_3"
    F_SPATIAL_MOMENT_2_0 = "SpatialMoment_2_0"
    F_SPATIAL_MOMENT_2_1 = "SpatialMoment_2_1"
    F_SPATIAL_MOMENT_2_2 = "SpatialMoment_2_2"
    F_SPATIAL_MOMENT_2_3 = "SpatialMoment_2_3"

    """The non-Zernike features"""
    F_STD_2D = [
        F_AREA,
        F_PERIMETER,
        F_MAXIMUM_RADIUS,
        F_MEAN_RADIUS,
        F_MEDIAN_RADIUS,
        F_MIN_FERET_DIAMETER,
        F_MAX_FERET_DIAMETER,
        F_ORIENTATION,
        F_ECCENTRICITY,
        F_FORM_FACTOR,
        F_SOLIDITY,
        F_CONVEX_AREA,
        F_COMPACTNESS,
        F_BBOX_AREA,
    ]
    F_STD_3D = [
        F_VOLUME,
        F_SURFACE_AREA,
        F_CENTER_Z,
        F_BBOX_VOLUME,
        F_MIN_Z,
        F_MAX_Z,
    ]
    F_ADV_2D = [
        F_SPATIAL_MOMENT_0_0,
        F_SPATIAL_MOMENT_0_1,
        F_SPATIAL_MOMENT_0_2,
        F_SPATIAL_MOMENT_0_3,
        F_SPATIAL_MOMENT_1_0,
        F_SPATIAL_MOMENT_1_1,
        F_SPATIAL_MOMENT_1_2,
        F_SPATIAL_MOMENT_1_3,
        F_SPATIAL_MOMENT_2_0,
        F_SPATIAL_MOMENT_2_1,
        F_SPATIAL_MOMENT_2_2,
        F_SPATIAL_MOMENT_2_3,
        F_CENTRAL_MOMENT_0_0,
        F_CENTRAL_MOMENT_0_1,
        F_CENTRAL_MOMENT_0_2,
        F_CENTRAL_MOMENT_0_3,
        F_CENTRAL_MOMENT_1_0,
        F_CENTRAL_MOMENT_1_1,
        F_CENTRAL_MOMENT_1_2,
        F_CENTRAL_MOMENT_1_3,
        F_CENTRAL_MOMENT_2_0,
        F_CENTRAL_MOMENT_2_1,
        F_CENTRAL_MOMENT_2_2,
        F_CENTRAL_MOMENT_2_3,
        F_NORMALIZED_MOMENT_0_0,
        F_NORMALIZED_MOMENT_0_1,
        F_NORMALIZED_MOMENT_0_2,
        F_NORMALIZED_MOMENT_0_3,
        F_NORMALIZED_MOMENT_1_0,
        F_NORMALIZED_MOMENT_1_1,
        F_NORMALIZED_MOMENT_1_2,
        F_NORMALIZED_MOMENT_1_3,
        F_NORMALIZED_MOMENT_2_0,
        F_NORMALIZED_MOMENT_2_1,
        F_NORMALIZED_MOMENT_2_2,
        F_NORMALIZED_MOMENT_2_3,
        F_NORMALIZED_MOMENT_3_0,
        F_NORMALIZED_MOMENT_3_1,
        F_NORMALIZED_MOMENT_3_2,
        F_NORMALIZED_MOMENT_3_3,
        F_HU_MOMENT_0,
        F_HU_MOMENT_1,
        F_HU_MOMENT_2,
        F_HU_MOMENT_3,
        F_HU_MOMENT_4,
        F_HU_MOMENT_5,
        F_HU_MOMENT_6,
        F_INERTIA_TENSOR_0_0,
        F_INERTIA_TENSOR_0_1,
        F_INERTIA_TENSOR_1_0,
        F_INERTIA_TENSOR_1_1,
        F_INERTIA_TENSOR_EIGENVALUES_0,
        F_INERTIA_TENSOR_EIGENVALUES_1,
    ]
    F_ADV_3D = [F_SOLIDITY]
    F_STANDARD = [
        F_EXTENT,
        F_EULER_NUMBER,
        F_EQUIVALENT_DIAMETER,
        F_MAJOR_AXIS_LENGTH,
        F_MINOR_AXIS_LENGTH,
        F_CENTER_X,
        F_CENTER_Y,
        F_MIN_X,
        F_MIN_Y,
        F_MAX_X,
        F_MAX_Y,
    ]
