from cellprofiler_library.functions.segmentation import (
    DENSE_AXIS_NAMES,
    SPARSE_AXES_FIELDS,
    DENSE_AXIS,
    _validate_labels,
    _validate_dense,
    _validate_ijv,
    _validate_sparse,
)
import numpy
import skimage
import centrosome
import centrosome.zernike
import cellprofiler_library
import scipy


def get_object_type(objects):
    """Determine what format objects are in: dense, sparse, ijv, or label"""
    # Dense check
    if objects.ndim == len(DENSE_AXIS_NAMES):
        _validate_dense(objects)
        return "dense"
    # Sparse check
    elif objects.ndim == 1:
        _validate_sparse(objects)
        return "sparse"
    # Label check
    elif objects.ndim == 2 or objects.ndim == 3:
        _validate_labels(objects)
        return "labels"
    # elif


def measureobjectsizeshape(
    objects,
    calculate_advanced: bool = True,
    calculate_zernikes: bool = True,
    volumetric: bool = False,
):
    """
    Objects: dense, sparse, ijv, or label objects?
    For now, we will assume dense
    """

    _validate_dense(objects)

    # Define the feature names
    feature_names = list(F_STANDARD)
    if volumetric:
        feature_names += list(F_STD_3D)
        if calculate_advanced:
            feature_names += list(F_ADV_3D)
    else:
        feature_names += list(F_STD_2D)
        if calculate_zernikes:
            feature_names += [
                f"Zernike_{index[0]}_{index[1]}"
                for index in centrosome.zernike.get_zernike_indexes(ZERNIKE_N + 1)
            ]
        if calculate_advanced:
            feature_names += list(F_ADV_2D)

    # Dictionary to store data
    # data = dict

    if len(numpy.unique(objects)) == 0:
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

    # Non-overlapping objects in the dense format have objects.shape[0] == 1
    if objects.shape[0] == 1:
        # Non overlapping labels
        # Add some generalised processing function here
        labels = convert_dense_to_labels(objects, validate=True)
        features_to_record = analyze_labels(
            labels, desired_properties, calculate_zernikes, calculate_advanced
        )
    else:
        features_to_record = {}
        for lab_dim in range(objects.shape[0]):
            labels = convert_dense_to_labels(objects[[lab_dim]])
            buffer = analyze_labels(
                labels, desired_properties, calculate_zernikes, calculate_advanced
            )
            for f, m in buffer.items():
                if f in features_to_record:
                    features_to_record[f] = numpy.concatenate(
                        (features_to_record[f], m)
                    )
                else:
                    features_to_record[f] = m
                
    return features_to_record


def convert_dense_to_labels(dense, validate=True):
    """
    Convert **non-overlapping** dense array into labels. If the dense array
    is overlapping,
    """
    if validate:
        _validate_dense(dense)
        assert (
            dense.shape[DENSE_AXIS.c.value] == 1
            and dense.shape[DENSE_AXIS.t.value] == 1
        ), f"dense must have shape where f{DENSE_AXIS.c.name} = 1 and f{DENSE_AXIS.t.name} = 1"

    # Technically this is wrong. They can be overlapping
    # assert dense.shape[DENSE_AXIS.label_idx.value] == 1, "Labels are overlapping"

    # label_dim = numpy.unique(numpy.argwhere(dense > 0)[:, 0])

    # assert len(label_dim) == 1, "Overlapping detected"

    # Remove axes with len == 1
    # label = dense[label_dim].squeeze()
    
    label = dense.squeeze()

    return label


def analyze_labels(
    labels,
    desired_properties,
    calculate_zernikes: bool = True,
    calculate_advanced: bool = True,
):

    label_indices = numpy.unique(labels[labels != 0])
    nobjects = len(label_indices)

    if len(labels.shape) == 2:
        # 2D
        props = skimage.measure.regionprops_table(labels, properties=desired_properties)

        formfactor = 4.0 * numpy.pi * props["area"] / props["perimeter"] ** 2
        denom = [max(x, 1) for x in 4.0 * numpy.pi * props["area"]]
        compactness = props["perimeter"] ** 2 / denom

        max_radius = numpy.zeros(nobjects)
        median_radius = numpy.zeros(nobjects)
        mean_radius = numpy.zeros(nobjects)
        min_feret_diameter = numpy.zeros(nobjects)
        max_feret_diameter = numpy.zeros(nobjects)
        zernike_numbers = centrosome.zernike.get_zernike_indexes(ZERNIKE_N + 1)

        zf = {}
        for n, m in zernike_numbers:
            zf[(n, m)] = numpy.zeros(nobjects)

        for index, mini_image in enumerate(props["image"]):
            # Pad image to assist distance tranform
            mini_image = numpy.pad(mini_image, 1)
            distances = scipy.ndimage.distance_transform_edt(mini_image)
            max_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.maximum(distances, mini_image)
            )
            mean_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                scipy.ndimage.mean(distances, mini_image)
            )
            median_radius[index] = centrosome.cpmorphology.median_of_labels(
                distances, mini_image.astype("int"), [1]
            )

        #
        # Zernike features
        #
        if calculate_zernikes:
            zf_l = centrosome.zernike.zernike(zernike_numbers, labels, label_indices)
            for (n, m), z in zip(zernike_numbers, zf_l.transpose()):
                zf[(n, m)] = z

        if nobjects > 0:
            chulls, chull_counts = centrosome.cpmorphology.convex_hull(
                labels, label_indices
            )
            #
            # Feret diameter
            #
            (
                min_feret_diameter,
                max_feret_diameter,
            ) = centrosome.cpmorphology.feret_diameter(
                chulls, chull_counts, label_indices
            )

            features_to_record = {
                F_AREA: props["area"],
                F_PERIMETER: props["perimeter"],
                F_MAJOR_AXIS_LENGTH: props["major_axis_length"],
                F_MINOR_AXIS_LENGTH: props["minor_axis_length"],
                F_ECCENTRICITY: props["eccentricity"],
                F_ORIENTATION: props["orientation"] * (180 / numpy.pi),
                F_CENTER_X: props["centroid-1"],
                F_CENTER_Y: props["centroid-0"],
                F_BBOX_AREA: props["bbox_area"],
                F_MIN_X: props["bbox-1"],
                F_MAX_X: props["bbox-3"],
                F_MIN_Y: props["bbox-0"],
                F_MAX_Y: props["bbox-2"],
                F_FORM_FACTOR: formfactor,
                F_EXTENT: props["extent"],
                F_SOLIDITY: props["solidity"],
                F_COMPACTNESS: compactness,
                F_EULER_NUMBER: props["euler_number"],
                F_MAXIMUM_RADIUS: max_radius,
                F_MEAN_RADIUS: mean_radius,
                F_MEDIAN_RADIUS: median_radius,
                F_CONVEX_AREA: props["convex_area"],
                F_MIN_FERET_DIAMETER: min_feret_diameter,
                F_MAX_FERET_DIAMETER: max_feret_diameter,
                F_EQUIVALENT_DIAMETER: props["equivalent_diameter"],
            }
            if calculate_advanced:
                features_to_record.update(
                    {
                        F_SPATIAL_MOMENT_0_0: props["moments-0-0"],
                        F_SPATIAL_MOMENT_0_1: props["moments-0-1"],
                        F_SPATIAL_MOMENT_0_2: props["moments-0-2"],
                        F_SPATIAL_MOMENT_0_3: props["moments-0-3"],
                        F_SPATIAL_MOMENT_1_0: props["moments-1-0"],
                        F_SPATIAL_MOMENT_1_1: props["moments-1-1"],
                        F_SPATIAL_MOMENT_1_2: props["moments-1-2"],
                        F_SPATIAL_MOMENT_1_3: props["moments-1-3"],
                        F_SPATIAL_MOMENT_2_0: props["moments-2-0"],
                        F_SPATIAL_MOMENT_2_1: props["moments-2-1"],
                        F_SPATIAL_MOMENT_2_2: props["moments-2-2"],
                        F_SPATIAL_MOMENT_2_3: props["moments-2-3"],
                        F_CENTRAL_MOMENT_0_0: props["moments_central-0-0"],
                        F_CENTRAL_MOMENT_0_1: props["moments_central-0-1"],
                        F_CENTRAL_MOMENT_0_2: props["moments_central-0-2"],
                        F_CENTRAL_MOMENT_0_3: props["moments_central-0-3"],
                        F_CENTRAL_MOMENT_1_0: props["moments_central-1-0"],
                        F_CENTRAL_MOMENT_1_1: props["moments_central-1-1"],
                        F_CENTRAL_MOMENT_1_2: props["moments_central-1-2"],
                        F_CENTRAL_MOMENT_1_3: props["moments_central-1-3"],
                        F_CENTRAL_MOMENT_2_0: props["moments_central-2-0"],
                        F_CENTRAL_MOMENT_2_1: props["moments_central-2-1"],
                        F_CENTRAL_MOMENT_2_2: props["moments_central-2-2"],
                        F_CENTRAL_MOMENT_2_3: props["moments_central-2-3"],
                        F_NORMALIZED_MOMENT_0_0: props["moments_normalized-0-0"],
                        F_NORMALIZED_MOMENT_0_1: props["moments_normalized-0-1"],
                        F_NORMALIZED_MOMENT_0_2: props["moments_normalized-0-2"],
                        F_NORMALIZED_MOMENT_0_3: props["moments_normalized-0-3"],
                        F_NORMALIZED_MOMENT_1_0: props["moments_normalized-1-0"],
                        F_NORMALIZED_MOMENT_1_1: props["moments_normalized-1-1"],
                        F_NORMALIZED_MOMENT_1_2: props["moments_normalized-1-2"],
                        F_NORMALIZED_MOMENT_1_3: props["moments_normalized-1-3"],
                        F_NORMALIZED_MOMENT_2_0: props["moments_normalized-2-0"],
                        F_NORMALIZED_MOMENT_2_1: props["moments_normalized-2-1"],
                        F_NORMALIZED_MOMENT_2_2: props["moments_normalized-2-2"],
                        F_NORMALIZED_MOMENT_2_3: props["moments_normalized-2-3"],
                        F_NORMALIZED_MOMENT_3_0: props["moments_normalized-3-0"],
                        F_NORMALIZED_MOMENT_3_1: props["moments_normalized-3-1"],
                        F_NORMALIZED_MOMENT_3_2: props["moments_normalized-3-2"],
                        F_NORMALIZED_MOMENT_3_3: props["moments_normalized-3-3"],
                        F_HU_MOMENT_0: props["moments_hu-0"],
                        F_HU_MOMENT_1: props["moments_hu-1"],
                        F_HU_MOMENT_2: props["moments_hu-2"],
                        F_HU_MOMENT_3: props["moments_hu-3"],
                        F_HU_MOMENT_4: props["moments_hu-4"],
                        F_HU_MOMENT_5: props["moments_hu-5"],
                        F_HU_MOMENT_6: props["moments_hu-6"],
                        F_INERTIA_TENSOR_0_0: props["inertia_tensor-0-0"],
                        F_INERTIA_TENSOR_0_1: props["inertia_tensor-0-1"],
                        F_INERTIA_TENSOR_1_0: props["inertia_tensor-1-0"],
                        F_INERTIA_TENSOR_1_1: props["inertia_tensor-1-1"],
                        F_INERTIA_TENSOR_EIGENVALUES_0: props[
                            "inertia_tensor_eigvals-0"
                        ],
                        F_INERTIA_TENSOR_EIGENVALUES_1: props[
                            "inertia_tensor_eigvals-1"
                        ],
                    }
                )

            if calculate_zernikes:
                features_to_record.update(
                    {f"Zernike_{n}_{m}": zf[(n, m)] for n, m in zernike_numbers}
                )

            else:
                # 3D
                props = skimage.measure.regionprops_table(
                    labels, properties=desired_properties
                )
                # SurfaceArea
                surface_areas = numpy.zeros(len(props["label"]))
                for index, label in enumerate(props["label"]):
                    # this seems less elegant than you might wish, given that regionprops returns a slice,
                    # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
                    volume = labels[
                        max(props["bbox-0"][index] - 1, 0) : min(
                            props["bbox-3"][index] + 1, labels.shape[0]
                        ),
                        max(props["bbox-1"][index] - 1, 0) : min(
                            props["bbox-4"][index] + 1, labels.shape[1]
                        ),
                        max(props["bbox-2"][index] - 1, 0) : min(
                            props["bbox-5"][index] + 1, labels.shape[2]
                        ),
                    ]
                    volume = volume == label
                    verts, faces, _normals, _values = skimage.measure.marching_cubes(
                        volume,
                        method="lewiner",
                        spacing=objects.parent_image.spacing
                        if objects.has_parent_image
                        else (1.0,) * labels.ndim,
                        level=0,
                    )
                    surface_areas[index] = skimage.measure.mesh_surface_area(
                        verts, faces
                    )

                features_to_record = {
                    F_VOLUME: props["area"],
                    F_SURFACE_AREA: surface_areas,
                    F_MAJOR_AXIS_LENGTH: props["major_axis_length"],
                    F_MINOR_AXIS_LENGTH: props["minor_axis_length"],
                    F_CENTER_X: props["centroid-2"],
                    F_CENTER_Y: props["centroid-1"],
                    F_CENTER_Z: props["centroid-0"],
                    F_BBOX_VOLUME: props["bbox_area"],
                    F_MIN_X: props["bbox-2"],
                    F_MAX_X: props["bbox-5"],
                    F_MIN_Y: props["bbox-1"],
                    F_MAX_Y: props["bbox-4"],
                    F_MIN_Z: props["bbox-0"],
                    F_MAX_Z: props["bbox-3"],
                    F_EXTENT: props["extent"],
                    F_EULER_NUMBER: props["euler_number"],
                    F_EQUIVALENT_DIAMETER: props["equivalent_diameter"],
                }
                if calculate_advanced:
                    features_to_record[F_SOLIDITY] = props["solidity"]
            return features_to_record


"""The category of the per-object measurements made by this module"""
AREA_SHAPE = "AreaShape"

"""Calculate Zernike features for N,M where N=0 through ZERNIKE_N"""
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
