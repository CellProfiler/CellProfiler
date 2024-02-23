import io

import centrosome.propagate
import numpy
import scipy.stats

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.module
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT


import cellprofiler.modules.measureobjectintensitydistribution
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.setting
import cellprofiler_core.workspace
import tests.frontend.modules

OBJECT_NAME = "objectname"
CENTER_NAME = "centername"
IMAGE_NAME = "imagename"
HEAT_MAP_NAME = "heatmapname"


def feature_frac_at_d(bin, bin_count, image_name=IMAGE_NAME):
    if bin == bin_count + 1:
        return "_".join(
            [
                cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
                cellprofiler.modules.measureobjectintensitydistribution.F_FRAC_AT_D,
                image_name,
                cellprofiler.modules.measureobjectintensitydistribution.FF_OVERFLOW,
            ]
        )

    return (
        cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY
        + "_"
        + cellprofiler.modules.measureobjectintensitydistribution.FF_FRAC_AT_D
        % (image_name, bin, bin_count)
    )


def feature_mean_frac(bin, bin_count, image_name=IMAGE_NAME):
    if bin == bin_count + 1:
        return "_".join(
            [
                cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
                cellprofiler.modules.measureobjectintensitydistribution.F_MEAN_FRAC,
                image_name,
                cellprofiler.modules.measureobjectintensitydistribution.FF_OVERFLOW,
            ]
        )

    return (
        cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY
        + "_"
        + cellprofiler.modules.measureobjectintensitydistribution.FF_MEAN_FRAC
        % (image_name, bin, bin_count)
    )


def feature_radial_cv(bin, bin_count, image_name=IMAGE_NAME):
    if bin == bin_count + 1:
        return "_".join(
            [
                cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
                cellprofiler.modules.measureobjectintensitydistribution.F_RADIAL_CV,
                image_name,
                cellprofiler.modules.measureobjectintensitydistribution.FF_OVERFLOW,
            ]
        )

    return (
        cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY
        + "_"
        + cellprofiler.modules.measureobjectintensitydistribution.FF_RADIAL_CV
        % (image_name, bin, bin_count)
    )


def test_please_implement_a_test_of_the_new_version():
    assert (
        cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution.variable_revision_number
        == 6
    )


def test_load_v2():
    data = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
DateRevision:20120126174947

MeasureObjectIntensityDistribution:[module_num:8|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
Hidden:2
Hidden:2
Hidden:2
Select an image to measure:EnhancedGreen
Select an image to measure:OrigBlue
Select objects to measure:Nuclei
Object to use as center?:These objects
Select objects to use as centers:Cells
Select objects to measure:Nuclei
Object to use as center?:Other objects
Select objects to use as centers:Cells
Scale bins?:No
Number of bins:4
Maximum radius:200
Scale bins?:Yes
Number of bins:5
Maximum radius:50
"""
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(
        module,
        cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution,
    )
    assert module.object_count.value == 2
    assert module.bin_counts_count.value == 2
    assert {"OrigBlue", "EnhancedGreen"}.issubset(module.images_list.value)
    assert module.objects[0].object_name == "Nuclei"
    assert (
        module.objects[0].center_choice
        == cellprofiler.modules.measureobjectintensitydistribution.C_SELF
    )
    assert module.objects[0].center_object_name == "Cells"
    assert (
        module.objects[1].center_choice
        == cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER
    )
    assert module.objects[1].center_object_name == "Cells"
    assert module.bin_counts[0].bin_count == 4
    assert not module.bin_counts[0].wants_scaled
    assert module.bin_counts[0].maximum_radius == 200
    assert module.bin_counts[1].bin_count == 5
    assert module.bin_counts[1].wants_scaled
    assert module.bin_counts[1].maximum_radius == 50


def test_load_v3():
    file = tests.frontend.modules.get_test_resources_directory(
        "measureobjectintensitydistribution/v3.pipeline"
    )
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(
        module,
        cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution,
    )
    assert module.object_count.value == 3
    assert module.bin_counts_count.value == 2
    assert {"OrigBlue", "EnhancedGreen"}.issubset(module.images_list.value)
    assert module.objects[0].object_name == "Nuclei"
    assert (
        module.objects[0].center_choice
        == cellprofiler.modules.measureobjectintensitydistribution.C_SELF
    )
    assert module.objects[0].center_object_name == "Cells"
    assert (
        module.objects[1].center_choice
        == cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER
    )
    assert module.objects[1].center_object_name == "Cells"
    assert (
        module.objects[2].center_choice
        == cellprofiler.modules.measureobjectintensitydistribution.C_EDGES_OF_OTHER
    )
    assert module.objects[2].center_object_name == "Cells"
    assert module.bin_counts[0].bin_count == 4
    assert not module.bin_counts[0].wants_scaled
    assert module.bin_counts[0].maximum_radius == 200
    assert module.bin_counts[1].bin_count == 5
    assert module.bin_counts[1].wants_scaled
    assert module.bin_counts[1].maximum_radius == 50
    assert len(module.heatmaps) == 0


def test_load_v4():
    file = tests.frontend.modules.get_test_resources_directory(
        "measureobjectintensitydistribution/v4.pipeline"
    )
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(
        module,
        cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution,
    )
    assert (
        module.wants_zernikes
        == cellprofiler.modules.measureobjectintensitydistribution.Z_NONE
    )
    assert module.zernike_degree == 9
    assert len(module.images_list.value) == 2
    assert {"CropGreen", "CropRed"}.issubset(module.images_list.value)
    assert len(module.objects) == 2
    for group, (object_name, center_choice, center_object_name) in zip(
        module.objects,
        (
            (
                "Nuclei",
                cellprofiler.modules.measureobjectintensitydistribution.C_SELF,
                "Ichthyosaurs",
            ),
            (
                "Cells",
                cellprofiler.modules.measureobjectintensitydistribution.C_EDGES_OF_OTHER,
                "Nuclei",
            ),
        ),
    ):
        assert group.object_name.value == object_name
        assert group.center_choice.value == center_choice
        assert group.center_object_name == center_object_name
    assert len(module.bin_counts) == 2
    for group, (bin_count, scale, max_radius) in zip(
        module.bin_counts, ((5, True, 100), (4, False, 100))
    ):
        assert group.wants_scaled == scale
        assert group.bin_count == bin_count
        assert group.maximum_radius == max_radius
    for (
        group,
        (
            image_name,
            object_name,
            bin_count,
            measurement,
            colormap,
            wants_to_save,
            output_image_name,
        ),
    ) in zip(
        module.heatmaps,
        (
            (
                "CropRed",
                "Cells",
                5,
                cellprofiler.modules.measureobjectintensitydistribution.A_FRAC_AT_D,
                "Default",
                True,
                "Heat",
            ),
            (
                "CropGreen",
                "Nuclei",
                4,
                cellprofiler.modules.measureobjectintensitydistribution.A_MEAN_FRAC,
                "Spectral",
                False,
                "A",
            ),
            (
                "CropRed",
                "Nuclei",
                5,
                cellprofiler.modules.measureobjectintensitydistribution.A_RADIAL_CV,
                "Default",
                False,
                "B",
            ),
        ),
    ):
        assert group.image_name.value == image_name
        assert group.object_name.value == object_name
        assert int(group.bin_count.value) == bin_count
        assert group.measurement == measurement
        assert group.colormap == colormap
        assert group.wants_to_save_display == wants_to_save
        assert group.display_name == output_image_name


def test_load_v5():
    file = tests.frontend.modules.get_test_resources_directory(
        "measureobjectintensitydistribution/v5.pipeline"
    )
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 2
    module = pipeline.modules()[0]
    assert isinstance(
        module,
        cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution,
    )
    assert (
        module.wants_zernikes
        == cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES
    )
    assert module.zernike_degree == 7
    assert len(module.images_list.value) == 2
    assert {"CropGreen", "CropRed"}.issubset(module.images_list.value)
    assert len(module.objects) == 2
    for group, (object_name, center_choice, center_object_name) in zip(
        module.objects,
        (
            (
                "Nuclei",
                cellprofiler.modules.measureobjectintensitydistribution.C_SELF,
                "Ichthyosaurs",
            ),
            (
                "Cells",
                cellprofiler.modules.measureobjectintensitydistribution.C_EDGES_OF_OTHER,
                "Nuclei",
            ),
        ),
    ):
        assert group.object_name.value == object_name
        assert group.center_choice.value == center_choice
        assert group.center_object_name == center_object_name
    assert len(module.bin_counts) == 2
    for group, (bin_count, scale, max_radius) in zip(
        module.bin_counts, ((5, True, 100), (4, False, 100))
    ):
        assert group.wants_scaled == scale
        assert group.bin_count == bin_count
        assert group.maximum_radius == max_radius
    for (
        group,
        (
            image_name,
            object_name,
            bin_count,
            measurement,
            colormap,
            wants_to_save,
            output_image_name,
        ),
    ) in zip(
        module.heatmaps,
        (
            (
                "CropRed",
                "Cells",
                5,
                cellprofiler.modules.measureobjectintensitydistribution.A_FRAC_AT_D,
                "Default",
                True,
                "Heat",
            ),
            (
                "CropGreen",
                "Nuclei",
                4,
                cellprofiler.modules.measureobjectintensitydistribution.A_MEAN_FRAC,
                "Spectral",
                False,
                "A",
            ),
            (
                "CropRed",
                "Nuclei",
                5,
                cellprofiler.modules.measureobjectintensitydistribution.A_RADIAL_CV,
                "Default",
                False,
                "B",
            ),
        ),
    ):
        assert group.image_name.value == image_name
        assert group.object_name.value == object_name
        assert int(group.bin_count.value) == bin_count
        assert group.measurement == measurement
        assert group.colormap == colormap
        assert group.wants_to_save_display == wants_to_save
        assert group.display_name == output_image_name

    module = pipeline.modules()[1]
    assert (
        module.wants_zernikes
        == cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES_AND_PHASE
    )


def test_01_get_measurement_columns():
    module = (
        cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution()
    )
    module.images_list.value = "DNA, Cytoplasm, Actin"
    for i, object_name, center_name in (
        (0, "Nucleii", None),
        (1, "Cells", "Nucleii"),
        (2, "Cytoplasm", "Nucleii"),
    ):
        if i:
            module.add_object()
        module.objects[i].object_name.value = object_name
        if center_name is None:
            module.objects[
                i
            ].center_choice.value = (
                cellprofiler.modules.measureobjectintensitydistribution.C_SELF
            )
        else:
            module.objects[
                i
            ].center_choice.value = (
                cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER
            )
            module.objects[i].center_object_name.value = center_name
    for i, bin_count in enumerate((4, 5, 6)):
        if i:
            module.add_bin_count()
        module.bin_counts[i].bin_count.value = bin_count
    module.bin_counts[2].wants_scaled.value = False

    columns = module.get_measurement_columns(None)
    column_dictionary = {}
    for object_name, feature, coltype in columns:
        key = (object_name, feature)
        assert not (key in column_dictionary)
        assert coltype == COLTYPE_FLOAT
        column_dictionary[key] = (object_name, feature, coltype)

    for object_name in [x.object_name.value for x in module.objects]:
        for image_name in module.images_list.value:
            for bin_count, wants_scaled in [
                (x.bin_count.value, x.wants_scaled.value) for x in module.bin_counts
            ]:
                for bin in range(1, bin_count + (1 if wants_scaled else 2)):
                    for feature_fn in (
                        feature_frac_at_d,
                        feature_mean_frac,
                        feature_radial_cv,
                    ):
                        measurement = feature_fn(bin, bin_count, image_name)
                        key = (object_name, measurement)
                        assert key in column_dictionary
                        del column_dictionary[key]
    assert len(column_dictionary) == 0


def test_02_get_zernike_columns():
    module = (
        cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution()
    )
    for wants_zernikes, ftrs in (
        (
            cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES,
            (
                cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_MAGNITUDE,
            ),
        ),
        (
            cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES_AND_PHASE,
            (
                cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_MAGNITUDE,
                cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_PHASE,
            ),
        ),
    ):
        module.wants_zernikes.value = wants_zernikes
        module.zernike_degree.value = 2
        module.images_list.value = "DNA, Cytoplasm, Actin"
        for i, object_name, center_name in (
            (0, "Nucleii", None),
            (1, "Cells", "Nucleii"),
            (2, "Cytoplasm", "Nucleii"),
        ):
            if i:
                module.add_object()
            module.objects[i].object_name.value = object_name
        columns = module.get_measurement_columns(None)
        for image_name in "DNA", "Cytoplasm", "Actin":
            for object_name in "Nucleii", "Cells", "Cytoplasm":
                for n, m in ((0, 0), (1, 1), (2, 0), (2, 2)):
                    for ftr in ftrs:
                        name = "_".join(
                            (
                                cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
                                ftr,
                                image_name,
                                str(n),
                                str(m),
                            )
                        )
                        col = (
                            object_name,
                            name,
                            COLTYPE_FLOAT,
                        )
                        assert col in columns


def test_01_get_measurements():
    module = (
        cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution()
    )
    module.images_list.value = "DNA, Cytoplasm, Actin"
    for i, object_name, center_name in (
        (0, "Nucleii", None),
        (1, "Cells", "Nucleii"),
        (2, "Cytoplasm", "Nucleii"),
    ):
        if i:
            module.add_object()
        module.objects[i].object_name.value = object_name
        if center_name is None:
            module.objects[
                i
            ].center_choice.value = (
                cellprofiler.modules.measureobjectintensitydistribution.C_SELF
            )
        else:
            module.objects[
                i
            ].center_choice.value = (
                cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER
            )
            module.objects[i].center_object_name.value = center_name
    for i, bin_count in ((0, 4), (0, 5), (0, 6)):
        if i:
            module.add_bin_count()
        module.bin_counts[i].bin_count.value = bin_count

    for object_name in [x.object_name.value for x in module.objects]:
        assert tuple(module.get_categories(None, object_name)) == (
            cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
        )
        for feature in cellprofiler.modules.measureobjectintensitydistribution.F_ALL:
            assert feature in module.get_measurements(
                None,
                object_name,
                cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
            )
        for image_name in module.images_list.value:
            for (
                feature
            ) in cellprofiler.modules.measureobjectintensitydistribution.F_ALL:
                assert image_name in module.get_measurement_images(
                    None,
                    object_name,
                    cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
                    feature,
                )
            for bin_count in [x.bin_count.value for x in module.bin_counts]:
                for bin in range(1, bin_count + 1):
                    for (
                        feature
                    ) in cellprofiler.modules.measureobjectintensitydistribution.F_ALL:
                        assert "%dof%d" % (
                            bin,
                            bin_count,
                        ) in module.get_measurement_scales(
                            None,
                            object_name,
                            cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
                            feature,
                            image_name,
                        )


def test_02_get_zernike_measurements():
    module = (
        cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution()
    )
    for wants_zernikes, ftrs in (
        (
            cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES,
            (
                cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_MAGNITUDE,
            ),
        ),
        (
            cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES_AND_PHASE,
            (
                cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_MAGNITUDE,
                cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_PHASE,
            ),
        ),
    ):
        module.wants_zernikes.value = wants_zernikes
        module.zernike_degree.value = 2

        module.images_list.value = "DNA, Cytoplasm, Actin"
        for i, object_name, center_name in (
            (0, "Nucleii", None),
            (1, "Cells", "Nucleii"),
            (2, "Cytoplasm", "Nucleii"),
        ):
            if i:
                module.add_object()
            module.objects[i].object_name.value = object_name
            if center_name is None:
                module.objects[
                    i
                ].center_choice.value = (
                    cellprofiler.modules.measureobjectintensitydistribution.C_SELF
                )
            else:
                module.objects[
                    i
                ].center_choice.value = (
                    cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER
                )
                module.objects[i].center_object_name.value = center_name

        for object_name in "Nucleii", "Cells", "Cytoplasm":
            result = module.get_measurements(
                None,
                object_name,
                cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
            )
            for ftr in ftrs:
                assert ftr in result
                iresult = module.get_measurement_images(
                    None,
                    object_name,
                    cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
                    ftr,
                )
                for image in "DNA", "Cytoplasm", "Actin":
                    assert image in iresult
                    sresult = module.get_measurement_scales(
                        None,
                        object_name,
                        cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
                        ftr,
                        image,
                    )
                    for n, m in ((0, 0), (1, 1), (2, 0), (2, 2)):
                        assert "%d_%d" % (n, m) in sresult


def test_default_heatmap_values():
    module = (
        cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution()
    )
    module.add_heatmap()
    module.heatmaps[0].image_name.value = IMAGE_NAME
    module.heatmaps[0].object_name.value = OBJECT_NAME
    module.heatmaps[0].bin_count.value = 10
    module.images_list.value = "Bar"
    module.objects[0].object_name.value = "Foo"
    module.bin_counts[0].bin_count.value = 2
    assert module.heatmaps[0].image_name.get_image_name() == "Bar"
    assert not module.heatmaps[0].image_name.is_visible()
    assert module.heatmaps[0].object_name.get_objects_name() == "Foo"
    assert not module.heatmaps[0].object_name.is_visible()
    assert module.heatmaps[0].get_number_of_bins() == 2
    module.images_list.value = "Bar, MoreBar"
    assert module.heatmaps[0].image_name.is_visible()
    assert module.heatmaps[0].image_name.get_image_name() == IMAGE_NAME
    module.add_object()
    assert module.heatmaps[0].object_name.is_visible()
    assert module.heatmaps[0].object_name.get_objects_name() == OBJECT_NAME
    module.add_bin_count()
    assert module.heatmaps[0].get_number_of_bins() == 10


def run_module(
    image,
    labels,
    center_labels=None,
    center_choice=cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER,
    bin_count=4,
    maximum_radius=100,
    wants_scaled=True,
    wants_workspace=False,
    wants_zernikes=cellprofiler.modules.measureobjectintensitydistribution.Z_NONE,
    zernike_degree=2,
):
    """Run the module, returning the measurements

    image - matrix representing the image to be analyzed
    labels - labels matrix of objects to be analyzed
    center_labels - labels matrix of alternate centers or None for self
                    centers
    bin_count - # of radial bins
    """
    module = (
        cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution()
    )
    module.wants_zernikes.value = wants_zernikes
    module.zernike_degree.value = zernike_degree
    module.images_list.value = IMAGE_NAME
    module.objects[0].object_name.value = OBJECT_NAME
    object_set = cellprofiler_core.object.ObjectSet()
    main_objects = cellprofiler_core.object.Objects()
    main_objects.segmented = labels
    object_set.add_objects(main_objects, OBJECT_NAME)
    if center_labels is None:
        module.objects[
            0
        ].center_choice.value = (
            cellprofiler.modules.measureobjectintensitydistribution.C_SELF
        )
    else:
        module.objects[0].center_choice.value = center_choice
        module.objects[0].center_object_name.value = CENTER_NAME
        center_objects = cellprofiler_core.object.Objects()
        center_objects.segmented = center_labels
        object_set.add_objects(center_objects, CENTER_NAME)
    module.bin_counts[0].bin_count.value = bin_count
    module.bin_counts[0].wants_scaled.value = wants_scaled
    module.bin_counts[0].maximum_radius.value = maximum_radius
    module.add_heatmap()
    module.add_heatmap()
    module.add_heatmap()
    for i, (a, f) in enumerate(
        (
            (
                cellprofiler.modules.measureobjectintensitydistribution.A_FRAC_AT_D,
                cellprofiler.modules.measureobjectintensitydistribution.F_FRAC_AT_D,
            ),
            (
                cellprofiler.modules.measureobjectintensitydistribution.A_MEAN_FRAC,
                cellprofiler.modules.measureobjectintensitydistribution.F_MEAN_FRAC,
            ),
            (
                cellprofiler.modules.measureobjectintensitydistribution.A_RADIAL_CV,
                cellprofiler.modules.measureobjectintensitydistribution.F_RADIAL_CV,
            ),
        )
    ):
        module.heatmaps[i].image_name.value = IMAGE_NAME
        module.heatmaps[i].object_name.value = OBJECT_NAME
        module.heatmaps[i].bin_count.value = str(bin_count)
        module.heatmaps[i].wants_to_save_display.value = True
        display_name = HEAT_MAP_NAME + f
        module.heatmaps[i].display_name.value = display_name
        module.heatmaps[i].colormap.value = "gray"
        module.heatmaps[i].measurement.value = a
    pipeline = cellprofiler_core.pipeline.Pipeline()
    measurements = cellprofiler_core.measurement.Measurements()
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = measurements
    img = cellprofiler_core.image.Image(image)
    image_set.add(IMAGE_NAME, img)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, module, image_set, object_set, measurements, image_set_list
    )
    module.run(workspace)
    if wants_workspace:
        return measurements, workspace
    return measurements


def test_zeros_self():
    """Test the module on an empty labels matrix, self-labeled"""
    m = run_module(
        numpy.zeros((10, 10)),
        numpy.zeros((10, 10), int),
        wants_zernikes=cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES_AND_PHASE,
        zernike_degree=2,
    )
    for bin in range(1, 5):
        for feature in (
            feature_frac_at_d(bin, 4),
            feature_mean_frac(bin, 4),
            feature_radial_cv(bin, 4),
        ):
            assert feature in m.get_feature_names(OBJECT_NAME)
            data = m.get_current_measurement(OBJECT_NAME, feature)
            assert len(data) == 0
    for ftr in (
        cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_MAGNITUDE,
        cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_PHASE,
    ):
        for n_, m_ in ((0, 0), (1, 1), (2, 0), (2, 2)):
            feature = "_".join(
                (
                    cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
                    ftr,
                    IMAGE_NAME,
                    str(n_),
                    str(m_),
                )
            )
            assert feature in m.get_feature_names(OBJECT_NAME)
            assert len(m[OBJECT_NAME, feature]) == 0


def test_circle():
    """Test the module on a uniform circle"""
    i, j = numpy.mgrid[-50:51, -50:51]
    labels = (numpy.sqrt(i * i + j * j) <= 40).astype(int)
    m, workspace = run_module(
        numpy.ones(labels.shape),
        labels,
        wants_workspace=True,
        wants_zernikes=True,
        zernike_degree=2,
    )
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    bins = labels * (1 + (numpy.sqrt(i * i + j * j) / 10).astype(int))
    for bin in range(1, 5):
        data = m.get_current_measurement(OBJECT_NAME, feature_frac_at_d(bin, 4))
        assert len(data) == 1
        area = (float(bin) * 2.0 - 1.0) / 16.0
        assert data[0] > area - 0.1
        assert data[0] < area + 0.1
        heatmap = workspace.image_set.get_image(
            HEAT_MAP_NAME
            + cellprofiler.modules.measureobjectintensitydistribution.F_FRAC_AT_D
        ).pixel_data
        data = data.astype(heatmap.dtype)
        assert scipy.stats.mode(heatmap[bins == bin])[0][0] == data[0]
        data = m.get_current_measurement(OBJECT_NAME, feature_mean_frac(bin, 4))
        assert len(data) == 1
        assert round(abs(data[0] - 1), 2) == 0
        heatmap = workspace.image_set.get_image(
            HEAT_MAP_NAME
            + cellprofiler.modules.measureobjectintensitydistribution.F_MEAN_FRAC
        ).pixel_data
        data = data.astype(heatmap.dtype)
        assert scipy.stats.mode(heatmap[bins == bin])[0][0] == data[0]
        data = m.get_current_measurement(OBJECT_NAME, feature_radial_cv(bin, 4))
        assert len(data) == 1
        assert round(abs(data[0] - 0), 2) == 0
        heatmap = workspace.image_set.get_image(
            HEAT_MAP_NAME
            + cellprofiler.modules.measureobjectintensitydistribution.F_RADIAL_CV
        ).pixel_data
        data = data.astype(heatmap.dtype)
        assert scipy.stats.mode(heatmap[bins == bin])[0][0] == data[0]
    module = workspace.module
    assert isinstance(
        module,
        cellprofiler.modules.measureobjectintensitydistribution.MeasureObjectIntensityDistribution,
    )
    data = m[OBJECT_NAME, module.get_zernike_magnitude_name(IMAGE_NAME, 0, 0)]
    assert len(data) == 1
    assert abs(data[0] - 1) < 0.001
    for n_, m_ in ((1, 1), (2, 0), (2, 2)):
        data = m[OBJECT_NAME, module.get_zernike_magnitude_name(IMAGE_NAME, n_, m_)]
        assert abs(data[0] - 0) < 0.001


def test_01_half_circle():
    """Test the module on a circle and an image that's 1/2 zeros

    The measurements here are somewhat considerably off because
    the propagate function uses a Manhattan distance with jaywalking
    allowed instead of the Euclidean distance.
    """
    i, j = numpy.mgrid[-50:51, -50:51]
    labels = (numpy.sqrt(i * i + j * j) <= 40).astype(int)
    image = numpy.zeros(labels.shape)
    image[i > 0] = (numpy.sqrt(i * i + j * j) / 100)[i > 0]
    image[j == 0] = 0
    image[i == j] = 0
    image[i == -j] = 0
    # 1/2 of the octants should be pretty much all zero and 1/2
    # should be all one
    x = [0, 0, 0, 0, 1, 1, 1, 1]
    expected_cv = numpy.std(x) / numpy.mean(x)
    m = run_module(image, labels)
    bin_labels = (numpy.sqrt(i * i + j * j) * 4 / 40.001).astype(int)
    mask = i * i + j * j <= 40 * 40
    total_intensity = numpy.sum(image[mask])
    for bin in range(1, 5):
        data = m.get_current_measurement(OBJECT_NAME, feature_frac_at_d(bin, 4))
        assert len(data) == 1
        bin_count = numpy.sum(bin_labels[mask] == bin - 1)
        frac_in_bin = float(bin_count) / numpy.sum(mask)
        bin_intensity = numpy.sum(image[mask & (bin_labels == bin - 1)])
        expected = bin_intensity / total_intensity
        assert numpy.abs(expected - data[0]) < 0.2 * expected
        data = m.get_current_measurement(OBJECT_NAME, feature_mean_frac(bin, 4))
        assert len(data) == 1
        expected = expected / frac_in_bin
        assert numpy.abs(data[0] - expected) < 0.2 * expected
        data = m.get_current_measurement(OBJECT_NAME, feature_radial_cv(bin, 4))
        assert len(data) == 1
        assert numpy.abs(data[0] - expected_cv) < 0.2 * expected_cv


def test_02_half_circle_zernike():
    i, j = numpy.mgrid[-50:50, -50:50]
    ii, jj = [_.astype(float) + 0.5 for _ in (i, j)]
    labels = (numpy.sqrt(ii * ii + jj * jj) <= 40).astype(int)
    image = numpy.zeros(labels.shape)
    image[ii > 0] = 1
    m = run_module(
        image,
        labels,
        wants_zernikes=cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES_AND_PHASE,
        zernike_degree=2,
    )
    for n_, m_, expected, delta in (
        (0, 0, 0.5, 0.001),
        (1, 1, 0.225, 0.1),
        (2, 0, 0, 0.01),
        (2, 2, 0, 0.01),
    ):
        ftr = "_".join(
            (
                cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
                cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_MAGNITUDE,
                IMAGE_NAME,
                str(n_),
                str(m_),
            )
        )
        assert abs(m[OBJECT_NAME, ftr][0] - expected) < delta
    ftr = "_".join(
        (
            cellprofiler.modules.measureobjectintensitydistribution.M_CATEGORY,
            cellprofiler.modules.measureobjectintensitydistribution.FF_ZERNIKE_PHASE,
            IMAGE_NAME,
            "1",
            "1",
        )
    )
    phase_i_1_1 = m[OBJECT_NAME, ftr][0]
    image = numpy.zeros(labels.shape)
    image[jj > 0] = 1
    m = run_module(
        image,
        labels,
        wants_zernikes=cellprofiler.modules.measureobjectintensitydistribution.Z_MAGNITUDES_AND_PHASE,
        zernike_degree=1,
    )
    phase_j_1_1 = m[OBJECT_NAME, ftr][0]
    assert numpy.abs(numpy.abs(phase_i_1_1 - phase_j_1_1) - numpy.pi / 2) == 0


def test_line():
    """Test the alternate centers with a line"""
    labels = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    centers = numpy.zeros(labels.shape, int)
    centers[2, 1] = 1
    distance_to_center = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 0],
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 0],
            [0, 1, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    distance_to_edge = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 2, 2, 2, 2, 2, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    numpy.random.seed(0)
    image = numpy.random.uniform(size=labels.shape)
    m = run_module(image, labels, centers)
    total_intensity = numpy.sum(image[labels == 1])
    normalized_distance = distance_to_center / (
        distance_to_center + distance_to_edge + 0.001
    )
    bin_labels = (normalized_distance * 4).astype(int)
    for bin in range(1, 5):
        data = m.get_current_measurement(OBJECT_NAME, feature_frac_at_d(bin, 4))
        assert len(data) == 1
        bin_intensity = numpy.sum(image[(labels == 1) & (bin_labels == bin - 1)])
        expected = bin_intensity / total_intensity
        assert numpy.abs(expected - data[0]) < 0.1 * expected
        data = m.get_current_measurement(OBJECT_NAME, feature_mean_frac(bin, 4))
        expected = (
            expected
            * numpy.sum(labels == 1)
            / numpy.sum((labels == 1) & (bin_labels == bin - 1))
        )
        assert numpy.abs(data[0] - expected) < 0.1 * expected
        data = m.get_current_measurement(OBJECT_NAME, feature_radial_cv(bin, 4))
        assert len(data) == 1


def test_no_scaling():
    i, j = numpy.mgrid[-40:40, -40:40]
    #
    # I'll try to calculate the distance the same way as propagate
    # jaywalk min(i,j) times and go straight abs(i - j) times
    #
    jaywalks = numpy.minimum(numpy.abs(i), numpy.abs(j))
    straights = numpy.abs(numpy.abs(i) - numpy.abs(j))
    distance = jaywalks * numpy.sqrt(2) + straights
    labels = (distance <= 35).astype(int)
    r = numpy.random.RandomState()
    r.seed(35)
    image = r.uniform(size=i.shape)
    total_intensity = numpy.sum(image[labels == 1])
    bin_labels = (distance / 5).astype(int)
    bin_labels[bin_labels > 4] = 4
    m = run_module(image, labels, bin_count=4, maximum_radius=20, wants_scaled=False)
    for bin in range(1, 6):
        data = m.get_current_measurement(OBJECT_NAME, feature_frac_at_d(bin, 4))
        assert len(data) == 1
        bin_intensity = numpy.sum(image[(labels == 1) & (bin_labels == bin - 1)])
        expected = bin_intensity / total_intensity
        assert round(abs(expected - data[0]), 4) == 0
        data = m.get_current_measurement(OBJECT_NAME, feature_mean_frac(bin, 4))
        expected = (
            expected
            * numpy.sum(labels == 1)
            / numpy.sum((labels == 1) & (bin_labels == bin - 1))
        )
        assert round(abs(data[0] - expected), 4) == 0
        data = m.get_current_measurement(OBJECT_NAME, feature_radial_cv(bin, 4))
        assert len(data) == 1


def test_edges_of_objects():
    r = numpy.random.RandomState()
    r.seed(36)
    i, j = numpy.mgrid[-20:21, -20:21]
    labels = ((i > -19) & (i < 19) & (j > -19) & (j < 19)).astype(int)
    centers = numpy.zeros(labels.shape, int)
    centers[(i > -5) * (i < 5) & (j > -5) & (j < 5)] = 1
    image = r.uniform(size=labels.shape)
    m = run_module(
        image,
        labels,
        center_labels=centers,
        center_choice=cellprofiler.modules.measureobjectintensitydistribution.C_EDGES_OF_OTHER,
        bin_count=4,
        maximum_radius=8,
        wants_scaled=False,
    )

    _, d_from_center = centrosome.propagate.propagate(
        numpy.zeros(labels.shape), centers, (labels > 0), 1
    )
    good_mask = (labels > 0) & (centers == 0)
    d_from_center = d_from_center[good_mask]
    bins = (d_from_center / 2).astype(int)
    bins[bins > 4] = 4
    bin_counts = numpy.bincount(bins)
    image_sums = numpy.bincount(bins, image[good_mask])
    frac_at_d = image_sums / numpy.sum(image_sums)
    for i in range(1, 6):
        data = m.get_current_measurement(OBJECT_NAME, feature_frac_at_d(i, 4))
        assert len(data) == 1
        assert round(abs(data[0] - frac_at_d[i - 1]), 7) == 0


def test_two_circles():
    i, j = numpy.mgrid[-50:51, -50:51]
    i, j = [numpy.hstack((x, x)) for x in (i, j)]
    d = numpy.sqrt(i * i + j * j)
    labels = (d <= 40).astype(int)
    labels[:, (j.shape[1] // 2) :] *= 2
    img = numpy.zeros(labels.shape)
    img[labels == 1] = 1
    img[labels == 2] = d[labels == 2] / 40
    m, workspace = run_module(img, labels, wants_workspace=True)
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    bins = (labels != 0) * (1 + (numpy.sqrt(i * i + j * j) / 10).astype(int))
    for bin in range(1, 5):
        data = m.get_current_measurement(OBJECT_NAME, feature_frac_at_d(bin, 4))
        assert len(data) == 2
        area = (float(bin) * 2.0 - 1.0) / 16.0
        bin_d = (float(bin) - 0.5) * 8 / 21
        assert numpy.abs(data[0] - area) < 0.1
        assert numpy.abs(data[1] - area * bin_d) < 0.1
        heatmap = workspace.image_set.get_image(
            HEAT_MAP_NAME
            + cellprofiler.modules.measureobjectintensitydistribution.F_FRAC_AT_D
        ).pixel_data
        data = data.astype(heatmap.dtype)
        for label in 1, 2:
            mask = (bins == bin) & (labels == label)
            assert scipy.stats.mode(heatmap[mask])[0][0] == data[label - 1]
        data = m.get_current_measurement(OBJECT_NAME, feature_mean_frac(bin, 4))
        assert len(data) == 2
        assert round(abs(data[0] - 1), 2) == 0
        heatmap = workspace.image_set.get_image(
            HEAT_MAP_NAME
            + cellprofiler.modules.measureobjectintensitydistribution.F_MEAN_FRAC
        ).pixel_data
        data = data.astype(heatmap.dtype)
        for label in 1, 2:
            mask = (bins == bin) & (labels == label)
            assert scipy.stats.mode(heatmap[mask])[0][0] == data[label - 1]
        data = m.get_current_measurement(OBJECT_NAME, feature_radial_cv(bin, 4))
        assert len(data) == 2
        assert round(abs(data[0] - 0), 2) == 0
        heatmap = workspace.image_set.get_image(
            HEAT_MAP_NAME
            + cellprofiler.modules.measureobjectintensitydistribution.F_RADIAL_CV
        ).pixel_data
        data = data.astype(heatmap.dtype)
        for label in 1, 2:
            mask = (bins == bin) & (labels == label)
            assert scipy.stats.mode(heatmap[mask])[0][0] == data[label - 1]


def test_img_607():
    """Regression test of bug IMG-607

    MeasureObjectIntensityDistribution fails if there are no pixels for
    some of the objects.
    """
    numpy.random.seed(41)
    labels = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 3, 3, 3, 0],
            [0, 1, 1, 1, 0, 0, 3, 3, 3, 0],
            [0, 1, 1, 1, 0, 0, 3, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    image = numpy.random.uniform(size=labels.shape)
    for center_labels in (labels, None):
        m = run_module(image, labels, center_labels=center_labels, bin_count=4)
        for bin in range(1, 5):
            data = m.get_current_measurement(OBJECT_NAME, feature_frac_at_d(bin, 4))
            assert len(data) == 3
            assert numpy.isnan(data[1])


def test_center_outside_of_object():
    """Make sure MeasureObjectIntensityDistribution can handle oddly shaped objects"""
    numpy.random.seed(42)
    labels = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    center_labels = numpy.zeros(labels.shape, int)
    center_labels[int(center_labels.shape[0] / 2), int(center_labels.shape[1] / 2)] = 1

    image = numpy.random.uniform(size=labels.shape)
    for center_choice in (
        cellprofiler.modules.measureobjectintensitydistribution.C_CENTERS_OF_OTHER,
        cellprofiler.modules.measureobjectintensitydistribution.C_EDGES_OF_OTHER,
    ):
        m = run_module(
            image,
            labels,
            center_labels=center_labels,
            center_choice=center_choice,
            bin_count=4,
        )
        for bin in range(1, 5):
            data = m.get_current_measurement(OBJECT_NAME, feature_frac_at_d(bin, 4))
            assert len(data) == 1

    m = run_module(image, labels, bin_count=4)
    for bin in range(1, 5):
        data = m.get_current_measurement(OBJECT_NAME, feature_frac_at_d(bin, 4))
        assert len(data) == 1


def test_wrong_size():
    """Regression test for IMG-961: objects & image of different sizes

    Make sure that the module executes without exception with and
    without centers and with similarly and differently shaped centers
    """
    numpy.random.seed(43)
    labels = numpy.ones((30, 40), int)
    image = numpy.random.uniform(size=(20, 50))
    m = run_module(image, labels)
    centers = numpy.zeros(labels.shape)
    centers[15, 20] = 1
    m = run_module(image, labels, centers)
    centers = numpy.zeros((35, 35), int)
    centers[15, 20] = 1
    m = run_module(image, labels, centers)


def test_more_labels_than_centers():
    """Regression test of img-1463"""
    numpy.random.seed(51)
    i, j = numpy.mgrid[0:100, 0:100]
    ir = (i % 10) - 5
    jr = (j % 10) - 5
    il = (i / 10).astype(int)
    jl = (j / 10).astype(int)
    ll = il + jl * 10 + 1

    center_labels = numpy.zeros((100, 100), int)
    center_labels[ir ** 2 + jr ** 2 < 25] = ll[ir ** 2 + jr ** 2 < 25]

    labels = numpy.zeros((100, 100), int)
    i = numpy.random.randint(1, 98, 2000)
    j = numpy.random.randint(1, 98, 2000)
    order = numpy.lexsort((i, j))
    i = i[order]
    j = j[order]
    duplicate = numpy.hstack([[False], (i[:-1] == i[1:]) & (j[:-1] == j[1:])])
    i = i[~duplicate]
    j = j[~duplicate]
    labels[i, j] = numpy.arange(1, len(i) + 1)
    image = numpy.random.uniform(size=(100, 100))
    #
    # Crash here prior to fix
    #
    m = run_module(image, labels, center_labels)


def test_more_centers_than_labels():
    """Regression test of img-1463"""
    numpy.random.seed(51)
    i, j = numpy.mgrid[0:100, 0:100]
    ir = (i % 10) - 5
    jr = (j % 10) - 5
    il = (i / 10).astype(int)
    jl = (j / 10).astype(int)
    ll = il + jl * 10 + 1

    labels = numpy.zeros((100, 100), int)
    labels[ir ** 2 + jr ** 2 < 25] = ll[ir ** 2 + jr ** 2 < 25]

    center_labels = numpy.zeros((100, 100), int)
    i = numpy.random.randint(1, 98, 2000)
    j = numpy.random.randint(1, 98, 2000)
    order = numpy.lexsort((i, j))
    i = i[order]
    j = j[order]
    duplicate = numpy.hstack([[False], (i[:-1] == i[1:]) & (j[:-1] == j[1:])])
    i = i[~duplicate]
    j = j[~duplicate]
    center_labels[i, j] = numpy.arange(1, len(i) + 1)
    image = numpy.random.uniform(size=(100, 100))
    #
    # Crash here prior to fix
    #
    m = run_module(image, labels, center_labels)
