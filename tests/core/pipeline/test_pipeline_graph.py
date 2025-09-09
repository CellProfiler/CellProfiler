import os
import io
import tests.core
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.pipeline.event import LoadException

def get_data_directory():
    folder = os.path.dirname(tests.core.__file__)
    return os.path.abspath(os.path.join(folder, "data/pipeline/"))

def get_pipeline(name):
    pipeline_file = os.path.join(
        get_data_directory(), f"{name}.cppipe"
    )
    with open(pipeline_file, "r") as fd:
        data = fd.read()

    pipeline = Pipeline()

    def callback(caller, event):
        assert not isinstance(event, LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))

    return pipeline

def setup_pipeline(name):
    pipeline = get_pipeline(name)
    for module in pipeline.modules():
        module.show_window = False
    return pipeline

def check_structure(dep_graph):
    """
    {
        modules: [
        {
            module_name: str,
            module_num: int,
            inputs: [
            {
                type: "image" | "object" | "measurement"
                name: str,
                source_module: str,
                source_module_num: int, # gte 1
                object_name: str, # if type == "measurement
                feature: str, # if type == "measurement
            },
            ...
            ],
            outputs: [
            {
                type: "image" | "object" | "measurement"
                name: str,
                destination_module: str,
                destination_module_num: int, # gte 1
                object_name: str, # if type == "measurement
                feature: str, # if type == "measurement
            },
            ...
            ],
        },
        ...
        ],
        metadata: 
        {
            total_modules: int,
            total_edges: int,
        }
    }
    """
    assert "metadata" in dep_graph
    assert "total_modules" in dep_graph["metadata"]
    assert "total_edges" in dep_graph["metadata"]

    assert "modules" in dep_graph
    assert isinstance(dep_graph["modules"], list)

    for module in dep_graph["modules"]:
        assert "module_name" in module
        assert "module_num" in module
        assert "inputs" in module
        assert "outputs" in module
        for input_item in module["inputs"]:
            assert "type" in input_item
            assert input_item["type"] in ("image", "object", "measurement")
            assert "name" in input_item
            assert "source_module" in input_item
            assert "source_module_num" in input_item
            if input_item["type"] == "measurement":
                assert "object_name" in input_item
                assert "feature" in input_item
        for output_item in module["inputs"]:
            assert "type" in output_item
            assert output_item["type"] in ("image", "object", "measurement")
            assert "name" in output_item
            assert "source_module" in output_item
            assert "source_module_num" in output_item
            if output_item["type"] == "measurement":
                assert "object_name" in output_item
                assert "feature" in output_item

def test_dependency_graph_structure():
    pipeline = setup_pipeline("ExampleFlyMeas")
    pipeline.describe_dependency_graph(edges=None, exclude_mes_leafs=True)
    dep_graph = pipeline.get_dependency_graph()

    check_structure(dep_graph)

    assert dep_graph["metadata"]["total_modules"] == len(pipeline.modules())
    assert len(dep_graph["modules"]) == len(pipeline.modules())
    assert dep_graph["metadata"]["total_edges"] == 445

def test_dependency_graph_liveness():
    """
    5 Crop - read `IM_orig_b` from disk, and write `IM_crop_b`
        `IM_orig_r`, `IM_orig_g`, `IM_orig_b` are input images
        `IM_crop_b` is needed for `IM_crop_g` (unfinished), `IM_crop_r` (unfinished), `OB_nuclei` (unfinished), `IM_rgb` (unfinished), `MES_intens` (unfinished), `MES_tex` (unfinished)
    6 Crop - read `IM_orig_g` and `IM_crop_b` from disk, and write `IM_crop_g`
        `IM_orig_r`, `IM_orig_g`, `IM_orig_b` are input images
        `IM_crop_b` is needed for `IM_crop_g` (finished), `IM_crop_r` (unfinished), `OB_nuclei` (unfinished), `IM_rgb` (unfinished), `MES_intens` (unfinished), `MES_tex` (unfinished)
        `IM_crop_g` is needed for `OB_cells` (unfinished), `IM_rgb` (unfinished)
    7 Crop - read `IM_orig_r`, and `IM_crop_b` from disk, and write `IM_crop_r`
        `IM_orig_r`, `IM_orig_g`, `IM_orig_b` are input images
        `IM_crop_b` is needed for `IM_crop_g` (finished), `IM_crop_r` (finished), `OB_nuclei` (unfinished), `IM_rgb` (unfinished), `MES_intens` (unfinished), `MES_tex` (unfinished)
        `IM_crop_g` is needed for `OB_cells` (unfinished), `IM_rgb` (unfinished)
        `IM_crop_r` is needed for `IM_rgb` (unfinished)
    8 IdentifyPrimary - read `IM_crop_b` from disk, and write `OB_nuclei`
        `IM_orig_r`, `IM_orig_g`, `IM_orig_b` are input images
        `IM_crop_b` is needed for `IM_crop_g` (finished), `IM_crop_r` (finished), `OB_nuclei` (finished), `IM_rgb` (unfinished), `MES_intens` (unfinished), `MES_tex` (unfinished)
        `IM_crop_g` is needed for `OB_cells` (unfinished), `IM_rgb` (unfinished)
        `IM_crop_r` is needed for `IM_rgb` (unfinished)
        `OB_nuclei` is needed for `OB_cells` (unfinished), `OB_cyto` (unfinished), `MES_size_shape` (unfinished), `MES_intens` (unfinished), `MES_tex` (unfinished)
    9 IdentifySecondary - read `IM_crop_g`, `OB_nuclei` from disk, write `OB_cells`
        `IM_orig_r`, `IM_orig_g`, `IM_orig_b` are input images
        `IM_crop_b` is needed for `IM_crop_g` (finished), `IM_crop_r` (finished), `OB_nuclei` (finished), `IM_rgb` (unfinished), `MES_intens` (unfinished), `MES_tex` (unfinished)
        `IM_crop_g` is needed for `OB_cells` (finished), `IM_rgb` (unfinished)
        `IM_crop_r` is needed for `IM_rgb` (unfinished)
        `OB_nuclei` is needed for `OB_cells` (finished), `OB_cyto` (unfinished), `MES_size_shape` (unfinished), `MES_intens` (unfinished), `MES_tex` (unfinished)
        `OB_cells` needed for `OB_cyto` (unfinished), `MES_size_shape` (unfinished), `MES_intens` (unfinished), `MES_tex` (unfinished)
    10 IdentifyTertiary - read `OB_cells`, `OB_nuclei` from disk, write `OB_cyto`
        `IM_orig_r`, `IM_orig_g`, `IM_orig_b` are input images
        `IM_crop_b` is needed for `IM_crop_g` (finished), `IM_crop_r` (finished), `OB_nuclei` (finished), and `IM_rgb` (unfinished)
        `IM_crop_g` is needed for `OB_cells` (finished), `IM_rgb` (unfinished)
        `IM_crop_r` is needed for `IM_rgb` (unfinished)
        `OB_nuclei` is needed for `OB_cells` (finished), `OB_cyto` (finished), `MES_size_shape` (unfinished), `MES_intens` (unfinished), `MES_tex` (unfinished)
        `OB_cells` needed for `OB_cyto` (finished), `MES_size_shape` (unfinished), `MES_intens` (unfinished), `MES_tex` (unfinished)
        `OB_cyto` needed for `MES_size_shape` (unfinished), `MES_intens` (unfinished), `MES_tex` (unfinished)
    11 MeasureObjectSizeAndShape - read `OB_cyto`, `OB_cells`, `OB_nuclei` from disk, store `MES_size_shape` measurement
        `IM_orig_r`, `IM_orig_g`, `IM_orig_b` are input images
        `IM_crop_b` is needed for `IM_crop_g` (finished), `IM_crop_r` (finished), `OB_nuclei` (finished), `IM_rgb` (unfinished), `MES_intens` (unfinished), `MES_tex` (unfinished)
        `IM_crop_g` is needed for `OB_cells` (finished), `IM_rgb` (unfinished)
        `IM_crop_r` is needed for `IM_rgb` (unfinished)
        `OB_nuclei` is needed for `OB_cells` (finished), `OB_cyto` (finished), `MES_size_shape` (finished), `MES_intens` (unfinished), `MES_tex` (unfinished)
        `OB_cells` needed for `OB_cyto` (finished), `MES_size_shape` (finished), `MES_intens` (unfinished), `MES_tex` (unfinished)
        `OB_cyto` needed for `MES_size_shape` (finished), `MES_intens` (unfinished), `MES_tex` (unfinished)
        `MES_size_shape` is stored and SAVED
    12 MeasureObjectIntensity - read `IM_crop_b`, `OB_cyto`, `OB_cells`, `OB_nuclei` from disk, store `MES_intens` measurement
        `IM_orig_r`, `IM_orig_g`, `IM_orig_b` are input images
        `IM_crop_b` is needed for `IM_crop_g` (finished), `IM_crop_r` (finished), `OB_nuclei` (finished), `IM_rgb` (unfinished), `MES_intens` (finished), `MES_tex` (unfinished)
        `IM_crop_g` is needed for `OB_cells` (finished), `IM_rgb` (unfinished)
        `IM_crop_r` is needed for `IM_rgb` (unfinished)
        `OB_nuclei` is needed for `OB_cells` (finished), `OB_cyto` (finished), `MES_size_shape` (finished), `MES_intens` (finished), `MES_tex` (unfinished)
        `OB_cells` needed for `OB_cyto` (finished), `MES_size_shape` (finished), `MES_intens` (finished), `MES_tex` (unfinished)
        `OB_cyto` needed for `MES_size_shape` (finished), `MES_intens` (finished), `MES_tex` (unfinished)
        `MES_size_shape` is stored and SAVED
        `MES_intens` is stored and SAVED
    13 MeasureTexture - read `IM_crop_b`, `OB_cyto`, `OB_cells`, `OB_nuclei` from disk, store `MES_tex` measurement
        `IM_orig_r`, `IM_orig_g`, `IM_orig_b` are input images
        `IM_crop_b` is needed for `IM_crop_g` (finished), `IM_crop_r` (finished), `OB_nuclei` (finished), `IM_rgb` (unfinished), `MES_intens` (finished), `MES_tex` (finished)
        `IM_crop_g` is needed for `OB_cells` (finished), `IM_rgb` (unfinished)
        `IM_crop_r` is needed for `IM_rgb` (unfinished)
        `OB_nuclei` is needed for `OB_cells` (finished), `OB_cyto` (finished), `MES_size_shape` (finished), `MES_intens` (finished), `MES_tex` (finished)
            delete `OB_nuclei`
        `OB_cells` needed for `OB_cyto` (finished), `MES_size_shape` (finished), `MES_intens` (finished), `MES_tex` (finished)
            delete `OB_cells`
        `OB_cyto` needed for `MES_size_shape` (finished), `MES_intens` (finished), `MES_tex` (finished)
            delete `OB_cyto`
        `MES_size_shape` is stored and SAVED
        `MES_intens` is stored and SAVED
        `MES_tex` is stored and SAVED
    14 GrayToColor - read `IM_crop_b`, `IM_crop_g`, `IM_crop_r` from disk, write `IM_rgb`
        `IM_orig_r`, `IM_orig_g`, `IM_orig_b` are input images
        `IM_crop_b` is needed for `IM_crop_g` (finished), `IM_crop_r` (finished), `OB_nuclei` (finished), `IM_rgb` (finished), `MES_intens` (finished), `MES_tex` (finished)
            delete `IM_crop_b`
        `IM_crop_g` is needed for `OB_cells` (finished), `IM_rgb` (finished)
            delete `IM_crop_g`
        `IM_crop_r` is needed for `IM_rgb` (finished)
            delete `IM_crop_r`
        `MES_size_shape` is stored and SAVED
        `MES_intens` is stored and SAVED
        `MES_tex` is stored and SAVED
        `IM_rgb` is needed for `IM_fly` (unfinished)
    15 SaveImages - read `IM_rgb` from disk, write `IM_fly`
        `IM_orig_r`, `IM_orig_g`, `IM_orig_b` are input images
        `MES_size_shape` is stored and SAVED
        `MES_intens` is stored and SAVED
        `MES_tex` is stored and SAVED
        `IM_rgb` is needed for `IM_fly` (finished)
            delete `IM_rgb`
        `IM_fly` is on disk and SAVED
    """
    pipeline = setup_pipeline("ExampleFlyMeas")
    pipeline.describe_dependency_graph(edges=None, exclude_mes_leafs=True)
    dep_graph = pipeline.get_dependency_graph(liveness=True)
    modules = dep_graph["modules"]

    def _module_gen():
        for i in range(17):
            yield modules[i]

    module_gen = _module_gen()

    def validate_module_liveness(module, live_expected, disposed_expected):
        assert len(module["live"]) == len(live_expected)
        for im_name in live_expected:
            assert im_name in module["live"]
        assert len(module["disposed"]) == len(disposed_expected)
        for im_name in disposed_expected:
            assert im_name in module["disposed"]

    # Images - 1
    validate_module_liveness(next(module_gen), [], [])
    # Metadata - 2
    validate_module_liveness(next(module_gen), [], [])
    # NamesAndTypes - 3
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed"],
        []
    )
    # Groups - 4
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed"],
        []
    )
    # Crop (Blue) - 5
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed", "CropBlue"],
        []
    )
    # Crop (Green) - 6
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed", "CropBlue", "CropGreen"],
        []
    )
    # Crop (Red) - 7
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed", "CropBlue", "CropGreen", "CropRed"],
        []
    )
    # IdentifyPrimaryObjects - 8
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed", "CropBlue", "CropGreen", "CropRed", "Nuclei"],
        []
    )
    # IdentifySecondaryObjects - 9
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed", "CropBlue", "CropGreen", "CropRed", "Nuclei", "Cells"],
        []
    )
    # IdentifyTertiaryObjects - 10
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed", "CropBlue", "CropGreen", "CropRed", "Nuclei", "Cells", "Cytoplasm"],
        []
    )
    # MeasureObjectSizeShape - 11
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed", "CropBlue", "CropGreen", "CropRed", "Nuclei", "Cells", "Cytoplasm"],
        []
    )
    # MeasureObjectIntensity - 12
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed", "CropBlue", "CropGreen", "CropRed", "Nuclei", "Cells", "Cytoplasm"],
        []
    )
    # MeasureTexture - 13
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed", "CropBlue", "CropGreen", "CropRed", "Nuclei", "Cells", "Cytoplasm"],
        []
    )
    # GrayToColor - 14
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed", "Nuclei", "Cells", "Cytoplasm", "RGBImage"],
        ["CropBlue", "CropGreen", "CropRed"]
    )
    # ExpandOrShrinkObjects- 15
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed", "Nuclei", "Cells", "Cytoplasm", "RGBImage"],
        ["ShrunkenNuclei"]
    )
    # SaveImages - 16
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed", "Nuclei", "Cells", "Cytoplasm"],
        ["RGBImage"]
    )
    # ExportToSpreadsheet - 17
    validate_module_liveness(
        next(module_gen),
        ["OrigRed", "OrigGreen", "OrigRed"],
        ["Nuclei", "Cells", "Cytoplasm"]
    )

    print('DONE')
