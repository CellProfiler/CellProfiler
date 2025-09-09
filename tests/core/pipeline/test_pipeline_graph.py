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
    # not needed for test, but nice debug info
    pipeline.describe_dependency_graph(edges=None, exclude_mes_leafs=True)
    dep_graph = pipeline.get_dependency_graph()

    check_structure(dep_graph)

    assert dep_graph["metadata"]["total_modules"] == len(pipeline.modules())
    assert len(dep_graph["modules"]) == len(pipeline.modules())
    assert dep_graph["metadata"]["total_edges"] == 445

def test_dependency_graph_liveness():
    pipeline = setup_pipeline("ExampleFlyMeas")
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
