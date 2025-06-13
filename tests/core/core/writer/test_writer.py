import os
import io
import tests.core
import cellprofiler_core.pipeline
import cellprofiler_core.modules.namesandtypes

def get_data_directory():
    folder = os.path.dirname(tests.core.__file__)
    return os.path.abspath(os.path.join(folder, "data/"))

def get_pipeline():
    pipeline_file = os.path.join(
        get_data_directory(), "tiled/tiled.cppipe"
    )
    with open(pipeline_file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))

    return pipeline

def test_01_load_pipeline():
    pipeline = get_pipeline()

    assert len(pipeline.modules()) == 4
    module = pipeline.modules()[2]
    assert isinstance(module, cellprofiler_core.modules.namesandtypes.NamesAndTypes)

    assert (
        module.assignment_method == cellprofiler_core.modules.namesandtypes.ASSIGN_ALL
    )
    assert (
        module.single_load_as_choice
            == cellprofiler_core.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE
    )
    assert module.single_image_provider.value == "DNA"

    assert module.assignments_count.value == 1
    assert (
        module.single_rescale_method.value
            == cellprofiler_core.modules.namesandtypes.INTENSITY_RESCALING_BY_DATATYPE
    )

    assert module.process_as_tiled.value # "Yes"
    assert not module.process_as_3d.value # "No"
