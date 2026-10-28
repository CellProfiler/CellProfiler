from itertools import product

import joblib
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cellprofiler.modules.classifyobjects import BY_MODEL, M_CATEGORY, ClassifyObjects
from cellprofiler.modules.filterobjects import FilterObjects
from cellprofiler.modules.flagimage import S_CLASSIFIER, FlagImage
from cellprofiler_core.constants.workspace import DISPOSITION_CONTINUE, DISPOSITION_SKIP
from cellprofiler_core.image import ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet, Objects
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.preferences import ABSOLUTE_FOLDER_NAME
from cellprofiler_core.workspace import Workspace
from cellprofiler_library.opts.filterobjects import FilterMode

MODEL_NAMES = [
    "forest",
    "gradient_boosting",
    "logistic",
    "naive_bayes",
    "neighbors",
    "svc",
]
CLASS_NAMES = ["Low", "Middle", "High"]
FEATURE_NAMES = [
    "Texture_Contrast_DNA",
    "Intensity_MeanIntensity_DNA",
    "AreaShape_Area",
]
IMAGE_FEATURE_NAMES = ["QC_Third", "QC_First", "QC_Second"]


def training_data(dtype):
    random = np.random.RandomState(314)
    train = (random.normal(size=(180, 3)) * [1, 3, 7] + [2, -4, 11]).astype(dtype)
    target = (
        np.digitize(train[:, 0] + train[:, 1] * 0.1 - train[:, 2] * 0.05, [0.5, 2.0])
        + 1
    )
    query = (random.normal(size=(12, 3)) * [1, 3, 7] + [2, -4, 11]).astype(dtype)
    return train, target, query


def fit_classifier(name, dtype, representation):
    train, target, query = training_data(dtype)
    constructors = {
        "forest": lambda: RandomForestClassifier(n_estimators=25, random_state=314),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=25, random_state=314
        ),
        "logistic": lambda: LogisticRegression(random_state=314, max_iter=1000),
        "naive_bayes": GaussianNB,
        "neighbors": KNeighborsClassifier,
        "svc": lambda: SVC(probability=True, random_state=314),
    }
    model = constructors[name]()
    if representation == "pipeline":
        model = make_pipeline(StandardScaler(), model)
    if representation == "attached":
        model.scaler = StandardScaler().fit(train)
        model.fit(model.scaler.transform(train), target)
    else:
        model.fit(train, target)
    return {"model": model, "query": query, "train": train, "target": target}


def model_input(bundle):
    model, query = bundle["model"], bundle["query"]
    return model.scaler.transform(query) if hasattr(model, "scaler") else query


def save_model(path, bundle, features):
    model = bundle["model"]
    joblib.dump((model, CLASS_NAMES, type(model).__name__, list(features)), path)


def workspace_for(module, query=None):
    module.set_module_num(1)
    pipeline = Pipeline()
    pipeline.add_module(module)
    measurements = Measurements()
    image_set_list = ImageSetList()
    object_set = ObjectSet()
    workspace = Workspace(
        pipeline,
        module,
        image_set_list.get_image_set(0),
        object_set,
        measurements,
        image_set_list,
    )
    if query is not None:
        objects = Objects()
        objects.segmented = np.arange(1, len(query) + 1, dtype=np.int32).reshape(3, 4)
        object_set.add_objects(objects, "Cells")
        for name, values in reversed(list(zip(FEATURE_NAMES, query.T))):
            measurements.add_measurement("Cells", name, values)
    return workspace


def set_model_path(directory, filename, path):
    directory.dir_choice = ABSOLUTE_FOLDER_NAME
    directory.custom_path = str(path.parent)
    filename.value = path.name


def selected_labels(labels, hits):
    mapping = np.zeros(len(hits) + 1, dtype=int)
    indexes = np.flatnonzero(hits) + 1
    mapping[indexes] = np.arange(1, len(indexes) + 1)
    return mapping[labels]


def classify_outputs(bundle, path):
    module = ClassifyObjects()
    module.contrast_choice.value = BY_MODEL
    module.object_name.value = "Cells"
    module.create_class_sets.value = True
    set_model_path(module.model_directory, module.model_file_name, path)
    while len(module.desired_classes) < len(CLASS_NAMES):
        module.add_single_class()
    for settings, label in zip(module.desired_classes, CLASS_NAMES):
        settings.class_name.value = label
        settings.class_objects_name.value = "Class_" + label
    workspace = workspace_for(module, bundle["query"])
    try:
        loaded = module.get_classifier()
        assert module.get_classifier() is loaded
        assert module.get_classifier_features() == [
            "Cells_" + name for name in FEATURE_NAMES
        ]
        assert module.get_bin_labels() == CLASS_NAMES
        module.run(workspace)
        expected = bundle["model"].predict(model_input(bundle))
        probabilities = bundle["model"].predict_proba(model_input(bundle))
        measurements = workspace.measurements
        classes = np.asarray(
            measurements.get_current_measurement("Cells", f"{M_CATEGORY}_Class"),
            dtype=str,
        )
        np.testing.assert_array_equal(
            classes, [CLASS_NAMES[index - 1] for index in expected]
        )
        actual_probabilities = np.column_stack(
            [
                measurements.get_current_measurement(
                    "Cells", f"{M_CATEGORY}_Probability_{name}"
                )
                for name in CLASS_NAMES
            ]
        )
        np.testing.assert_array_equal(actual_probabilities, probabilities)
        counts = np.array(
            [
                measurements.get_current_image_measurement(f"{M_CATEGORY}_Count_{name}")
                for name in CLASS_NAMES
            ]
        )
        np.testing.assert_array_equal(
            counts, [np.count_nonzero(expected == index) for index in [1, 2, 3]]
        )
        result = {
            "classes": classes,
            "probabilities": actual_probabilities,
            "counts": counts,
        }
        labels = workspace.object_set.get_objects("Cells").segmented
        for index, name in enumerate(CLASS_NAMES, 1):
            actual = workspace.object_set.get_objects("Class_" + name).segmented
            np.testing.assert_array_equal(
                actual, selected_labels(labels, expected == index)
            )
            result["objects/" + name] = actual.copy()
        assert module.get_dictionary_for_worker() == {}
        return result
    finally:
        workspace.measurements.close()


def filter_outputs(bundle, path):
    module = FilterObjects()
    module.x_name.value = "Cells"
    module.y_name.value = "Kept"
    module.mode.value = FilterMode.CLASSIFIERS.value
    module.rules_class.value = CLASS_NAMES[0]
    module.keep_removed_objects.value = True
    module.removed_objects_name.value = "Removed"
    set_model_path(module.rules_directory, module.rules_file_name, path)
    workspace = workspace_for(module, bundle["query"])
    try:
        assert module.get_classifier_features() == [
            "Cells_" + name for name in FEATURE_NAMES
        ]
        loaded = module.get_classifier()
        assert module.get_classifier() is loaded
        module.run(workspace)
        hits = bundle["model"].predict(model_input(bundle)) == 1
        labels = workspace.object_set.get_objects("Cells").segmented
        kept = workspace.object_set.get_objects("Kept").segmented
        removed = workspace.object_set.get_objects("Removed").segmented
        np.testing.assert_array_equal(kept, selected_labels(labels, hits))
        np.testing.assert_array_equal(removed, selected_labels(labels, ~hits))
        measurements = workspace.measurements
        parents = measurements.get_current_measurement("Kept", "Parent_Cells")
        children = measurements.get_current_measurement("Cells", "Children_Kept_Count")
        np.testing.assert_array_equal(parents, np.flatnonzero(hits) + 1)
        np.testing.assert_array_equal(children, hits.astype(int))
        counts = np.array(
            [
                measurements.get_current_image_measurement("Count_Kept"),
                measurements.get_current_image_measurement("Count_Removed"),
            ]
        )
        np.testing.assert_array_equal(
            counts, [np.count_nonzero(hits), np.count_nonzero(~hits)]
        )
        return {
            "kept": kept.copy(),
            "removed": removed.copy(),
            "parents": parents.copy(),
            "children": children.copy(),
            "counts": counts,
        }
    finally:
        workspace.measurements.close()


def flag_outputs(bundle, path):
    module = FlagImage()
    flag = module.flags[0]
    flag.category.value = "QC"
    flag.feature_name.value = "ClassifierFlag"
    flag.wants_skip.value = True
    module.ignore_flag_on_last.value = False
    settings = flag.measurement_settings[0]
    settings.source_choice.value = S_CLASSIFIER
    settings.rules_class.value = [CLASS_NAMES[0], CLASS_NAMES[2]]
    set_model_path(settings.rules_directory, settings.rules_file_name, path)
    values, dispositions = [], []
    expected = bundle["model"].predict(bundle["query"])
    loaded = module.get_classifier(settings)
    assert module.get_classifier(settings) is loaded
    assert module.get_classifier_features(settings) == IMAGE_FEATURE_NAMES
    for row, predicted in zip(bundle["query"], expected):
        workspace = workspace_for(module)
        try:
            for name, value in reversed(list(zip(IMAGE_FEATURE_NAMES, row))):
                workspace.measurements.add_image_measurement(name, value)
            module.run(workspace)
            value = workspace.measurements.get_current_image_measurement(
                "QC_ClassifierFlag"
            )
            fail = predicted in [1, 3]
            assert value == int(fail)
            assert workspace.disposition == (
                DISPOSITION_SKIP if fail else DISPOSITION_CONTINUE
            )
            values.append(value)
            dispositions.append(int(workspace.disposition == DISPOSITION_SKIP))
        finally:
            workspace.measurements.close()
    return {"flags": np.asarray(values), "skip": np.asarray(dispositions)}


CASES = list(product(MODEL_NAMES, ["float32", "float64"], ["raw", "pipeline"]))


@pytest.fixture(scope="module", params=CASES, ids=lambda case: "-".join(case))
def trained_model(request, tmp_path_factory):
    name, dtype, representation = request.param
    bundle = fit_classifier(name, dtype, representation)
    directory = tmp_path_factory.mktemp("classifier-" + "-".join(request.param))
    bundle["object_path"] = directory / "objects.model"
    bundle["image_path"] = directory / "images.model"
    save_model(
        bundle["object_path"], bundle, ["Cells_" + name for name in FEATURE_NAMES]
    )
    save_model(bundle["image_path"], bundle, IMAGE_FEATURE_NAMES)
    loaded = joblib.load(bundle["object_path"])
    assert loaded[1:] == (
        CLASS_NAMES,
        type(bundle["model"]).__name__,
        ["Cells_" + name for name in FEATURE_NAMES],
    )
    np.testing.assert_array_equal(
        loaded[0].predict(bundle["query"]), bundle["model"].predict(bundle["query"])
    )
    np.testing.assert_array_equal(
        loaded[0].predict_proba(bundle["query"]),
        bundle["model"].predict_proba(bundle["query"]),
    )
    return bundle


def test_classifyobjects_matching_version_model(trained_model):
    classify_outputs(trained_model, trained_model["object_path"])


def test_filterobjects_matching_version_model(trained_model):
    filter_outputs(trained_model, trained_model["object_path"])


def test_flagimage_matching_version_model(trained_model):
    flag_outputs(trained_model, trained_model["image_path"])


@pytest.mark.parametrize(
    "name,dtype,consumer",
    list(
        product(
            ["gradient_boosting", "logistic"],
            ["float32", "float64"],
            ["classify", "filter"],
        )
    ),
)
def test_cpa_attached_scaler(tmp_path, name, dtype, consumer):
    bundle = fit_classifier(name, dtype, "attached")
    path = tmp_path / "scaled.model"
    save_model(path, bundle, ["Cells_" + name for name in FEATURE_NAMES])
    if consumer == "classify":
        classify_outputs(bundle, path)
    else:
        filter_outputs(bundle, path)
