import base64
import io
import os
import tempfile
import zlib

import imageio

import centrosome.outline
import numpy
import scipy.ndimage

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.modules
import cellprofiler.modules.untangleworms
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler_core.setting
import cellprofiler_core.workspace
import tests.frontend.modules

cellprofiler.modules.untangleworms.CAROLINAS_HACK = False

IMAGE_NAME = "myimage"
OVERLAP_OBJECTS_NAME = "overlapobjects"
NON_OVERLAPPING_OBJECTS_NAME = "nonoverlappingobjects"
OVERLAPPING_OUTLINES_NAME = "overlapoutlines"
NON_OVERLAPPING_OUTLINES_NAME = "nonoverlappingoutlines"

A02_binary = (
    "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"
    "hQdAh4mRheEDHOHyfwASV/wk"
)

PARAMS = (
    "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"
    "HXiW/wDc+AeK"
)

handle, path = tempfile.mkstemp(suffix=".png")
fd = os.fdopen(handle, "wb")
fd.write(zlib.decompress(base64.b64decode(A02_binary)))
fd.close()

A02_image = imageio.imread(path)[:, :, 0] > 0


def test_load_v1():
    file = tests.frontend.modules.get_test_resources_directory("untangleworms/v1.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 3
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.untangleworms.UntangleWorms)
    assert module.image_name == "BinaryWorms"
    assert module.overlap == cellprofiler.modules.untangleworms.OO_BOTH
    assert module.overlap_objects == "OverlappingWorms"
    assert module.nonoverlapping_objects == "NonOverlappingWorms"
    assert not module.wants_training_set_weights
    assert module.override_overlap_weight.value == 3
    assert module.override_leftover_weight.value == 5
    assert module.mode == cellprofiler.modules.untangleworms.MODE_TRAIN
    module = pipeline.modules()[1]
    assert isinstance(module, cellprofiler.modules.untangleworms.UntangleWorms)
    assert module.overlap == cellprofiler.modules.untangleworms.OO_WITH_OVERLAP
    assert module.wants_training_set_weights
    assert module.mode == cellprofiler.modules.untangleworms.MODE_UNTANGLE
    module = pipeline.modules()[2]
    assert isinstance(module, cellprofiler.modules.untangleworms.UntangleWorms)
    assert module.overlap == cellprofiler.modules.untangleworms.OO_WITHOUT_OVERLAP
    assert module.mode == cellprofiler.modules.untangleworms.MODE_UNTANGLE
    assert module.complexity == cellprofiler.modules.untangleworms.C_ALL
    assert module.custom_complexity == 400


def test_load_v2():
    file = tests.frontend.modules.get_test_resources_directory("untangleworms/v2.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 5
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.untangleworms.UntangleWorms)
    assert module.image_name == "BinaryWorms"
    assert module.overlap == cellprofiler.modules.untangleworms.OO_BOTH
    assert module.overlap_objects == "OverlappingWorms"
    assert module.nonoverlapping_objects == "NonOverlappingWorms"
    assert not module.wants_training_set_weights
    assert module.override_overlap_weight.value == 3
    assert module.override_leftover_weight.value == 5
    assert module.custom_complexity == 399

    for module, complexity in zip(
        pipeline.modules(),
        (
            cellprofiler.modules.untangleworms.C_ALL,
            cellprofiler.modules.untangleworms.C_MEDIUM,
            cellprofiler.modules.untangleworms.C_HIGH,
            cellprofiler.modules.untangleworms.C_VERY_HIGH,
            cellprofiler.modules.untangleworms.C_CUSTOM,
        ),
    ):
        assert module.complexity == complexity


def make_workspace(image, data=None, write_mode="wb"):
    """Make a workspace to run the given image and params file

    image - a binary image
    data - the binary of the params file to run
    """
    if data is not None:
        with open(path, "wb") as fd:
            fd.write(data)

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(
            event,
            (
                cellprofiler_core.pipeline.event.RunException,
                cellprofiler_core.pipeline.event.LoadException,
            ),
        )

    pipeline.add_listener(callback)
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    module.image_name.value = IMAGE_NAME
    module.nonoverlapping_objects.value = NON_OVERLAPPING_OBJECTS_NAME
    module.overlap_objects.value = OVERLAP_OBJECTS_NAME
    module.set_module_num(1)
    pipeline.add_module(module)
    img = cellprofiler_core.image.Image(image)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add(IMAGE_NAME, img)
    module.training_set_directory.dir_choice = (
        cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
    )
    (
        module.training_set_directory.custom_path,
        module.training_set_file_name.value,
    ) = os.path.split(path)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return workspace, module


def make_params(d):
    """Make a fake params structure from a dictionary

    e.g., x = dict(foo=dict(bar=5)) -> x.foo.bar = 5
    """

    class X(object):
        def __init__(self, d):
            for key in list(d.keys()):
                value = d[key]
                if isinstance(value, dict):
                    value = X(value)
                setattr(self, key, value)

    return X(d)


def test_load_params():
    data = zlib.decompress(base64.b64decode(PARAMS))
    workspace, module = make_workspace(numpy.zeros((10, 10), bool), data)
    assert isinstance(module, cellprofiler.modules.untangleworms.UntangleWorms)
    module.prepare_group(workspace, None, None)
    params = module.read_params()
    assert round(abs(params.min_worm_area - 601.2), 0) == 0
    assert round(abs(params.max_area - 1188.5), 0) == 0
    assert round(abs(params.cost_threshold - 200.8174), 3) == 0
    assert params.num_control_points == 21
    numpy.testing.assert_almost_equal(
        params.mean_angles,
        numpy.array(
            [
                -0.00527796256445404,
                -0.0315202989978013,
                -0.00811839821858939,
                -0.0268318268190547,
                -0.0120476701544335,
                -0.0202651421433172,
                -0.0182919505672029,
                -0.00990476055380843,
                -0.0184558846126189,
                -0.0100827620455749,
                -0.0121729201775023,
                -0.0129790204861452,
                0.0170195137830520,
                0.00185471766328753,
                0.00913261528049730,
                -0.0106805477987750,
                0.00473369321673608,
                -0.00835547063778011,
                -0.00382606935405797,
                127.129708001680,
            ]
        ),
    )
    numpy.testing.assert_almost_equal(
        params.inv_angles_covariance_matrix,
        numpy.array(
            [
                [
                    16.1831499639022,
                    -5.06131059821028,
                    -7.03307454602146,
                    -2.88853420387429,
                    3.34017866010302,
                    3.45512576204933,
                    -1.09841786238497,
                    -2.79348760430306,
                    -1.85931734891389,
                    -0.652858408126458,
                    -1.22752367898365,
                    4.15573185568687,
                    1.99443112453893,
                    -2.26823701209981,
                    -1.25144655688072,
                    0.321931535265876,
                    0.230928100005575,
                    1.47235070063732,
                    0.487902558505382,
                    -0.0240787101275336,
                ],
                [
                    -5.06131059821028,
                    25.7442868663138,
                    -7.04197641326958,
                    -13.5057449289369,
                    -2.23928687231578,
                    4.31232681113232,
                    6.56454500463435,
                    0.336556097291049,
                    0.175759837346977,
                    -2.77098858956934,
                    0.307050758321026,
                    -2.12899901988826,
                    1.32985035426604,
                    2.77299577778623,
                    6.03717697873141,
                    -2.84152938638523,
                    -2.50027246248360,
                    2.88188404703382,
                    -2.94724985136021,
                    -0.00349792622125952,
                ],
                [
                    -7.03307454602146,
                    -7.04197641326958,
                    34.8868022738369,
                    -2.41698367836302,
                    -11.9074612429652,
                    -5.03219465159153,
                    0.566581294377262,
                    4.65965515408864,
                    4.40918302814844,
                    2.12317351869194,
                    -1.29767770791342,
                    -4.66814018817306,
                    -1.18082874743096,
                    3.51827877502392,
                    2.85186107108145,
                    -1.26716616779540,
                    -1.09593786866014,
                    -2.32869644778286,
                    3.48194316456812,
                    0.0623642923643842,
                ],
                [
                    -2.88853420387429,
                    -13.5057449289369,
                    -2.41698367836302,
                    44.6605979566156,
                    0.348020753098590,
                    -17.3115366179766,
                    -12.7256060767026,
                    5.70440571321352,
                    6.41590904344264,
                    -0.304578996360776,
                    1.47801450095277,
                    -0.908814536484512,
                    -1.48164287245030,
                    -2.34708447702134,
                    -2.23115474987353,
                    2.88954249368066,
                    2.74733203099146,
                    -3.04745351430166,
                    2.86603729585242,
                    0.00665888346219492,
                ],
                [
                    3.34017866010302,
                    -2.23928687231578,
                    -11.9074612429652,
                    0.348020753098590,
                    46.4099158672193,
                    -3.25559842794185,
                    -21.3692910255085,
                    -10.0863357869636,
                    -1.88512464797353,
                    -5.09750253669453,
                    1.04201155533543,
                    9.59270554140012,
                    0.145271525356847,
                    -5.72994886885862,
                    -6.15723880027164,
                    1.88718468304502,
                    0.283089642962522,
                    1.66577561191334,
                    -3.04240109786268,
                    0.0462492758625229,
                ],
                [
                    3.45512576204933,
                    4.31232681113232,
                    -5.03219465159153,
                    -17.3115366179766,
                    -3.25559842794185,
                    57.1526341670534,
                    7.57995189380428,
                    -30.3232000529691,
                    -13.9152901068606,
                    1.20521891799049,
                    9.87949588048337,
                    10.3794242331984,
                    -4.19429285108215,
                    -9.73213279908661,
                    0.320110631214874,
                    4.02636261974896,
                    -1.45438578469807,
                    -2.11793646742091,
                    -1.21519495438964,
                    -0.00660397622823739,
                ],
                [
                    -1.09841786238497,
                    6.56454500463435,
                    0.566581294377262,
                    -12.7256060767026,
                    -21.3692910255085,
                    7.57995189380428,
                    55.2613391936066,
                    -6.13432369324871,
                    -20.0367084309419,
                    -4.90180311830919,
                    5.26313027293073,
                    1.43916280645744,
                    -0.336838408057983,
                    2.29636776603810,
                    5.18308930951763,
                    -1.98288423853561,
                    -2.53995069544169,
                    -1.21462394180208,
                    1.97319648119690,
                    -0.0627444956856546,
                ],
                [
                    -2.79348760430306,
                    0.336556097291049,
                    4.65965515408864,
                    5.70440571321352,
                    -10.0863357869636,
                    -30.3232000529691,
                    -6.13432369324871,
                    79.4076938002120,
                    11.3722618218994,
                    -26.1279088515910,
                    -18.2231695390128,
                    -2.67921008934274,
                    8.52472948160932,
                    3.40897885951299,
                    -0.0156673992253622,
                    0.391511866283792,
                    2.43961939136012,
                    -4.02463696447393,
                    1.21200189852376,
                    -0.0276025739334060,
                ],
                [
                    -1.85931734891389,
                    0.175759837346977,
                    4.40918302814844,
                    6.41590904344264,
                    -1.88512464797353,
                    -13.9152901068606,
                    -20.0367084309419,
                    11.3722618218994,
                    62.2896410297082,
                    -4.73903263913512,
                    -17.7829659347680,
                    -17.3704452255960,
                    -1.11146124458858,
                    2.69303406406718,
                    5.35251557583661,
                    7.57574529347207,
                    -2.24432157539803,
                    -1.01589845612927,
                    2.74166325038759,
                    0.00616263316699783,
                ],
                [
                    -0.652858408126458,
                    -2.77098858956934,
                    2.12317351869194,
                    -0.304578996360776,
                    -5.09750253669453,
                    1.20521891799049,
                    -4.90180311830919,
                    -26.1279088515910,
                    -4.73903263913512,
                    53.6045562587984,
                    1.49208866907909,
                    -18.7976123565674,
                    -15.3160914187456,
                    4.62369094805509,
                    6.25594149186720,
                    -1.86433478999824,
                    2.40465791383637,
                    0.860045694295453,
                    -5.03379983998103,
                    -0.00250271852621389,
                ],
                [
                    -1.22752367898365,
                    0.307050758321026,
                    -1.29767770791342,
                    1.47801450095277,
                    1.04201155533543,
                    9.87949588048337,
                    5.26313027293073,
                    -18.2231695390128,
                    -17.7829659347680,
                    1.49208866907909,
                    52.6257209831917,
                    4.12959322744854,
                    -12.4142184568466,
                    -9.82200985900629,
                    -3.97638811187418,
                    1.15423868070705,
                    6.11175904439983,
                    2.88103313127626,
                    -0.0202321884301434,
                    -0.0770486841949908,
                ],
                [
                    4.15573185568687,
                    -2.12899901988826,
                    -4.66814018817306,
                    -0.908814536484512,
                    9.59270554140012,
                    10.3794242331984,
                    1.43916280645744,
                    -2.67921008934274,
                    -17.3704452255960,
                    -18.7976123565674,
                    4.12959322744854,
                    59.4024702854915,
                    -0.643591318201096,
                    -17.8872119905991,
                    -14.6283664729331,
                    0.921599492881119,
                    0.898368585097109,
                    2.02174339844234,
                    1.40192545975918,
                    0.0866552397218132,
                ],
                [
                    1.99443112453893,
                    1.32985035426604,
                    -1.18082874743096,
                    -1.48164287245030,
                    0.145271525356847,
                    -4.19429285108215,
                    -0.336838408057983,
                    8.52472948160932,
                    -1.11146124458858,
                    -15.3160914187456,
                    -12.4142184568466,
                    -0.643591318201096,
                    51.7737313171135,
                    -2.98701120529969,
                    -20.1761854847240,
                    -5.56229914135487,
                    2.56729654359925,
                    1.84129317709747,
                    2.90488161640993,
                    -0.0294908776237644,
                ],
                [
                    -2.26823701209981,
                    2.77299577778623,
                    3.51827877502392,
                    -2.34708447702134,
                    -5.72994886885862,
                    -9.73213279908661,
                    2.29636776603810,
                    3.40897885951299,
                    2.69303406406718,
                    4.62369094805509,
                    -9.82200985900629,
                    -17.8872119905991,
                    -2.98701120529969,
                    38.7045301665171,
                    0.0132666292353374,
                    -12.2104685640281,
                    -4.94147299831573,
                    3.32199768047190,
                    -0.225506641087443,
                    0.0431435753786928,
                ],
                [
                    -1.25144655688072,
                    6.03717697873141,
                    2.85186107108145,
                    -2.23115474987353,
                    -6.15723880027164,
                    0.320110631214874,
                    5.18308930951763,
                    -0.0156673992253622,
                    5.35251557583661,
                    6.25594149186720,
                    -3.97638811187418,
                    -14.6283664729331,
                    -20.1761854847240,
                    0.0132666292353374,
                    50.0037711812637,
                    1.51067910097860,
                    -12.7274402250448,
                    -7.12911129084980,
                    2.74828112041922,
                    0.0251424008457656,
                ],
                [
                    0.321931535265876,
                    -2.84152938638523,
                    -1.26716616779540,
                    2.88954249368066,
                    1.88718468304502,
                    4.02636261974896,
                    -1.98288423853561,
                    0.391511866283792,
                    7.57574529347207,
                    -1.86433478999824,
                    1.15423868070705,
                    0.921599492881119,
                    -5.56229914135487,
                    -12.2104685640281,
                    1.51067910097860,
                    46.6921496750757,
                    -8.43566372164764,
                    -15.0997112563034,
                    3.65384550078426,
                    -0.00453606919854300,
                ],
                [
                    0.230928100005575,
                    -2.50027246248360,
                    -1.09593786866014,
                    2.74733203099146,
                    0.283089642962522,
                    -1.45438578469807,
                    -2.53995069544169,
                    2.43961939136012,
                    -2.24432157539803,
                    2.40465791383637,
                    6.11175904439983,
                    0.898368585097109,
                    2.56729654359925,
                    -4.94147299831573,
                    -12.7274402250448,
                    -8.43566372164764,
                    32.8028397335256,
                    -3.87691864187466,
                    -6.41814264219731,
                    -0.0905326089208310,
                ],
                [
                    1.47235070063732,
                    2.88188404703382,
                    -2.32869644778286,
                    -3.04745351430166,
                    1.66577561191334,
                    -2.11793646742091,
                    -1.21462394180208,
                    -4.02463696447393,
                    -1.01589845612927,
                    0.860045694295453,
                    2.88103313127626,
                    2.02174339844234,
                    1.84129317709747,
                    3.32199768047190,
                    -7.12911129084980,
                    -15.0997112563034,
                    -3.87691864187466,
                    28.0765623007788,
                    -8.07326731403660,
                    0.0228470307583052,
                ],
                [
                    0.487902558505382,
                    -2.94724985136021,
                    3.48194316456812,
                    2.86603729585242,
                    -3.04240109786268,
                    -1.21519495438964,
                    1.97319648119690,
                    1.21200189852376,
                    2.74166325038759,
                    -5.03379983998103,
                    -0.0202321884301434,
                    1.40192545975918,
                    2.90488161640993,
                    -0.225506641087443,
                    2.74828112041922,
                    3.65384550078426,
                    -6.41814264219731,
                    -8.07326731403660,
                    18.1513394317232,
                    0.0139561765245330,
                ],
                [
                    -0.0240787101275336,
                    -0.00349792622125952,
                    0.0623642923643842,
                    0.00665888346219492,
                    0.0462492758625229,
                    -0.00660397622823739,
                    -0.0627444956856546,
                    -0.0276025739334060,
                    0.00616263316699783,
                    -0.00250271852621389,
                    -0.0770486841949908,
                    0.0866552397218132,
                    -0.0294908776237644,
                    0.0431435753786928,
                    0.0251424008457656,
                    -0.00453606919854300,
                    -0.0905326089208310,
                    0.0228470307583052,
                    0.0139561765245330,
                    0.00759059668024605,
                ],
            ]
        ),
    )
    assert round(abs(params.max_radius - 5.0990), 3) == 0
    assert round(abs(params.max_skel_length - 155.4545), 3) == 0
    assert round(abs(params.min_path_length - 84.401680541266530), 7) == 0
    assert round(abs(params.max_path_length - 171.8155554421827), 7) == 0
    assert round(abs(params.median_worm_area - 1007.5), 7) == 0
    assert round(abs(params.worm_radius - 5.099019527435303), 7) == 0
    assert params.overlap_weight == 5
    assert params.leftover_weight == 10
    numpy.testing.assert_almost_equal(
        params.radii_from_training,
        numpy.array(
            [
                1.19132055711746,
                2.75003945541382,
                3.56039281511307,
                4.05681743049622,
                4.39353294944763,
                4.52820824432373,
                4.66245639991760,
                4.75254730796814,
                4.76993056106567,
                4.78852712249756,
                4.73509162521362,
                4.76029792976379,
                4.75030451583862,
                4.69090248298645,
                4.59827183151245,
                4.55065062236786,
                4.35989559841156,
                4.10916972160339,
                3.58363935613632,
                2.83766316795349,
                1.15910302543640,
            ]
        ),
    )


def test_load_xml_params():
    file = tests.frontend.modules.get_test_resources_directory("untangleworms/parameters.xml")
    with open(file, "r") as fd:
        data = fd.read()
    data = data.encode()
    workspace, module = make_workspace(
        numpy.zeros((10, 10), bool), data, write_mode="w"
    )
    assert isinstance(module, cellprofiler.modules.untangleworms.UntangleWorms)
    module.prepare_group(workspace, None, None)
    params = module.read_params()
    assert params.version == 10680
    assert round(abs(params.min_worm_area - 596.898), 7) == 0
    assert round(abs(params.max_area - 1183), 7) == 0
    assert round(abs(params.cost_threshold - 100), 7) == 0
    assert params.num_control_points == 21
    assert round(abs(params.max_skel_length - 158.038188951), 7) == 0
    assert round(abs(params.min_path_length - 76.7075694196), 7) == 0
    assert round(abs(params.max_path_length - 173.842007846), 7) == 0
    assert round(abs(params.median_worm_area - 1006.0), 7) == 0
    assert round(abs(params.max_radius - 5.07108224265), 7) == 0
    assert params.overlap_weight == 5
    assert params.leftover_weight == 10
    expected = numpy.array(
        [
            -0.010999071156,
            -0.011369928253,
            -0.0063572663907,
            -0.00537369691481,
            -0.00491960423727,
            -0.00888319810452,
            0.000958176954014,
            -0.00329354759164,
            -0.00182154382306,
            -0.00252850541515,
            0.00583731052007,
            -0.00629326705054,
            -0.000221502806058,
            0.000541997532415,
            7.10858314614e-05,
            -0.000536334650268,
            0.00846699296415,
            0.00229934152116,
            0.013315157629,
            127.475166584,
        ]
    )
    numpy.testing.assert_almost_equal(expected, params.mean_angles)
    expected = numpy.array(
        [
            1.42515381896,
            2.85816813675,
            3.625274645,
            4.0753094944,
            4.35612078737,
            4.52117778419,
            4.63350649815,
            4.68616131779,
            4.72754363339,
            4.7538931664,
            4.74523602328,
            4.73738576257,
            4.71422245337,
            4.67997686977,
            4.63299502256,
            4.54769060478,
            4.37493539203,
            4.10516708918,
            3.6478380576,
            2.87519387935,
            1.41510654664,
        ]
    )
    numpy.testing.assert_almost_equal(expected, params.radii_from_training)
    expected = numpy.array(
        [
            [
                14.4160767437,
                -3.62948521476,
                -5.54563467306,
                -2.03727846881,
                1.28497735663,
                1.69152924813,
                0.155723103489,
                0.0968570278119,
                -0.830941512378,
                -0.212837030458,
                0.343854099462,
                0.541856348331,
                0.0113083798588,
                -0.526926108341,
                0.499035732802,
                -0.714281934407,
                0.407365302927,
                0.0412386837587,
                0.399436023924,
                0.0100796538444,
            ],
            [
                -3.62948521476,
                28.3822272923,
                -5.69796350599,
                -12.8940771298,
                -3.24074903822,
                6.60066559736,
                4.7027587735,
                -0.228375351756,
                -0.93975886278,
                -0.404133419773,
                0.461061138309,
                -0.620175812393,
                0.426458145116,
                0.242860102398,
                -1.395472436,
                -0.416345646399,
                -0.744042489471,
                1.2607667493,
                0.0266160618793,
                0.0125632227269,
            ],
            [
                -5.54563467306,
                -5.69796350599,
                33.965523631,
                -2.5184071008,
                -11.9739358904,
                -5.55211501206,
                1.54963332447,
                1.51853522595,
                2.94238214784,
                1.58963458609,
                -1.09690230963,
                -3.03348214943,
                -0.499685304962,
                0.0832942797245,
                2.64055890507,
                1.47285696771,
                -0.837591498704,
                -1.32897383529,
                -0.0909573472139,
                -0.00747385975607,
            ],
            [
                -2.03727846881,
                -12.8940771298,
                -2.5184071008,
                38.910896265,
                -0.501289252952,
                -16.9651461757,
                -8.5206491768,
                0.599043361039,
                4.0425703374,
                3.46803482473,
                1.5750544267,
                0.828548143764,
                -0.468487536336,
                1.18597661827,
                0.798189409259,
                0.862661371456,
                -0.14246505168,
                -1.3841655628,
                0.236725547536,
                -0.0163619841363,
            ],
            [
                1.28497735663,
                -3.24074903822,
                -11.9739358904,
                -0.501289252952,
                50.690805239,
                -2.50730714299,
                -19.7892493084,
                -8.70870454399,
                0.544130042179,
                3.96597772445,
                4.72570280412,
                2.85472249374,
                -2.84084610444,
                -4.25238476559,
                -1.41656624794,
                2.35443215524,
                1.26857012684,
                -0.0117059492945,
                0.193085174503,
                -0.0128948234117,
            ],
            [
                1.69152924813,
                6.60066559736,
                -5.55211501206,
                -16.9651461757,
                -2.50730714299,
                49.7744788862,
                2.21292071874,
                -18.9379120677,
                -11.6040186403,
                -0.650553765753,
                3.61258885051,
                2.19212295163,
                -0.250008762357,
                -2.6870585839,
                -1.82043770689,
                0.109737469347,
                0.560938864345,
                0.219973313189,
                -0.590133590346,
                -0.00967309489576,
            ],
            [
                0.155723103489,
                4.7027587735,
                1.54963332447,
                -8.5206491768,
                -19.7892493084,
                2.21292071874,
                51.0843827257,
                0.934339159637,
                -16.1389407803,
                -13.597573567,
                -1.54861446638,
                3.38169064096,
                4.06874131702,
                -0.827598209967,
                0.403883540178,
                -2.06739129329,
                -2.88536104019,
                1.06444751058,
                0.425819183522,
                0.0218279774716,
            ],
            [
                0.0968570278119,
                -0.228375351756,
                1.51853522595,
                0.599043361039,
                -8.70870454399,
                -18.9379120677,
                0.934339159637,
                53.1553410056,
                3.83284330941,
                -21.8128767252,
                -12.4239738428,
                -0.689818407647,
                4.93635164952,
                4.04304737017,
                1.11729234765,
                -0.61148362918,
                -1.50162558801,
                -1.61722109339,
                0.491305564623,
                0.00813673085389,
            ],
            [
                -0.830941512378,
                -0.93975886278,
                2.94238214784,
                4.0425703374,
                0.544130042179,
                -11.6040186403,
                -16.1389407803,
                3.83284330941,
                47.6933352778,
                1.02465850736,
                -18.704856196,
                -9.30970873094,
                1.7845053387,
                2.83710840227,
                3.85412837972,
                -0.420823216821,
                1.56974432254,
                -0.212411753395,
                -0.638990283092,
                -0.00117994378546,
            ],
            [
                -0.212837030458,
                -0.404133419773,
                1.58963458609,
                3.46803482473,
                3.96597772445,
                -0.650553765753,
                -13.597573567,
                -21.8128767252,
                1.02465850736,
                55.2260388503,
                4.35803059752,
                -17.4350070993,
                -9.9394563068,
                0.592362874638,
                4.03037893175,
                0.749631051365,
                0.179619159884,
                1.09520337409,
                0.198303530561,
                -0.0128674812863,
            ],
            [
                0.343854099462,
                0.461061138309,
                -1.09690230963,
                1.5750544267,
                4.72570280412,
                3.61258885051,
                -1.54861446638,
                -12.4239738428,
                -18.704856196,
                4.35803059752,
                51.04055356,
                3.08936728044,
                -17.5902587966,
                -10.8714973146,
                -0.045009571053,
                4.87264332876,
                1.30470158026,
                -0.320349338202,
                0.55323063623,
                -0.00361862544014,
            ],
            [
                0.541856348331,
                -0.620175812393,
                -3.03348214943,
                0.828548143764,
                2.85472249374,
                2.19212295163,
                3.38169064096,
                -0.689818407647,
                -9.30970873094,
                -17.4350070993,
                3.08936728044,
                47.4661853593,
                -1.87723439855,
                -15.2700084763,
                -7.6273108814,
                4.14811581054,
                1.42240471385,
                0.0505728359147,
                -0.0106613679324,
                -0.000211505765068,
            ],
            [
                0.0113083798588,
                0.426458145116,
                -0.499685304962,
                -0.468487536336,
                -2.84084610444,
                -0.250008762357,
                4.06874131702,
                4.93635164952,
                1.7845053387,
                -9.9394563068,
                -17.5902587966,
                -1.87723439855,
                47.8240218251,
                2.57791664619,
                -14.5709240372,
                -4.78007676552,
                1.87167780269,
                0.359928009336,
                -1.18561757081,
                0.014074799611,
            ],
            [
                -0.526926108341,
                0.242860102398,
                0.0832942797245,
                1.18597661827,
                -4.25238476559,
                -2.6870585839,
                -0.827598209967,
                4.04304737017,
                2.83710840227,
                0.592362874638,
                -10.8714973146,
                -15.2700084763,
                2.57791664619,
                46.696159054,
                -4.74906899066,
                -15.6278488145,
                -4.24795289841,
                2.87455853452,
                3.07635737509,
                0.00532906905096,
            ],
            [
                0.499035732802,
                -1.395472436,
                2.64055890507,
                0.798189409259,
                -1.41656624794,
                -1.82043770689,
                0.403883540178,
                1.11729234765,
                3.85412837972,
                4.03037893175,
                -0.045009571053,
                -7.6273108814,
                -14.5709240372,
                -4.74906899066,
                40.1689374127,
                -3.67980915371,
                -10.3797521361,
                -0.841069955948,
                3.27779133415,
                0.0045492369767,
            ],
            [
                -0.714281934407,
                -0.416345646399,
                1.47285696771,
                0.862661371456,
                2.35443215524,
                0.109737469347,
                -2.06739129329,
                -0.61148362918,
                -0.420823216821,
                0.749631051365,
                4.87264332876,
                4.14811581054,
                -4.78007676552,
                -15.6278488145,
                -3.67980915371,
                37.0559195085,
                -1.42897044519,
                -7.88598395567,
                -1.9964210551,
                -0.0047454750271,
            ],
            [
                0.407365302927,
                -0.744042489471,
                -0.837591498704,
                -0.14246505168,
                1.26857012684,
                0.560938864345,
                -2.88536104019,
                -1.50162558801,
                1.56974432254,
                0.179619159884,
                1.30470158026,
                1.42240471385,
                1.87167780269,
                -4.24795289841,
                -10.3797521361,
                -1.42897044519,
                31.0878364175,
                -3.67706594057,
                -7.74307062767,
                -0.0186367239616,
            ],
            [
                0.0412386837587,
                1.2607667493,
                -1.32897383529,
                -1.3841655628,
                -0.0117059492945,
                0.219973313189,
                1.06444751058,
                -1.61722109339,
                -0.212411753395,
                1.09520337409,
                -0.320349338202,
                0.0505728359147,
                0.359928009336,
                2.87455853452,
                -0.841069955948,
                -7.88598395567,
                -3.67706594057,
                17.6561215842,
                -0.53359878584,
                0.00910717515256,
            ],
            [
                0.399436023924,
                0.0266160618793,
                -0.0909573472139,
                0.236725547536,
                0.193085174503,
                -0.590133590346,
                0.425819183522,
                0.491305564623,
                -0.638990283092,
                0.198303530561,
                0.55323063623,
                -0.0106613679324,
                -1.18561757081,
                3.07635737509,
                3.27779133415,
                -1.9964210551,
                -7.74307062767,
                -0.53359878584,
                13.2416621872,
                0.00936896857131,
            ],
            [
                0.0100796538444,
                0.0125632227269,
                -0.00747385975607,
                -0.0163619841363,
                -0.0128948234117,
                -0.00967309489576,
                0.0218279774715,
                0.00813673085389,
                -0.00117994378545,
                -0.0128674812862,
                -0.00361862544014,
                -0.000211505765068,
                0.014074799611,
                0.00532906905096,
                0.0045492369767,
                -0.0047454750271,
                -0.0186367239616,
                0.00910717515256,
                0.00936896857131,
                0.00505367806039,
            ],
        ]
    )
    numpy.testing.assert_almost_equal(expected, params.inv_angles_covariance_matrix)


def test_trace_segments_none():
    """Test the trace_segments function on a blank image"""
    image = numpy.zeros((10, 20), bool)
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    i, j, label, order, distance, count = module.trace_segments(image)
    assert count == 0
    for x in (i, j, label, order, distance):
        assert len(x) == 0


def test_trace_one_segment():
    """Trace a single segment"""
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    image = numpy.zeros((10, 20), bool)
    image[5, 1:18] = True
    expected_order = numpy.zeros(image.shape, int)
    expected_order[image] = numpy.arange(numpy.sum(image))
    i, j, label, order, distance, count = module.trace_segments(image)
    assert count == 1
    assert numpy.all(label == 1)
    for x in (i, j, order, distance):
        assert len(x) == numpy.sum(image)

    result_order = numpy.zeros(image.shape, int)
    result_order[i, j] = order
    assert numpy.all(image[i, j])
    assert numpy.all(expected_order == result_order)


def test_trace_short_segment():
    """Trace a segment of a single point"""
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    image = numpy.zeros((10, 20), bool)
    for i in range(1, 3):
        image[5, 10 : (10 + i)] = True
        expected_order = numpy.zeros(image.shape, int)
        expected_order[image] = numpy.arange(numpy.sum(image))
        i, j, label, order, distance, count = module.trace_segments(image)
        assert count == 1
        assert numpy.all(label == 1)
        for x in (i, j, order, distance):
            assert len(x) == numpy.sum(image)

        result_order = numpy.zeros(image.shape, int)
        result_order[i, j] = order
        assert numpy.all(image[i, j])
        assert numpy.all(expected_order == result_order)


def test_trace_loop():
    """Trace an object that loops on itself"""
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    image = numpy.zeros((10, 20), bool)
    image[1:-1, 1:-1] = True
    image[2:-2, 2:-2] = False
    # Lop off the corners as would happen if skeletonized
    image[1, 1] = image[1, -2] = image[-2, 1] = image[-2, -2] = False
    #
    # It should go clockwise, starting from 1,2
    #
    expected_order = numpy.zeros(image.shape, int)
    i, j = numpy.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    slices = (
        (1, slice(2, -2)),
        (slice(2, -2), -2),
        (-2, slice(-3, 1, -1)),
        (slice(-3, 1, -1), 1),
    )
    ii, jj = numpy.array((2, 0), int)
    ii = numpy.hstack([i[islice, jslice].flatten() for islice, jslice in slices])
    jj = numpy.hstack([j[islice, jslice].flatten() for islice, jslice in slices])
    expected_order[ii, jj] = numpy.arange(len(ii))
    i, j, label, order, distance, count = module.trace_segments(image)
    result_order = numpy.zeros(image.shape, int)
    result_order[i, j] = order
    assert numpy.all(expected_order == result_order)


def test_trace_two():
    """Trace two objects"""
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    image = numpy.zeros((10, 20), bool)
    image[1:-1, 5] = True
    image[1:-1, 15] = True
    i, j, label, order, distance, count = module.trace_segments(image)
    assert count == 2
    result_order = numpy.zeros(image.shape, int)
    result_order[i, j] = order
    for j in (5, 15):
        assert numpy.all(result_order[1:-1, j] == numpy.arange(image.shape[0] - 2))


def test_make_incidence_matrix_of_nothing():
    """Make incidence matrix with two empty labels matrices"""

    module = cellprofiler.modules.untangleworms.UntangleWorms()
    result = module.make_incidence_matrix(
        numpy.zeros((10, 20), int), 0, numpy.zeros((10, 20), int), 0
    )
    assert tuple(result.shape) == (0, 0)


def test_make_incidence_matrix_of_things_that_do_not_touch():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    L1 = numpy.zeros((10, 20), int)
    L2 = numpy.zeros((10, 20), int)
    L1[5, 5] = 1
    L2[5, 15] = 1
    result = module.make_incidence_matrix(L1, 1, L2, 1)
    assert tuple(result.shape) == (1, 1)
    assert numpy.all(~result)


def test_make_incidence_matrix_of_things_that_touch():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    L1 = numpy.zeros((10, 20), int)
    L2 = numpy.zeros((10, 20), int)
    L1[5, 5] = 1
    for i2, j2 in ((4, 4), (4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5), (6, 6)):
        L2[i2, j2] = 1
        result = module.make_incidence_matrix(L1, 1, L2, 1)
        assert tuple(result.shape) == (1, 1)
        assert numpy.all(result)


def test_make_incidence_matrix_of_many_things():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    L1 = numpy.zeros((10, 20), int)
    L2 = numpy.zeros((10, 20), int)
    L1[2, 1:5] = 1
    L1[4, 6:10] = 2
    L1[6, 11:15] = 3
    L1[1:6, 16] = 4
    L1[0, 2:15] = 5
    L2[1, 1] = 1
    L2[3, 5] = 2
    L2[5, 10] = 3
    L2[6, 15] = 4
    L2[1, 15] = 5
    expected = numpy.zeros((5, 5), bool)
    expected[0, 0] = True
    expected[0, 1] = True
    expected[1, 1] = True
    expected[1, 2] = True
    expected[2, 2] = True
    expected[2, 3] = True
    expected[3, 3] = True
    expected[3, 4] = True
    expected[4, 4] = True
    expected[4, 0] = True
    result = module.make_incidence_matrix(L1, 5, L2, 5)
    assert numpy.all(result == expected)


def test_get_all_paths_recur_none():
    module = cellprofiler.modules.untangleworms.UntangleWorms()

    class Result(object):
        def __init__(self):
            self.branch_areas = []
            self.segments = []
            self.incidence_matrix = numpy.zeros((0, 0), bool)
            self.segment_lengths = []

    paths_list = list(module.get_all_paths_recur(Result(), [], [], 0, 0, 1000))
    assert len(paths_list) == 0


def test_get_all_paths_recur_one():
    module = cellprofiler.modules.untangleworms.UntangleWorms()

    #
    # Branch # 0 connects segment 0 and segment 1
    #
    class Result(object):
        def __init__(self):
            self.incident_branch_areas = [[0], [0]]
            self.incident_segments = [[0, 1]]
            self.segments = [numpy.zeros((2, 1)), numpy.zeros((2, 1))]
            self.segment_lengths = [1, 1]
            self.incidence_directions = numpy.array([[False, True]])

    paths_list = list(module.get_all_paths_recur(Result(), [0], [[0]], 1, 0, 1000))
    assert len(paths_list) == 1
    path = paths_list[0]
    assert isinstance(path, module.Path)
    assert tuple(path.segments) == (0, 1)
    assert tuple(path.branch_areas) == (0,)


def test_get_all_paths_recur_depth_two():
    module = cellprofiler.modules.untangleworms.UntangleWorms()

    #
    # Branch # 0 connects segment 0 and segment 1
    # Branch # 1 connects segment 1 and 2
    #
    class Result(object):
        def __init__(self):
            self.incident_branch_areas = [[0], [0, 1], [1]]
            self.incident_segments = [[0, 1], [1, 2]]
            self.segments = [numpy.zeros((2, 1)), numpy.zeros((2, 1))] * 3
            self.segment_lengths = [1, 1, 1]
            self.incidence_directions = numpy.array(
                [[False, True, False], [False, True, False]]
            )

    paths_list = list(module.get_all_paths_recur(Result(), [0], [[0]], 1, 0, 1000))
    assert len(paths_list) == 2
    expected = (((0, 1), (0,)), ((0, 1, 2), (0, 1)))
    sorted_list = tuple(
        sorted(
            [(tuple(path.segments), tuple(path.branch_areas)) for path in paths_list]
        )
    )
    assert sorted_list == expected


def test_get_all_paths_recur_many():
    module = cellprofiler.modules.untangleworms.UntangleWorms()

    #
    # A hopeless tangle where all branches connect to all segments
    #
    class Result(object):
        def __init__(self):
            self.incident_branch_areas = [list(range(3))] * 4
            self.incident_segments = [list(range(4))] * 3
            self.segments = [(numpy.zeros((2, 1)), numpy.zeros((2, 1)))] * 4
            self.segment_lengths = [1] * 4
            self.incidence_directions = numpy.ones((3, 4), bool)

    paths_list = module.get_all_paths_recur(
        Result(), [0], [[i] for i in range(3)], 1, 0, 1000
    )
    sorted_list = tuple(
        sorted(
            [(tuple(path.segments), tuple(path.branch_areas)) for path in paths_list]
        )
    )
    #
    # All possible permutations of 1,2,3 * all possible permutations
    # of 1,2,3
    #
    permutations = ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))
    #
    # Singles...
    #
    expected = sum(
        [
            [((0, segment), (branch_area,)) for branch_area in range(3)]
            for segment in range(1, 4)
        ],
        [],
    )
    #
    # Doubles
    #
    expected += sum(
        [
            sum(
                [
                    [
                        (tuple([0] + list(ps[:2])), (b1, b2))
                        for b1 in range(3)
                        if b2 != b1
                    ]
                    for b2 in range(3)
                ],
                [],
            )
            for ps in permutations
        ],
        [],
    )
    #
    # Triples
    #
    expected += sum(
        [
            sum(
                [
                    sum(
                        [
                            [
                                (tuple([0] + list(ps)), (b1, b2, b3))
                                for b1 in range(3)
                                if b2 != b1 and b1 != b3
                            ]
                            for b2 in range(3)
                            if b3 != b2
                        ],
                        [],
                    )
                    for b3 in range(3)
                ],
                [],
            )
            for ps in permutations
        ],
        [],
    )
    expected = tuple(sorted(expected))
    assert sorted_list == expected


def test_get_all_paths_none():
    module = cellprofiler.modules.untangleworms.UntangleWorms()

    class Result(object):
        def __init__(self):
            self.branch_areas = []
            self.segments = []
            self.incidence_matrix = numpy.zeros((0, 0), bool)

    path_list = list(module.get_all_paths(Result(), 0, 1000))
    assert len(path_list) == 0


def test_get_all_paths_one():
    module = cellprofiler.modules.untangleworms.UntangleWorms()

    class Result(object):
        def __init__(self):
            self.branch_areas = []
            self.segments = [[numpy.zeros((1, 2)), numpy.zeros((1, 2))]]
            self.incidence_matrix = numpy.zeros((0, 1), bool)
            self.incidence_directions = [[True, False]]

    path_list = list(module.get_all_paths(Result(), 0, 1000))
    assert len(path_list) == 1
    path = path_list[0]
    assert isinstance(path, module.Path)
    assert tuple(path.segments) == (0,)
    assert len(path.branch_areas) == 0


def test_get_all_paths_two_segments():
    module = cellprofiler.modules.untangleworms.UntangleWorms()

    class Result(object):
        def __init__(self):
            self.branch_areas = [1]
            self.segments = [[numpy.zeros((1, 2)), numpy.zeros((1, 2))]] * 2
            self.incidence_matrix = numpy.ones((1, 2), bool)
            self.incidence_directions = numpy.array([[True, False]])

    path_list = list(module.get_all_paths(Result(), 0, 1000))
    assert len(path_list) == 3
    sorted_list = tuple(
        sorted([(tuple(path.segments), tuple(path.branch_areas)) for path in path_list])
    )
    expected = (((0,), ()), ((0, 1), (0,)), ((1,), ()))
    assert sorted_list == expected


def test_get_all_paths_many():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    numpy.random.seed(63)

    class Result(object):
        def __init__(self):
            self.branch_areas = [0, 1, 2]
            self.segments = [[numpy.zeros((1, 2)), numpy.zeros((1, 2))]] * 4
            self.incidence_directions = numpy.random.uniform(size=(3, 4)) > 0.25
            self.incidence_matrix = self.incidence_directions | (
                numpy.random.uniform(size=(3, 4)) > 0.25
            )

    graph = Result()
    path_list = module.get_all_paths(graph, 0, 1000)
    for path in path_list:
        assert len(path.segments) == len(path.branch_areas) + 1
        if len(path.segments) > 1:
            assert path.segments[0] < path.segments[-1]
            for prev, next, branch_area in zip(
                path.segments[:-1], path.segments[1:], path.branch_areas
            ):
                assert graph.incidence_matrix[branch_area, prev]
                assert graph.incidence_matrix[branch_area, next]


def test_sample_control_points():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    path_coords = numpy.random.randint(0, 20, size=(11, 2))
    distances = numpy.linspace(0.0, 10.0, 11)
    result = module.sample_control_points(path_coords, distances, 6)
    assert len(result) == 6
    assert tuple(path_coords[0]) == tuple(result[0])
    assert tuple(path_coords[-1]) == tuple(result[-1])
    for i in range(1, 5):
        assert tuple(path_coords[i * 2]) == tuple(result[i])


def test_sample_non_linear_control_points():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    path_coords = numpy.array([numpy.arange(11)] * 2).transpose()
    distances = numpy.sqrt(numpy.arange(11))
    result = module.sample_control_points(path_coords, distances, 6)
    assert numpy.all(result[:, 0] >= numpy.linspace(0.0, 1.0, 6) ** 2 * 10)
    assert numpy.all(result[:, 0] < numpy.linspace(0.0, 1.0, 6) ** 2 * 10 + 0.5)


def test_only_two_sample_points():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    path_coords = numpy.array([[0, 0], [1, 2]])
    distances = numpy.array([0, 5])
    result = module.sample_control_points(path_coords, distances, 6)
    numpy.testing.assert_almost_equal(result[:, 0], numpy.linspace(0, 1, 6))
    numpy.testing.assert_almost_equal(result[:, 1], numpy.linspace(0, 2, 6))


def test_worm_descriptor_building_none():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    params = make_params(dict(worm_radius=5, num_control_points=20))
    result, _, _, _, _ = module.worm_descriptor_building([], params, (0, 0))
    assert len(result) == 0


def test_worm_descriptor_building_one():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    params = make_params(
        dict(radii_from_training=numpy.array([5, 5, 5]), num_control_points=3)
    )
    result, _, _, _, _ = module.worm_descriptor_building(
        [numpy.array([[10, 15], [20, 25]])], params, (40, 50)
    )
    expected = numpy.zeros((40, 50), bool)
    expected[numpy.arange(10, 21), numpy.arange(15, 26)] = True
    ii, jj = numpy.mgrid[-5:6, -5:6]
    expected = scipy.ndimage.binary_dilation(expected, ii * ii + jj * jj <= 25)
    expected = numpy.argwhere(expected)
    eorder = numpy.lexsort((expected[:, 0], expected[:, 1]))
    rorder = numpy.lexsort((result[:, 0], result[:, 1]))
    assert numpy.all(result[:, 2] == 1)
    assert len(expected) == len(result)
    assert numpy.all(result[rorder, :2] == expected[eorder, :])


def test_worm_descriptor_building_oob():
    """Test performance if part of the worm is out of bounds"""
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    params = make_params(
        dict(radii_from_training=numpy.array([5, 5, 5]), num_control_points=3)
    )
    result, _, _, _, _ = module.worm_descriptor_building(
        [numpy.array([[1, 15], [11, 25]])], params, (40, 27)
    )
    expected = numpy.zeros((40, 27), bool)
    expected[numpy.arange(1, 12), numpy.arange(15, 26)] = True
    ii, jj = numpy.mgrid[-5:6, -5:6]
    expected = scipy.ndimage.binary_dilation(expected, ii * ii + jj * jj <= 25)
    expected = numpy.argwhere(expected)
    eorder = numpy.lexsort((expected[:, 0], expected[:, 1]))
    rorder = numpy.lexsort((result[:, 0], result[:, 1]))
    assert numpy.all(result[:, 2] == 1)
    assert len(expected) == len(result)
    assert numpy.all(result[rorder, :2] == expected[eorder, :])


def test_worm_descriptor_building_two():
    """Test rebuilding two worms"""

    module = cellprofiler.modules.untangleworms.UntangleWorms()
    params = make_params(
        dict(radii_from_training=numpy.array([5, 5, 5]), num_control_points=3)
    )
    result, _, _, _, _ = module.worm_descriptor_building(
        [numpy.array([[10, 15], [20, 25]]), numpy.array([[10, 25], [20, 15]])],
        params,
        (40, 50),
    )
    expected = numpy.zeros((40, 50), bool)
    expected[numpy.arange(10, 21), numpy.arange(15, 26)] = True
    ii, jj = numpy.mgrid[-5:6, -5:6]
    expected = scipy.ndimage.binary_dilation(expected, ii * ii + jj * jj <= 25)
    epoints = numpy.argwhere(expected)
    elabels = numpy.ones(len(epoints), int)

    expected = numpy.zeros((40, 50), bool)
    expected[numpy.arange(10, 21), numpy.arange(25, 14, -1)] = True
    expected = scipy.ndimage.binary_dilation(expected, ii * ii + jj * jj <= 25)
    epoints = numpy.vstack((epoints, numpy.argwhere(expected)))
    elabels = numpy.hstack((elabels, numpy.ones(numpy.sum(expected), int) * 2))

    eorder = numpy.lexsort((epoints[:, 0], epoints[:, 1]))
    rorder = numpy.lexsort((result[:, 0], result[:, 1]))
    assert len(epoints) == len(result)
    assert numpy.all(result[rorder, 2] == elabels[eorder])
    assert numpy.all(result[rorder, :2] == epoints[eorder])


def test_fast_selection_two():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    costs = numpy.array([1, 1])
    path_segment_matrix = numpy.array([[True, False], [False, True]])
    segment_lengths = numpy.array([5, 5])
    best_paths, best_cost = module.fast_selection(
        costs, path_segment_matrix, segment_lengths, 1, 1, 10000
    )
    assert tuple(best_paths) == (0, 1)
    assert best_cost == 2


def test_fast_selection_overlap():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    costs = numpy.array([1, 1, 10])
    path_segment_matrix = numpy.array(
        [[True, False, True], [True, True, True], [False, True, True]]
    )
    segment_lengths = numpy.array([5, 3, 5])
    best_paths, best_cost = module.fast_selection(
        costs, path_segment_matrix, segment_lengths, 2, 5, 10000
    )
    assert tuple(best_paths) == (0, 1)
    assert best_cost == 2 + 3 * 2


def test_fast_selection_gap():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    costs = numpy.array([1, 1, 10])
    path_segment_matrix = numpy.array(
        [[True, False, True], [False, False, True], [False, True, True]]
    )
    segment_lengths = numpy.array([5, 3, 5])
    best_paths, best_cost = module.fast_selection(
        costs, path_segment_matrix, segment_lengths, 5, 2, 10000
    )
    assert tuple(best_paths) == (0, 1)
    assert best_cost == 2 + 3 * 2


def test_fast_selection_no_overlap():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    costs = numpy.array([1, 1, 7])
    path_segment_matrix = numpy.array(
        [[True, False, True], [True, True, True], [False, True, True]]
    )
    segment_lengths = numpy.array([5, 3, 5])
    best_paths, best_cost = module.fast_selection(
        costs, path_segment_matrix, segment_lengths, 2, 5, 10000
    )
    assert tuple(best_paths) == (2,)
    assert best_cost == 7


def test_fast_selection_no_gap():
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    costs = numpy.array([1, 1, 7])
    path_segment_matrix = numpy.array(
        [[True, False, True], [False, False, True], [False, True, True]]
    )
    segment_lengths = numpy.array([5, 3, 5])
    best_paths, best_cost = module.fast_selection(
        costs, path_segment_matrix, segment_lengths, 5, 2, 10000
    )
    assert tuple(best_paths) == (2,)
    assert best_cost == 7


def test_A02():
    params = zlib.decompress(base64.b64decode(PARAMS))
    workspace, module = make_workspace(A02_image, params)
    assert isinstance(module, cellprofiler.modules.untangleworms.UntangleWorms)
    module.prepare_group(workspace, None, None)
    module.wants_training_set_weights.value = False
    module.override_leftover_weight.value = 6
    module.override_overlap_weight.value = 3
    module.run(workspace)
    object_set = workspace.object_set
    assert isinstance(object_set, cellprofiler_core.object.ObjectSet)
    worms = object_set.get_objects(OVERLAP_OBJECTS_NAME)
    assert isinstance(worms, cellprofiler_core.object.Objects)
    worm_ijv = worms.get_ijv()
    assert numpy.max(worm_ijv[:, 2]) == 15
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    ocount = m.get_current_image_measurement("Count_" + OVERLAP_OBJECTS_NAME)
    assert ocount == 15
    ncount = m.get_current_image_measurement("Count_" + NON_OVERLAPPING_OBJECTS_NAME)
    assert ncount == 15
    columns = module.get_measurement_columns(workspace.pipeline)
    for column in columns:
        oname, feature = column[:2]
        v = m.get_current_measurement(oname, feature)


def test_nonoverlapping_outlines():
    params = zlib.decompress(base64.b64decode(PARAMS))
    workspace, module = make_workspace(A02_image, params)
    assert isinstance(module, cellprofiler.modules.untangleworms.UntangleWorms)
    module.prepare_group(workspace, None, None)
    module.wants_training_set_weights.value = False
    module.override_leftover_weight.value = 6
    module.override_overlap_weight.value = 3
    module.wants_nonoverlapping_outlines.value = True
    module.nonoverlapping_outlines_name.value = NON_OVERLAPPING_OUTLINES_NAME
    module.run(workspace)
    object_set = workspace.object_set
    assert isinstance(object_set, cellprofiler_core.object.ObjectSet)
    worms = object_set.get_objects(NON_OVERLAPPING_OBJECTS_NAME).segmented
    outlines = workspace.image_set.get_image(NON_OVERLAPPING_OUTLINES_NAME).pixel_data
    expected = centrosome.outline.outline(worms) > 0
    assert numpy.all(outlines == expected)


def test_overlapping_outlines():
    params = zlib.decompress(base64.b64decode(PARAMS))
    workspace, module = make_workspace(A02_image, params)
    assert isinstance(module, cellprofiler.modules.untangleworms.UntangleWorms)
    module.prepare_group(workspace, None, None)
    module.wants_training_set_weights.value = False
    module.override_leftover_weight.value = 6
    module.override_overlap_weight.value = 3
    module.wants_overlapping_outlines.value = True
    module.overlapping_outlines_name.value = OVERLAPPING_OUTLINES_NAME
    module.run(workspace)
    object_set = workspace.object_set
    assert isinstance(object_set, cellprofiler_core.object.ObjectSet)
    worms = object_set.get_objects(OVERLAP_OBJECTS_NAME)
    outlines = workspace.image_set.get_image(OVERLAPPING_OUTLINES_NAME).pixel_data
    outlines = numpy.sum(outlines, 2) > 0  # crunch color dimension
    i, j, v = worms.ijv.transpose()
    expected = numpy.zeros(outlines.shape, bool)
    expected[i, j] = True
    # all outlines are in some object...
    assert numpy.all(expected[outlines])


def test_train_dot():
    # Test training a single pixel
    # Regression test of bugs regarding this case
    #
    image = numpy.zeros((10, 20), bool)
    image[5, 10] = True
    workspace, module = make_workspace(image)
    assert isinstance(module, cellprofiler.modules.untangleworms.UntangleWorms)
    module.mode.value = cellprofiler.modules.untangleworms.MODE_TRAIN
    module.prepare_group(workspace, None, None)
    module.run(workspace)


def test_trace_segments():
    #
    # Regression test of img-1541, branch_areas_binary is not zero
    # but segments_binary is
    #
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    i, j, labels, segment_order, distances, num_segments = module.trace_segments(
        numpy.zeros((10, 13), bool)
    )
    assert len(i) == 0
    assert len(j) == 0
    numpy.testing.assert_equal(labels, 0)
    assert len(segment_order) == 0
    assert len(distances) == 0
    assert num_segments == 0


def test_get_graph_from_branching_areas_and_segments():
    #
    # Regression test of img-1541, branch_areas_binary is not zero
    # but segments_binary is
    #
    module = cellprofiler.modules.untangleworms.UntangleWorms()
    branch_areas = numpy.zeros((31, 15), bool)
    branch_areas[7:25, 7:10] = True
    result = module.get_graph_from_branching_areas_and_segments(
        branch_areas, numpy.zeros(branch_areas.shape, bool)
    )
    assert tuple(branch_areas.shape) == result.image_size
    assert len(result.segment_coords) == 0
    assert len(result.segment_counts) == 0
    assert len(result.segment_order) == 0
    assert len(result.segments) == 0


def test_recalculate_single_worm_control_points():
    i, j = numpy.mgrid[0:10, 0:10]
    l0 = ((i == 3) & (j >= 2) & (j <= 6)).astype(int)
    l0[(i == 7) & (j >= 3) & (j <= 7)] = 3

    l1 = ((j == 3) & (i >= 2) & (i <= 6)).astype(int) * 2
    l1[(j == 7) & (i >= 3) & (i <= 7)] = 4

    expected = numpy.array(
        (
            ((3, 2), (3, 4), (3, 6)),
            ((2, 3), (4, 3), (6, 3)),
            ((7, 3), (7, 5), (7, 7)),
            ((3, 7), (5, 7), (7, 7)),
        )
    )

    (
        result,
        lengths,
    ) = cellprofiler.modules.untangleworms.recalculate_single_worm_control_points(
        [l0, l1], 3
    )
    assert tuple(result.shape) == (4, 3, 2)
    assert len(lengths) == 4
    # Flip any worms that aren't ordered canonically
    for i in range(4):
        if tuple(result[i, -1, :]) < tuple(result[i, 0, :]):
            result[i, :, :] = result[i, ::-1, :]

    assert numpy.all(lengths == 4)
    numpy.testing.assert_array_equal(expected, result)


def test_recalculate_single_worm_control_points_no_objects():
    # regression test of issue #930
    (
        result,
        lengths,
    ) = cellprofiler.modules.untangleworms.recalculate_single_worm_control_points(
        [numpy.zeros((10, 15), int)], 3
    )
    assert tuple(result.shape) == (0, 3, 2)
    assert len(lengths) == 0
