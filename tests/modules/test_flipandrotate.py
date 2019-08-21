"""test_flipandrotate - test the FlipAndRotate module
"""

import base64
import zlib

import numpy as np
from six.moves import StringIO

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.pipeline as cpp
import cellprofiler.object as cpo
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.modules.flipandrotate as F
from centrosome.cpmorphology import draw_line

IMAGE_NAME = "my_image"
OUTPUT_IMAGE = "my_output_image"


def test_load_v2():
    """Load a v2 pipeline"""
    data = (
        "eJztWFtPGkEUXhCtl6bVpEn7OI/SAlmstkoaFaWmpIJEaBtjbDuyA0wyO0N2"
        "Z1VsTPrYn9af4M/oYx87gwu7TFeXi6QPZc0Gz9nzfecyZ5bDFLKV/ewOWEvp"
        "oJCtJGuYIFAikNeYZWYA5QmwayHIkQEYzYACo+CgyoG+BtJ6Jv06s/oKrOj6"
        "hjbcFckXHomP5hNNmxGfs+KOuo+mXTniu6VcRpxjWrentZj2zNVfi/sjtDA8"
        "JegjJA6yPRcdfZ7WWKXV7D4qMMMhqAhNv7G4io55iiz7oNYBuo9L+AKRMr5E"
        "Sgods0N0hm3MqIt3+VVt1y/jit9yg53vWSIchV/W53rWq09EqY+s15JPL+3f"
        "aZ59LKCefvtFV8bUwGfYcCAB2IT1bnSSTw/hm+rhm9JyxWwbtx2CW1TikHcF"
        "XfDk2wsomsyEvNroh+exwiPlPYKbTWQcWLi+IzqirzwiPTwR7aWbf5j/BcW/"
        "lHMMUMZBTYTRrWMYz7zCM+/xODbS+uaZU3ik/MFGwGRdmr7qEe3hiWpFNhou"
        "rC+fKnFLOYdq0CEc5GVTghy2UJUzq9VXHR4qfFLOd/uctILqMKPwdK4Oz9wA"
        "9VP3hZ7Qh8Idid0oceshuGklXynribSu9xnvbes2TJ0bzMKXjPLb6nyf/abu"
        "26B8h13XQXBhdQral7sNSCki6eQ9+B/2PT0uf+q6pMecX6zHX0z0D0Wj+Pse"
        "4u+91rueUv68vFV6IwcotJl6Ef8ipU+IkEN2vnmcTZZO4h3NLiOOSTeP9eTG"
        "ybd0YuXqxriMBbKtjAfmPUj8jZD415X4pSxjOELQcgNbvYonpUoMfrzh6lZc"
        "XQ62PM0ocf58MNh8M679GfT92x6G6hZzmuP3HzQPef6BGNFQ81+81ya4CW6C"
        "+39w2z7c5D01wQ2K++3Dqd/n6pwv7b9qd/fbc62336RcFSNU02LyvMpKme1D"
        "FTtFGDRuTi9S++LfvO8gox8/uuJHv82P/FUPqWExLua8lDxqyFLjsC2pdZsL"
        "8OPPPyr+lubvrrdaZ6/+v7aG8ReL/u1vIQQXcysmcT+0wdZ3+Q77Tm6j2A+a"
        "f0QIfwBNmhIA"
    )
    pipeline = cpp.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cpp.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
    assert len(pipeline.modules()) == 2
    module = pipeline.modules()[1]
    assert isinstance(module, F.FlipAndRotate)
    assert module.image_name == "DNA"
    assert module.output_name == "FlippedOrigBlue"
    assert module.flip_choice == F.FLIP_NONE
    assert module.rotate_choice == F.ROTATE_MOUSE
    assert not module.wants_crop.value
    assert module.how_often == F.IO_INDIVIDUALLY
    assert module.angle == 0
    assert module.first_pixel.x == 0
    assert module.first_pixel.y == 0
    assert module.second_pixel.x == 0
    assert module.second_pixel.y == 100
    assert module.horiz_or_vert == F.C_HORIZONTALLY


def run_module(image, mask=None, fn=None):
    """Run the FlipAndRotate module

    image - pixel data to be transformed
    mask  - optional mask on the pixel data
    fn    - function with signature, "fn(module)" that will be
            called with the FlipAndRotate module
    returns an Image object containing the flipped/rotated/masked/cropped
    image and the angle measurement.
    """
    img = cpi.Image(image, mask)
    image_set_list = cpi.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add(IMAGE_NAME, img)
    module = F.FlipAndRotate()
    module.image_name.value = IMAGE_NAME
    module.output_name.value = OUTPUT_IMAGE
    module.set_module_num(1)
    if fn is not None:
        fn(module)
    pipeline = cpp.Pipeline()
    pipeline.add_module(module)

    def error_callback(caller, event):
        assert not isinstance(event, cpp.RunExceptionEvent)

    pipeline.add_listener(error_callback)
    measurements = cpmeas.Measurements()
    workspace = cpw.Workspace(
        pipeline, module, image_set, cpo.ObjectSet(), measurements, image_set_list
    )
    module.run(workspace)
    feature = F.M_ROTATION_F % OUTPUT_IMAGE
    assert feature in measurements.get_feature_names(cpmeas.IMAGE)
    angle = measurements.get_current_image_measurement(feature)
    output_image = image_set.get_image(OUTPUT_IMAGE)
    return output_image, angle


def test_flip_left_to_right():
    np.random.seed(0)
    image = np.random.uniform(size=(3, 3))
    mask = np.array([[True, True, True], [False, True, True], [True, False, True]])
    expected_mask = np.array(
        [[True, True, True], [True, True, False], [True, False, True]]
    )
    expected = image.copy()
    expected[:, 2] = image[:, 0]
    expected[:, 0] = image[:, 2]

    def fn(module):
        assert isinstance(module, F.FlipAndRotate)
        module.flip_choice.value = F.FLIP_LEFT_TO_RIGHT
        module.rotate_choice.value = F.ROTATE_NONE

    output_image, angle = run_module(image, mask=mask, fn=fn)
    assert angle == 0
    assert np.all(output_image.mask == expected_mask)
    assert np.all(
        np.abs(output_image.pixel_data - expected) <= np.finfo(np.float32).eps
    )


def test_flip_top_to_bottom():
    np.random.seed(0)
    image = np.random.uniform(size=(3, 3)).astype(np.float32)
    mask = np.array([[True, True, True], [False, True, True], [True, False, True]])
    expected_mask = np.array(
        [[True, False, True], [False, True, True], [True, True, True]]
    )
    expected = image.copy()
    expected[2, :] = image[0, :]
    expected[0, :] = image[2, :]

    def fn(module):
        assert isinstance(module, F.FlipAndRotate)
        module.flip_choice.value = F.FLIP_TOP_TO_BOTTOM
        module.rotate_choice.value = F.ROTATE_NONE

    output_image, angle = run_module(image, mask=mask, fn=fn)
    assert angle == 0
    assert np.all(output_image.mask == expected_mask)
    assert np.all(np.abs(output_image.pixel_data - expected) <= np.finfo(float).eps)


def test_flip_both():
    np.random.seed(0)
    image = np.random.uniform(size=(3, 3)).astype(np.float32)
    mask = np.array([[True, True, True], [False, True, True], [True, False, True]])
    expected_mask = np.array(
        [[True, False, True], [True, True, False], [True, True, True]]
    )
    expected = image[
        np.array([[2, 2, 2], [1, 1, 1], [0, 0, 0]]),
        np.array([[2, 1, 0], [2, 1, 0], [2, 1, 0]]),
    ]

    def fn(module):
        assert isinstance(module, F.FlipAndRotate)
        module.flip_choice.value = F.FLIP_BOTH
        module.rotate_choice.value = F.ROTATE_NONE

    output_image, angle = run_module(image, mask=mask, fn=fn)
    assert angle == 0
    assert np.all(output_image.mask == expected_mask)
    assert np.all(np.abs(output_image.pixel_data - expected) <= np.finfo(float).eps)


def test_rotate_angle():
    """Rotate an image through an angle"""
    #
    # Draw a rectangle with intensity that varies monotonically according
    # to angle.
    #
    i, j = np.mgrid[-5:6, -9:10]
    angle = np.arctan2(i.astype(float) / 5.0, j.astype(float) / 9.0)
    img = (1 + np.cos(angle)) / 2
    assert round(abs(img[5, 0] - 0), 7) == 0
    assert round(abs(img[5, 18] - 1), 7) == 0
    assert round(abs(img[0, 9] - 0.5), 7) == 0
    assert round(abs(img[10, 9] - 0.5), 7) == 0
    #
    # The pixels with low values get masked out
    #
    mask = img > 0.5
    #
    # Rotate the rectangle from 10 to 350
    #
    for angle in range(10, 360, 10):

        def fn(module, angle=angle):
            assert isinstance(module, F.FlipAndRotate)
            module.flip_choice.value = F.FLIP_NONE
            module.rotate_choice.value = F.ROTATE_ANGLE
            module.wants_crop.value = False
            module.angle.value = angle

        output_image, measured_angle = run_module(img, mask, fn)
        assert round(abs(measured_angle - angle), 3) == 0
        rangle = float(angle) * np.pi / 180.0
        pixel_data = output_image.pixel_data
        #
        # Check that the output contains the four corners of the original
        #
        corners_in = np.array([[-5, -9], [-5, 9], [5, -9], [5, 9]], float)
        corners_out_i = np.sum(
            corners_in * np.array([np.cos(rangle), -np.sin(rangle)]), 1
        )
        corners_out_j = np.sum(
            corners_in * np.array([np.sin(rangle), np.cos(rangle)]), 1
        )
        i_width = np.max(corners_out_i) - np.min(corners_out_i)
        j_width = np.max(corners_out_j) - np.min(corners_out_j)
        assert i_width < pixel_data.shape[0]
        assert i_width > pixel_data.shape[0] - 2
        assert j_width < pixel_data.shape[1]
        assert j_width > pixel_data.shape[1] - 2
        # The maximum rotates clockwise - i starts at center and increases
        # and j starts at max and decreases
        #
        i_max = min(
            pixel_data.shape[0] - 1,
            max(0, int(-np.sin(rangle) * 8 + float(pixel_data.shape[0]) / 2)),
        )
        j_max = min(
            pixel_data.shape[1] - 1,
            max(0, int(np.cos(rangle) * 8 + float(pixel_data.shape[1] / 2))),
        )
        assert pixel_data[i_max, j_max] > 0.9
        assert output_image.mask[i_max, j_max]
        i_min = min(
            pixel_data.shape[0] - 1,
            max(0, int(np.sin(rangle) * 8 + float(pixel_data.shape[0]) / 2)),
        )
        j_min = min(
            pixel_data.shape[1] - 1,
            max(0, int(-np.cos(rangle) * 8 + float(pixel_data.shape[1]) / 2)),
        )
        assert pixel_data[i_min, j_min] < 0.1
        assert not output_image.mask[i_min, j_min]
        #
        # The corners of the image should be masked except for angle
        # in 90,180,270
        #
        if angle not in (90, 180, 270):
            for ci, cj in ((0, 0), (-1, 0), (-1, -1), (0, -1)):
                assert not output_image.mask[ci, cj]


def test_rotate_coordinates():
    """Test rotating a line to the horizontal and vertical"""

    img = np.zeros((20, 20))
    pt0 = (2, 2)
    pt1 = (6, 18)
    draw_line(img, pt0, pt1, 1)
    i, j = np.mgrid[0:20, 0:20]
    for option in (F.C_HORIZONTALLY, F.C_VERTICALLY):

        def fn(module):
            assert isinstance(module, F.FlipAndRotate)
            module.flip_choice.value = F.FLIP_NONE
            module.rotate_choice.value = F.ROTATE_COORDINATES
            module.horiz_or_vert.value = option
            module.wants_crop.value = False
            module.first_pixel.value = pt0
            module.second_pixel.value = pt1

        output_image, angle = run_module(img, fn=fn)
        pixels = output_image.pixel_data

        if option == F.C_HORIZONTALLY:
            assert (
                round(
                    abs(
                        angle
                        - np.arctan2(pt1[0] - pt0[0], pt1[1] - pt0[1]) * 180.0 / np.pi
                    ),
                    3,
                )
                == 0
            )
            #
            # Account for extra pixels due to twisting
            #
            line_i = 4 + (pixels.shape[0] - 20) / 2
            line_j = 4 + (pixels.shape[1] - 20) / 2
            assert np.all(pixels[line_i, line_j : line_j + 12] > 0.2)
            assert np.all(pixels[:20, :20][np.abs(i - line_i) > 1] < 0.1)
        else:
            assert (
                round(
                    abs(
                        angle
                        - -np.arctan2(pt1[1] - pt0[1], pt1[0] - pt0[0]) * 180.0 / np.pi
                    ),
                    3,
                )
                == 0
            )
            line_i = 4 + (pixels.shape[0] - 20) / 2
            line_j = 15 + (pixels.shape[1] - 20) / 2
            assert np.all(pixels[line_i : line_i + 12, line_j] > 0.2)
            assert np.all(pixels[:20, :20][np.abs(j - line_j) > 1] < 0.1)


def test_crop():
    """Turn cropping on and check that the cropping mask covers the mask"""
    image = np.random.uniform(size=(19, 21))
    i, j = np.mgrid[0:19, 0:21].astype(float)
    image = i / 100 + j / 10000
    for angle in range(10, 360, 10):
        #
        # Run the module with cropping to get the crop mask
        #
        def fn(module, angle=angle):
            assert isinstance(module, F.FlipAndRotate)
            module.flip_choice.value = F.FLIP_NONE
            module.rotate_choice.value = F.ROTATE_ANGLE
            module.angle.value = angle
            module.wants_crop.value = True

        crop_output_image, angle = run_module(image, fn=fn)
        crop_mask = crop_output_image.crop_mask
        crop_image = crop_output_image.pixel_data
        assert np.all(crop_output_image.mask[1:-1, 1:-1])

        #
        # Run the module without cropping to get the mask
        #
        def fn(module, angle=angle):
            assert isinstance(module, F.FlipAndRotate)
            module.flip_choice.value = F.FLIP_NONE
            module.rotate_choice.value = F.ROTATE_ANGLE
            module.angle.value = angle
            module.wants_crop.value = False

        output_image, angle = run_module(image, fn=fn)
        assert isinstance(crop_output_image, cpi.Image)
        pixel_data = output_image.pixel_data
        slop = (np.array(pixel_data.shape) - np.array(image.shape)) / 2
        mask = output_image.mask
        pixel_data = pixel_data[
            slop[0] : image.shape[0] + slop[0], slop[1] : image.shape[1] + slop[1]
        ]
        mask = mask[
            slop[0] : image.shape[0] + slop[0], slop[1] : image.shape[1] + slop[1]
        ]
        #
        # Slight misregistration: rotate returns even # shape
        #
        # recrop_image = crop_output_image.crop_image_similarly(pixel_data)
        # assertTrue(np.all(recrop_image == crop_image))
        # assertTrue(np.all(crop_output_image.crop_image_similarly(mask)))


def test_get_measurements():
    """Test the get_measurements and allied methods"""
    module = F.FlipAndRotate()
    module.output_name.value = OUTPUT_IMAGE
    columns = module.get_measurement_columns(None)
    assert len(columns) == 1
    assert columns[0][0] == cpmeas.IMAGE
    assert columns[0][1] == F.M_ROTATION_F % OUTPUT_IMAGE
    assert columns[0][2] == cpmeas.COLTYPE_FLOAT

    categories = module.get_categories(None, cpmeas.IMAGE)
    assert len(categories) == 1
    assert categories[0] == F.M_ROTATION_CATEGORY
    assert len(module.get_categories(None, "Foo")) == 0

    measurements = module.get_measurements(None, cpmeas.IMAGE, F.M_ROTATION_CATEGORY)
    assert len(measurements) == 1
    assert measurements[0] == OUTPUT_IMAGE
    assert len(module.get_measurements(None, cpmeas.IMAGE, "Foo")) == 0
    assert len(module.get_measurements(None, "Foo", F.M_ROTATION_CATEGORY)) == 0
