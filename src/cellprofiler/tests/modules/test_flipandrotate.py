import centrosome.cpmorphology
import numpy

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT


import cellprofiler.modules.flipandrotate
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

IMAGE_NAME = "my_image"
OUTPUT_IMAGE = "my_output_image"


def run_module(image, mask=None, fn=None):
    """Run the FlipAndRotate module

    image - pixel data to be transformed
    mask  - optional mask on the pixel data
    fn    - function with signature, "fn(module)" that will be
            called with the FlipAndRotate module
    returns an Image object containing the flipped/rotated/masked/cropped
    image and the angle measurement.
    """
    img = cellprofiler_core.image.Image(image, mask)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add(IMAGE_NAME, img)
    module = cellprofiler.modules.flipandrotate.FlipAndRotate()
    module.image_name.value = IMAGE_NAME
    module.output_name.value = OUTPUT_IMAGE
    module.set_module_num(1)
    if fn is not None:
        fn(module)
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)

    def error_callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(error_callback)
    measurements = cellprofiler_core.measurement.Measurements()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        measurements,
        image_set_list,
    )
    module.run(workspace)
    feature = cellprofiler.modules.flipandrotate.M_ROTATION_F % OUTPUT_IMAGE
    assert feature in measurements.get_feature_names(
        "Image"
    )
    angle = measurements.get_current_image_measurement(feature)
    output_image = image_set.get_image(OUTPUT_IMAGE)
    return output_image, angle


def test_flip_left_to_right():
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(3, 3))
    mask = numpy.array([[True, True, True], [False, True, True], [True, False, True]])
    expected_mask = numpy.array(
        [[True, True, True], [True, True, False], [True, False, True]]
    )
    expected = image.copy()
    expected[:, 2] = image[:, 0]
    expected[:, 0] = image[:, 2]

    def fn(module):
        assert isinstance(module, cellprofiler.modules.flipandrotate.FlipAndRotate)
        module.flip_choice.value = cellprofiler.modules.flipandrotate.FLIP_LEFT_TO_RIGHT
        module.rotate_choice.value = cellprofiler.modules.flipandrotate.ROTATE_NONE

    output_image, angle = run_module(image, mask=mask, fn=fn)
    assert angle == 0
    assert numpy.all(output_image.mask == expected_mask)
    assert numpy.all(
        numpy.abs(output_image.pixel_data - expected) <= numpy.finfo(numpy.float32).eps
    )


def test_flip_top_to_bottom():
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(3, 3)).astype(numpy.float32)
    mask = numpy.array([[True, True, True], [False, True, True], [True, False, True]])
    expected_mask = numpy.array(
        [[True, False, True], [False, True, True], [True, True, True]]
    )
    expected = image.copy()
    expected[2, :] = image[0, :]
    expected[0, :] = image[2, :]

    def fn(module):
        assert isinstance(module, cellprofiler.modules.flipandrotate.FlipAndRotate)
        module.flip_choice.value = cellprofiler.modules.flipandrotate.FLIP_TOP_TO_BOTTOM
        module.rotate_choice.value = cellprofiler.modules.flipandrotate.ROTATE_NONE

    output_image, angle = run_module(image, mask=mask, fn=fn)
    assert angle == 0
    assert numpy.all(output_image.mask == expected_mask)
    assert numpy.all(
        numpy.abs(output_image.pixel_data - expected) <= numpy.finfo(float).eps
    )


def test_flip_both():
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(3, 3)).astype(numpy.float32)
    mask = numpy.array([[True, True, True], [False, True, True], [True, False, True]])
    expected_mask = numpy.array(
        [[True, False, True], [True, True, False], [True, True, True]]
    )
    expected = image[
        numpy.array([[2, 2, 2], [1, 1, 1], [0, 0, 0]]),
        numpy.array([[2, 1, 0], [2, 1, 0], [2, 1, 0]]),
    ]

    def fn(module):
        assert isinstance(module, cellprofiler.modules.flipandrotate.FlipAndRotate)
        module.flip_choice.value = cellprofiler.modules.flipandrotate.FLIP_BOTH
        module.rotate_choice.value = cellprofiler.modules.flipandrotate.ROTATE_NONE

    output_image, angle = run_module(image, mask=mask, fn=fn)
    assert angle == 0
    assert numpy.all(output_image.mask == expected_mask)
    assert numpy.all(
        numpy.abs(output_image.pixel_data - expected) <= numpy.finfo(float).eps
    )


def test_rotate_angle():
    """Rotate an image through an angle"""
    #
    # Draw a rectangle with intensity that varies monotonically according
    # to angle.
    #
    i, j = numpy.mgrid[-5:6, -9:10]
    angle = numpy.arctan2(i.astype(float) / 5.0, j.astype(float) / 9.0)
    img = (1 + numpy.cos(angle)) / 2
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
            assert isinstance(module, cellprofiler.modules.flipandrotate.FlipAndRotate)
            module.flip_choice.value = cellprofiler.modules.flipandrotate.FLIP_NONE
            module.rotate_choice.value = cellprofiler.modules.flipandrotate.ROTATE_ANGLE
            module.wants_crop.value = False
            module.angle.value = angle

        output_image, measured_angle = run_module(img, mask, fn)
        assert round(abs(measured_angle - angle), 3) == 0
        rangle = float(angle) * numpy.pi / 180.0
        pixel_data = output_image.pixel_data
        #
        # Check that the output contains the four corners of the original
        #
        corners_in = numpy.array([[-5, -9], [-5, 9], [5, -9], [5, 9]], float)
        corners_out_i = numpy.sum(
            corners_in * numpy.array([numpy.cos(rangle), -numpy.sin(rangle)]), 1
        )
        corners_out_j = numpy.sum(
            corners_in * numpy.array([numpy.sin(rangle), numpy.cos(rangle)]), 1
        )
        i_width = numpy.max(corners_out_i) - numpy.min(corners_out_i)
        j_width = numpy.max(corners_out_j) - numpy.min(corners_out_j)
        assert i_width < pixel_data.shape[0]
        assert i_width > pixel_data.shape[0] - 2
        assert j_width < pixel_data.shape[1]
        assert j_width > pixel_data.shape[1] - 2
        # The maximum rotates clockwise - i starts at center and increases
        # and j starts at max and decreases
        #
        i_max = min(
            pixel_data.shape[0] - 1,
            max(0, int(-numpy.sin(rangle) * 8 + float(pixel_data.shape[0]) / 2)),
        )
        j_max = min(
            pixel_data.shape[1] - 1,
            max(0, int(numpy.cos(rangle) * 8 + float(pixel_data.shape[1] / 2))),
        )
        assert pixel_data[i_max, j_max] > 0.9
        assert output_image.mask[i_max, j_max]
        i_min = min(
            pixel_data.shape[0] - 1,
            max(0, int(numpy.sin(rangle) * 8 + float(pixel_data.shape[0]) / 2)),
        )
        j_min = min(
            pixel_data.shape[1] - 1,
            max(0, int(-numpy.cos(rangle) * 8 + float(pixel_data.shape[1]) / 2)),
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

    img = numpy.zeros((20, 20))
    pt0 = (2, 2)
    pt1 = (6, 18)
    centrosome.cpmorphology.draw_line(img, pt0, pt1, 1)
    i, j = numpy.mgrid[0:20, 0:20]
    for option in (
        cellprofiler.modules.flipandrotate.C_HORIZONTALLY,
        cellprofiler.modules.flipandrotate.C_VERTICALLY,
    ):

        def fn(module):
            assert isinstance(module, cellprofiler.modules.flipandrotate.FlipAndRotate)
            module.flip_choice.value = cellprofiler.modules.flipandrotate.FLIP_NONE
            module.rotate_choice.value = (
                cellprofiler.modules.flipandrotate.ROTATE_COORDINATES
            )
            module.horiz_or_vert.value = option
            module.wants_crop.value = False
            module.first_pixel.value = pt0
            module.second_pixel.value = pt1

        output_image, angle = run_module(img, fn=fn)
        pixels = output_image.pixel_data

        if option == cellprofiler.modules.flipandrotate.C_HORIZONTALLY:
            assert (
                round(
                    abs(
                        angle
                        - numpy.arctan2(pt1[0] - pt0[0], pt1[1] - pt0[1])
                        * 180.0
                        / numpy.pi
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
            assert numpy.all(pixels[int(line_i), int(line_j) : int(line_j) + 12] > 0.2)
            assert numpy.all(pixels[:20, :20][numpy.abs(i - line_i) > 1] < 0.1)
        else:
            assert (
                round(
                    abs(
                        angle
                        - -numpy.arctan2(pt1[1] - pt0[1], pt1[0] - pt0[0])
                        * 180.0
                        / numpy.pi
                    ),
                    3,
                )
                == 0
            )
            line_i = 4 + (pixels.shape[0] - 20) / 2
            line_j = 15 + (pixels.shape[1] - 20) / 2
            assert numpy.all(pixels[int(line_i) : int(line_i) + 12, int(line_j)] > 0.2)
            assert numpy.all(pixels[:20, :20][numpy.abs(j - line_j) > 1] < 0.1)


def test_crop():
    """Turn cropping on and check that the cropping mask covers the mask"""
    image = numpy.random.uniform(size=(19, 21))
    i, j = numpy.mgrid[0:19, 0:21].astype(float)
    image = i / 100 + j / 10000
    for angle in range(10, 360, 10):
        #
        # Run the module with cropping to get the crop mask
        #
        def fn(module, angle=angle):
            assert isinstance(module, cellprofiler.modules.flipandrotate.FlipAndRotate)
            module.flip_choice.value = cellprofiler.modules.flipandrotate.FLIP_NONE
            module.rotate_choice.value = cellprofiler.modules.flipandrotate.ROTATE_ANGLE
            module.angle.value = angle
            module.wants_crop.value = True

        crop_output_image, angle = run_module(image, fn=fn)
        crop_mask = crop_output_image.crop_mask
        crop_image = crop_output_image.pixel_data
        assert numpy.all(crop_output_image.mask[1:-1, 1:-1])

        #
        # Run the module without cropping to get the mask
        #
        def fn(module, angle=angle):
            assert isinstance(module, cellprofiler.modules.flipandrotate.FlipAndRotate)
            module.flip_choice.value = cellprofiler.modules.flipandrotate.FLIP_NONE
            module.rotate_choice.value = cellprofiler.modules.flipandrotate.ROTATE_ANGLE
            module.angle.value = angle
            module.wants_crop.value = False

        output_image, angle = run_module(image, fn=fn)
        assert isinstance(crop_output_image, cellprofiler_core.image.Image)
        pixel_data = output_image.pixel_data
        slop = (numpy.array(pixel_data.shape) - numpy.array(image.shape)) / 2
        mask = output_image.mask
        # pixel_data = pixel_data[
        #     slop[0] : image.shape[0] + slop[0], slop[1] : image.shape[1] + slop[1]
        # ]
        # mask = mask[
        #     slop[0] : image.shape[0] + slop[0], slop[1] : image.shape[1] + slop[1]
        # ]
        #
        # Slight misregistration: rotate returns even # shape
        #
        # recrop_image = crop_output_image.crop_image_similarly(pixel_data)
        # assertTrue(np.all(recrop_image == crop_image))
        # assertTrue(np.all(crop_output_image.crop_image_similarly(mask)))


def test_get_measurements():
    """Test the get_measurements and allied methods"""
    module = cellprofiler.modules.flipandrotate.FlipAndRotate()
    module.output_name.value = OUTPUT_IMAGE
    columns = module.get_measurement_columns(None)
    assert len(columns) == 1
    assert columns[0][0] == "Image"
    assert (
        columns[0][1] == cellprofiler.modules.flipandrotate.M_ROTATION_F % OUTPUT_IMAGE
    )
    assert columns[0][2] == COLTYPE_FLOAT

    categories = module.get_categories(None, "Image")
    assert len(categories) == 1
    assert categories[0] == cellprofiler.modules.flipandrotate.M_ROTATION_CATEGORY
    assert len(module.get_categories(None, "Foo")) == 0

    measurements = module.get_measurements(
        None,
        "Image",
        cellprofiler.modules.flipandrotate.M_ROTATION_CATEGORY,
    )
    assert len(measurements) == 1
    assert measurements[0] == OUTPUT_IMAGE
    assert (
        len(module.get_measurements(None, "Image", "Foo"))
        == 0
    )
    assert (
        len(
            module.get_measurements(
                None, "Foo", cellprofiler.modules.flipandrotate.M_ROTATION_CATEGORY
            )
        )
        == 0
    )
