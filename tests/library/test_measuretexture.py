from unittest.mock import patch

import mahotas
import numpy
import pytest
import skimage.exposure
import skimage.util
from cellprofiler_library.modules import _measuretexture as texture
from pydantic import ValidationError


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("gray_levels", [1, 0, -1, 257, 512])
def test_invalid_gray_levels(dimensions, gray_levels):
    pixels = numpy.linspace(0.1, 1, 4**dimensions).reshape((4,) * dimensions)
    labels = numpy.ones(pixels.shape, dtype=int)
    mask = numpy.ones(pixels.shape, dtype=bool)
    mask[0] = False
    originals = [array.copy() for array in (pixels, labels, mask)]

    with patch.object(
        texture,
        "measure_haralick_features_image",
        wraps=texture.measure_haralick_features_image,
    ) as kernel:
        with pytest.raises(ValidationError, match="gray_levels"):
            texture.measure_image_texture(
                pixels, gray_levels=gray_levels, scale=1, image_name="Image"
            )
        kernel.assert_not_called()

    for empty in (False, True):
        object_labels = numpy.zeros_like(labels) if empty else labels
        indices = numpy.array([], dtype=int) if empty else numpy.array([1])
        with patch.object(
            texture,
            "measure_haralick_features_objects",
            wraps=texture.measure_haralick_features_objects,
        ) as kernel:
            with pytest.raises(ValidationError, match="gray_levels"):
                texture.measure_object_texture(
                    "Objects",
                    object_labels,
                    "Image",
                    pixels.view(),
                    mask,
                    gray_levels=gray_levels,
                    unique_labels=indices,
                    scale=1,
                    volumetric=dimensions == 3,
                )
            kernel.assert_not_called()

    for array, original in zip((pixels, labels, mask), originals):
        numpy.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("gray_levels", [2, 8, 256])
def test_valid_gray_levels_match_haralick(dimensions, gray_levels):
    pixels = numpy.linspace(0.1, 1, 4**dimensions).reshape((4,) * dimensions)
    pixels[(slice(1, 3),) * dimensions] = 1
    original = pixels.copy()
    quantized = skimage.util.img_as_ubyte(pixels)
    if gray_levels != 256:
        quantized = skimage.exposure.rescale_intensity(
            quantized, in_range=(0, 255), out_range=(0, gray_levels - 1)
        ).astype(numpy.uint8)

    expected_image = mahotas.features.haralick(quantized, distance=1)
    expected_object = mahotas.features.haralick(
        quantized, distance=1, ignore_zeros=True
    )
    image_result = texture.measure_image_texture(pixels, gray_levels, 1, "Image")
    object_result = texture.measure_object_texture(
        "Objects",
        numpy.ones(pixels.shape, dtype=int),
        "Image",
        pixels,
        None,
        gray_levels,
        numpy.array([1]),
        1,
        dimensions == 3,
    )
    expected_names = set()
    for direction in range(13 if dimensions == 3 else 4):
        for index, feature in enumerate(texture.F_HARALICK):
            name = f"Texture_{feature.value}_Image_1_{direction:02d}_{gray_levels}"
            expected_names.add(name)
            numpy.testing.assert_equal(
                image_result.image[name], expected_image[direction, index]
            )
            numpy.testing.assert_array_equal(
                object_result.objects["Objects"][name],
                [expected_object[direction, index]],
            )
    assert set(image_result.image) == expected_names
    assert set(object_result.objects["Objects"]) == expected_names
    numpy.testing.assert_array_equal(pixels, original)
