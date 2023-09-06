import mahotas
import numpy
import numpy.testing
import pytest
import scipy.ndimage
import skimage.color
import skimage.feature
import skimage.filters
import skimage.filters.rank
import skimage.segmentation
import skimage.transform

import cellprofiler_core.image
import cellprofiler.modules.watershed

instance = cellprofiler.modules.watershed.Watershed()

# Override conftest images with bioimages
@pytest.fixture(
    scope="module",
    params=[
        (skimage.data.human_mitosis()[0:128, 0:128], 2),
        (skimage.color.gray2rgb(skimage.data.human_mitosis()[0:128, 0:128]), 2),
        (skimage.data.cells3d()[30:33, 1, 0:64, 0:64], 3),
    ],
    ids=["grayscale_image", "multichannel_image", "grayscale_volume"],
)
def image(request):
    data, dimensions = request.param

    return cellprofiler_core.image.Image(image=data, dimensions=dimensions)


@pytest.fixture(scope="module", params=[1, 2], ids=["1connectivity", "2connectivity"])
def connectivity(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[0.0, 1.0, 2.0],
    ids=["0compactness", "1compactness", "2compactness"],
)
def compactness(request):
    return request.param


@pytest.fixture(scope="module", params=[False, True], ids=["noline", "yesline"])
def watershed_line(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 4], ids=["downsample1", "downsample4"])
def downsample(request):
    return request.param


@pytest.fixture(scope="module", params=[0.0, 1.0], ids=["blur0", "blur1"])
def gaussian_sigma(request):
    return request.param


@pytest.fixture(scope="module", params=["Local", "Regional"], ids=["local", "regional"])
def maxima_method(request):
    return request.param


def test_run_distance_declump_intensity(
    image,
    module,
    image_set,
    workspace,
    connectivity,
    compactness,
    watershed_line,
    downsample,
    gaussian_sigma,
    maxima_method,
):
    module.use_advanced.value = True

    module.watershed_method.value = "Distance"

    module.declump_method.value = "Intensity"

    module.x_name.value = "input_image"

    module.y_name.value = "watershed"

    module.intensity_name.value = "intensity"

    module.connectivity.value = connectivity

    module.compactness.value = compactness

    module.watershed_line.value = watershed_line

    module.downsample.value = downsample

    module.gaussian_sigma.value = gaussian_sigma

    module.seed_method.value = maxima_method

    if image.dimensions == 3:
        module.structuring_element.value = "Ball,1"
        structuring_element = "Ball"
        structuring_element_size = 1
    else:
        module.structuring_element.value = "Disk,1"
        structuring_element = "Disk"
        structuring_element_size = 1

    if image.multichannel:
        image.pixel_data = skimage.color.rgb2gray(image.pixel_data)

    # Create an intensity image that is similar to the input
    intensity_image = skimage.filters.rank.median(image.pixel_data)

    image_set.add(
        "intensity",
        cellprofiler_core.image.Image(
            image=intensity_image, convert=False, dimensions=image.dimensions
        ),
    )

    # Watershed requires a thresholded input
    input_image = image.pixel_data > skimage.filters.threshold_otsu(image.pixel_data)

    image_set.add(
        "input_image",
        cellprofiler_core.image.Image(
            image=input_image, convert=False, dimensions=image.dimensions
        ),
    )

    module.run(workspace)

    actual = workspace.get_objects("watershed")

    # Generate expected output
    input_shape = input_image.shape
    if input_image.ndim > 2:
        # Only scale x and y
        factors = (1, downsample, downsample)
    else:
        factors = (downsample, downsample)

    footprint = 8

    if input_image.ndim == 3:
        maxima_footprint = numpy.ones((footprint, footprint, footprint))
    else:
        maxima_footprint = numpy.ones((footprint, footprint))

    input_image = skimage.transform.downscale_local_mean(input_image, factors)
    intensity_image = skimage.transform.downscale_local_mean(intensity_image, factors)
    smoothed_input_image = skimage.filters.gaussian(input_image, sigma=gaussian_sigma)
    distance = scipy.ndimage.distance_transform_edt(smoothed_input_image)
    watershed_input_image = 1 - intensity_image
    strel = getattr(skimage.morphology, structuring_element.casefold())(
        structuring_element_size
    )
    if maxima_method.casefold() == "local":
        seed_coords = skimage.feature.peak_local_max(
            distance,
            min_distance=1,
            threshold_rel=0,
            footprint=maxima_footprint,
            num_peaks=numpy.inf,
            exclude_border=False
        )
        seeds = numpy.zeros(distance.shape, dtype=bool)
        seeds[tuple(seed_coords.T)] = True
        seeds = skimage.morphology.binary_dilation(seeds, strel)
        seeds = scipy.ndimage.label(seeds)[0]

    elif maxima_method.casefold() == "regional":
        seeds = mahotas.regmax(distance, maxima_footprint)
        seeds = skimage.morphology.binary_dilation(seeds, strel)
        seeds = scipy.ndimage.label(seeds)[0]

    expected = skimage.segmentation.watershed(
        watershed_input_image,
        markers=seeds,
        mask=input_image != 0,
        connectivity=connectivity,
        compactness=compactness,
        watershed_line=watershed_line,
    )
    if downsample > 1:
        expected = skimage.transform.resize(
            expected, input_shape, mode="edge", order=0, preserve_range=True
        )
        expected = numpy.rint(expected).astype(numpy.uint16)

    numpy.testing.assert_array_equal(actual.segmented, expected)


def test_run_distance_declump_shape(
    image, module, image_set, workspace, downsample, gaussian_sigma, maxima_method
):
    module.use_advanced.value = True

    module.watershed_method.value = "Distance"

    module.declump_method.value = "Shape"

    module.x_name.value = "input_image"

    module.y_name.value = "watershed"

    module.downsample.value = downsample

    module.gaussian_sigma.value = gaussian_sigma

    module.seed_method.value = maxima_method

    if image.dimensions == 3:
        module.structuring_element.value = "Ball,1"
        structuring_element = "Ball"
        structuring_element_size = 1
    else:
        module.structuring_element.value = "Disk,1"
        structuring_element = "Disk"
        structuring_element_size = 1

    if image.multichannel:
        image.pixel_data = skimage.color.rgb2gray(image.pixel_data)

    # Watershed requires a thresholded input
    input_image = image.pixel_data > skimage.filters.threshold_otsu(image.pixel_data)

    image_set.add(
        "input_image",
        cellprofiler_core.image.Image(
            image=input_image, convert=False, dimensions=image.dimensions
        ),
    )

    module.run(workspace)

    actual = workspace.get_objects("watershed")

    # Generate expected output
    input_shape = input_image.shape
    if input_image.ndim > 2:
        # Only scale x and y
        factors = (1, downsample, downsample)
    else:
        factors = (downsample, downsample)

    footprint = 8

    if input_image.ndim == 3:
        maxima_footprint = numpy.ones((footprint, footprint, footprint))
    else:
        maxima_footprint = numpy.ones((footprint, footprint))

    input_image = skimage.transform.downscale_local_mean(input_image, factors)
    smoothed_input_image = skimage.filters.gaussian(input_image, sigma=gaussian_sigma)
    distance = scipy.ndimage.distance_transform_edt(smoothed_input_image)
    watershed_input_image = -distance
    strel = getattr(skimage.morphology, structuring_element.casefold())(
        structuring_element_size
    )
    if maxima_method.casefold() == "local":
        seed_coords = skimage.feature.peak_local_max(
            distance,
            min_distance=1,
            threshold_rel=0,
            footprint=maxima_footprint,
            num_peaks=numpy.inf,
            exclude_border=False
        )
        seeds = numpy.zeros(distance.shape, dtype=bool)
        seeds[tuple(seed_coords.T)] = True
        seeds = skimage.morphology.binary_dilation(seeds, strel)
        seeds = scipy.ndimage.label(seeds)[0]

    elif maxima_method.casefold() == "regional":
        seeds = mahotas.regmax(distance, maxima_footprint)
        seeds = skimage.morphology.binary_dilation(seeds, strel)
        seeds = scipy.ndimage.label(seeds)[0]

    expected = skimage.segmentation.watershed(
        watershed_input_image,
        markers=seeds,
        mask=input_image != 0,
    )
    if downsample > 1:
        expected = skimage.transform.resize(
            expected, input_shape, mode="edge", order=0, preserve_range=True
        )
        expected = numpy.rint(expected).astype(numpy.uint16)

    numpy.testing.assert_array_equal(actual.segmented, expected)


def test_run_markers_declump_shape(
    image, module, image_set, workspace, downsample, gaussian_sigma
):
    module.use_advanced.value = True

    module.watershed_method.value = "Markers"

    module.declump_method.value = "Shape"

    module.x_name.value = "input_image"

    module.y_name.value = "watershed"

    module.markers_name.value = "markers"

    module.downsample.value = downsample

    module.gaussian_sigma.value = gaussian_sigma

    if image.multichannel:
        image.pixel_data = skimage.color.rgb2gray(image.pixel_data)

    seed_coords = skimage.feature.peak_local_max(
        image.pixel_data,
        min_distance=1,
        threshold_rel=0,
        num_peaks=numpy.inf,
        exclude_border=False
    )
    markers_image = numpy.zeros(image.pixel_data.shape, dtype=bool)
    markers_image[tuple(seed_coords.T)] = True
    markers_image = scipy.ndimage.label(markers_image)[0]

    if image.dimensions == 3:
        module.structuring_element.value = "Ball,1"
        structuring_element = "Ball"
        structuring_element_size = 1
    else:
        module.structuring_element.value = "Disk,1"
        structuring_element = "Disk"
        structuring_element_size = 1

    image_set.add(
        "markers",
        cellprofiler_core.image.Image(
            image=markers_image, convert=False, dimensions=image.dimensions
        ),
    )

    # Watershed requires a thresholded input
    input_image = image.pixel_data > skimage.filters.threshold_otsu(image.pixel_data)

    image_set.add(
        "input_image",
        cellprofiler_core.image.Image(
            image=input_image, convert=False, dimensions=image.dimensions
        ),
    )

    module.run(workspace)

    actual = workspace.get_objects("watershed")

    # Generate expected output
    input_shape = input_image.shape
    if input_image.ndim > 2:
        # Only scale x and y
        factors = (1, downsample, downsample)
    else:
        factors = (downsample, downsample)

    footprint = 8

    if input_image.ndim == 3:
        maxima_footprint = numpy.ones((footprint, footprint, footprint))
    else:
        maxima_footprint = numpy.ones((footprint, footprint))

    input_image = skimage.transform.downscale_local_mean(input_image, factors)
    markers_image = skimage.transform.downscale_local_mean(markers_image, factors)
    smoothed_input_image = skimage.filters.gaussian(input_image, sigma=gaussian_sigma)
    distance = scipy.ndimage.distance_transform_edt(smoothed_input_image)
    watershed_input_image = -distance
    strel = getattr(skimage.morphology, structuring_element.casefold())(
        structuring_element_size
    )

    seeds = markers_image
    seeds = skimage.morphology.binary_dilation(seeds, strel)

    expected = skimage.segmentation.watershed(
        watershed_input_image,
        markers=seeds,
        mask=input_image != 0,
    )
    if downsample > 1:
        expected = skimage.transform.resize(
            expected, input_shape, mode="edge", order=0, preserve_range=True
        )
        expected = numpy.rint(expected).astype(numpy.uint16)

    numpy.testing.assert_array_equal(actual.segmented, expected)


def test_run_markers_declump_intensity(
    image, module, image_set, workspace, downsample, gaussian_sigma
):
    module.use_advanced.value = True

    module.watershed_method.value = "Markers"

    module.declump_method.value = "Intensity"

    module.x_name.value = "input_image"

    module.y_name.value = "watershed"

    module.markers_name.value = "markers"

    module.intensity_name.value = "intensity"

    module.downsample.value = downsample

    module.gaussian_sigma.value = gaussian_sigma

    if image.multichannel:
        image.pixel_data = skimage.color.rgb2gray(image.pixel_data)

    seed_coords = skimage.feature.peak_local_max(
        image.pixel_data,
        min_distance=1,
        threshold_rel=0,
        num_peaks=numpy.inf,
        exclude_border=False
    )
    markers_image = numpy.zeros(image.pixel_data.shape, dtype=bool)
    markers_image[tuple(seed_coords.T)] = True
    markers_image = scipy.ndimage.label(markers_image)[0]

    if image.dimensions == 3:
        module.structuring_element.value = "Ball,1"
        structuring_element = "Ball"
        structuring_element_size = 1
    else:
        module.structuring_element.value = "Disk,1"
        structuring_element = "Disk"
        structuring_element_size = 1

    image_set.add(
        "markers",
        cellprofiler_core.image.Image(
            image=markers_image, convert=False, dimensions=image.dimensions
        ),
    )

    # Create an intensity image that is similar to the input
    intensity_image = skimage.filters.rank.median(image.pixel_data)

    image_set.add(
        "intensity",
        cellprofiler_core.image.Image(
            image=intensity_image, convert=False, dimensions=image.dimensions
        ),
    )

    # Watershed requires a thresholded input
    input_image = image.pixel_data > skimage.filters.threshold_otsu(image.pixel_data)

    image_set.add(
        "input_image",
        cellprofiler_core.image.Image(
            image=input_image, convert=False, dimensions=image.dimensions
        ),
    )

    module.run(workspace)

    actual = workspace.get_objects("watershed")

    # Generate expected output
    input_shape = input_image.shape
    if input_image.ndim > 2:
        # Only scale x and y
        factors = (1, downsample, downsample)
    else:
        factors = (downsample, downsample)

    footprint = 8

    if input_image.ndim == 3:
        maxima_footprint = numpy.ones((footprint, footprint, footprint))
    else:
        maxima_footprint = numpy.ones((footprint, footprint))

    input_image = skimage.transform.downscale_local_mean(input_image, factors)
    markers_image = skimage.transform.downscale_local_mean(markers_image, factors)
    intensity_image = skimage.transform.downscale_local_mean(intensity_image, factors)
    smoothed_input_image = skimage.filters.gaussian(input_image, sigma=gaussian_sigma)
    distance = scipy.ndimage.distance_transform_edt(smoothed_input_image)
    watershed_input_image = 1 - intensity_image
    strel = getattr(skimage.morphology, structuring_element.casefold())(
        structuring_element_size
    )

    seeds = markers_image
    seeds = skimage.morphology.binary_dilation(seeds, strel)

    expected = skimage.segmentation.watershed(
        watershed_input_image,
        markers=seeds,
        mask=input_image != 0,
    )
    if downsample > 1:
        expected = skimage.transform.resize(
            expected, input_shape, mode="edge", order=0, preserve_range=True
        )
        expected = numpy.rint(expected).astype(numpy.uint16)

    numpy.testing.assert_array_equal(actual.segmented, expected)


def test_non_advanced(
    image,
    module,
    image_set,
    workspace,
):
    """Ensure that advanced settings are not passed to CellProfiler
    Libarary watershed"""

    module.use_advanced.value = False

    module.watershed_method.value = "Distance"

    module.declump_method.value = "Shape"

    module.x_name.value = "input_image"

    module.y_name.value = "watershed"

    # Change an advanced-only setting
    module.gaussian_sigma.value = 5

    if image.dimensions == 3:
        module.structuring_element.value = "Ball,1"
        structuring_element = "Ball"
        structuring_element_size = 1
    else:
        module.structuring_element.value = "Disk,1"
        structuring_element = "Disk"
        structuring_element_size = 1

    if image.multichannel:
        image.pixel_data = skimage.color.rgb2gray(image.pixel_data)

    # Watershed requires a thresholded input
    input_image = image.pixel_data > skimage.filters.threshold_otsu(image.pixel_data)

    image_set.add(
        "input_image",
        cellprofiler_core.image.Image(
            image=input_image, convert=False, dimensions=image.dimensions
        ),
    )

    module.run(workspace)

    actual = workspace.get_objects("watershed")

    if input_image.ndim == 3:
        maxima_footprint = numpy.ones((8, 8, 8))
    else:
        maxima_footprint = numpy.ones((8, 8))

    distance = scipy.ndimage.distance_transform_edt(input_image)
    watershed_input_image = -distance
    # watershed_input_image = watershed_input_image - watershed_input_image.min()
    strel = getattr(skimage.morphology, structuring_element.casefold())(
        structuring_element_size
    )

    seed_coords = skimage.feature.peak_local_max(
        distance,
        footprint=maxima_footprint,
        min_distance=1,
        threshold_rel=0,
        num_peaks=numpy.inf,
        exclude_border=False
    )
    seeds = numpy.zeros(image.pixel_data.shape, dtype=bool)
    seeds[tuple(seed_coords.T)] = True
    seeds = skimage.morphology.binary_dilation(seeds, strel)
    seeds = scipy.ndimage.label(seeds)[0]

    expected = skimage.segmentation.watershed(
        watershed_input_image,
        markers=seeds,
        mask=input_image != 0,
        connectivity=1,
        compactness=0.0,
        watershed_line=False,
    )

    numpy.testing.assert_array_equal(actual.segmented, expected)


def test_run_intensity_declump_shape(
    image,
    module,
    image_set,
    workspace,
    maxima_method,
):
    module.use_advanced.value = True

    module.watershed_method.value = "Intensity"

    module.declump_method.value = "Shape"

    module.x_name.value = "input_image"

    module.y_name.value = "watershed"

    module.intensity_name.value = "intensity"

    module.seed_method.value = maxima_method

    if image.dimensions == 3:
        module.structuring_element.value = "Ball,1"
        structuring_element = "Ball"
        structuring_element_size = 1
    else:
        module.structuring_element.value = "Disk,1"
        structuring_element = "Disk"
        structuring_element_size = 1

    if image.multichannel:
        image.pixel_data = skimage.color.rgb2gray(image.pixel_data)

    # Create an intensity image that is similar to the input
    intensity_image = skimage.filters.rank.median(image.pixel_data)

    image_set.add(
        "intensity",
        cellprofiler_core.image.Image(
            image=intensity_image, convert=False, dimensions=image.dimensions
        ),
    )

    # Watershed requires a thresholded input
    input_image = image.pixel_data > skimage.filters.threshold_otsu(image.pixel_data)

    image_set.add(
        "input_image",
        cellprofiler_core.image.Image(
            image=input_image, convert=False, dimensions=image.dimensions
        ),
    )

    module.run(workspace)

    actual = workspace.get_objects("watershed")

    # Generate expected output
    footprint = 8

    if input_image.ndim == 3:
        maxima_footprint = numpy.ones((footprint, footprint, footprint))
    else:
        maxima_footprint = numpy.ones((footprint, footprint))

    # Shape based declumping, so input image to watershed is the distance transform
    distance = scipy.ndimage.distance_transform_edt(input_image)
    watershed_input_image = 1 - distance

    strel = getattr(skimage.morphology, structuring_element.casefold())(
        structuring_element_size
    )
    if maxima_method.casefold() == "local":
        seed_coords = skimage.feature.peak_local_max(
            intensity_image,
            min_distance=1,
            threshold_rel=0,
            footprint=maxima_footprint,
            num_peaks=numpy.inf,
            exclude_border=False
        )
        seeds = numpy.zeros(intensity_image.shape, dtype=bool)
        seeds[tuple(seed_coords.T)] = True
        seeds = skimage.morphology.binary_dilation(seeds, strel)
        seeds = scipy.ndimage.label(seeds)[0]

    elif maxima_method.casefold() == "regional":
        seeds = mahotas.regmax(intensity_image, maxima_footprint)
        seeds = skimage.morphology.binary_dilation(seeds, strel)
        seeds = scipy.ndimage.label(seeds)[0]

    expected = skimage.segmentation.watershed(
        watershed_input_image,
        markers=seeds,
        mask=input_image != 0,
    )

    numpy.testing.assert_array_equal(actual.segmented, expected)


def test_declump_none(
    image,
    module,
    image_set,
    workspace,
):
    module.use_advanced.value = True

    module.declump_method.value = "None"

    module.x_name.value = "input_image"

    module.y_name.value = "watershed"

    if image.multichannel:
        image.pixel_data = skimage.color.rgb2gray(image.pixel_data)

    # Watershed requires a thresholded input
    input_image = image.pixel_data > skimage.filters.threshold_otsu(image.pixel_data)

    image_set.add(
        "input_image",
        cellprofiler_core.image.Image(
            image=input_image, convert=False, dimensions=image.dimensions
        ),
    )

    module.run(workspace)

    actual = workspace.get_objects("watershed")

    expected = scipy.ndimage.label(input_image)[0]

    numpy.testing.assert_array_equal(actual.segmented, expected)
