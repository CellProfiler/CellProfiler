import mahotas
import numpy
import numpy.testing
import pytest
import scipy.ndimage
import skimage.color
import skimage.feature
import skimage.filters
import skimage.filters.rank
import skimage.measure
import skimage.segmentation
import skimage.transform
import skimage.util

import cellprofiler_core.image
import cellprofiler.modules.watershed

from cellprofiler.library.functions.object_processing import watershed

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


@pytest.fixture(scope="module", params=["Local", "Regional"], ids=["local", "regional"])
def maxima_method(request):
    return request.param


def test_run_distance_declump_intensity(
    image, module, image_set, workspace, connectivity, compactness, watershed_line, downsample, maxima_method
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

    expected = watershed(
        input_image, 
        method="distance", 
        declump_method="intensity",
        intensity_image=intensity_image,
        maxima_method=maxima_method,
        connectivity=connectivity,
        compactness=compactness,
        watershed_line=watershed_line,
        downsample=downsample,
        structuring_element=structuring_element,
        structuring_element_size=structuring_element_size
        )

    numpy.testing.assert_array_equal(actual.segmented, expected)


def test_run_distance_declump_shape(
    image, module, image_set, workspace, connectivity, compactness, watershed_line, downsample, maxima_method
):
    module.use_advanced.value = True

    module.watershed_method.value = "Distance"

    module.declump_method.value = "Shape"

    module.x_name.value = "input_image"

    module.y_name.value = "watershed"

    module.connectivity.value = connectivity

    module.compactness.value = compactness

    module.watershed_line.value = watershed_line

    module.downsample.value = downsample

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

    expected = watershed(
        input_image, 
        method="distance", 
        declump_method="shape",
        maxima_method=maxima_method,
        connectivity=connectivity,
        compactness=compactness,
        watershed_line=watershed_line,
        downsample=downsample,
        structuring_element=structuring_element,
        structuring_element_size=structuring_element_size
        )

    numpy.testing.assert_array_equal(actual.segmented, expected)


def test_run_distance_declump_none(
    image, module, image_set, workspace, connectivity, compactness, watershed_line, downsample, maxima_method
):
    module.use_advanced.value = True

    module.watershed_method.value = "Distance"

    module.declump_method.value = "None"

    module.x_name.value = "input_image"

    module.y_name.value = "watershed"

    module.connectivity.value = connectivity

    module.compactness.value = compactness

    module.watershed_line.value = watershed_line

    module.downsample.value = downsample

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

    expected = watershed(
        input_image, 
        method="distance", 
        declump_method="none",
        maxima_method=maxima_method,
        connectivity=connectivity,
        compactness=compactness,
        watershed_line=watershed_line,
        downsample=downsample,
        structuring_element=structuring_element,
        structuring_element_size=structuring_element_size
        )

    numpy.testing.assert_array_equal(actual.segmented, expected)


def test_run_markers_declump_shape(
    image, module, image_set, workspace, connectivity, compactness, watershed_line, downsample
):
    module.use_advanced.value = True

    module.watershed_method.value = "Markers"

    module.declump_method.value = "Shape"

    module.x_name.value = "input_image"

    module.y_name.value = "watershed"

    module.markers_name.value = "markers"

    module.connectivity.value = connectivity

    module.compactness.value = compactness

    module.watershed_line.value = watershed_line

    module.downsample.value = downsample

    if image.multichannel:
        image.pixel_data = skimage.color.rgb2gray(image.pixel_data)

    seed_coords = skimage.feature.peak_local_max(
        image.pixel_data,
        min_distance=1,
        threshold_rel=0,
        num_peaks=numpy.inf,
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

    expected = watershed(
        input_image, 
        method="markers", 
        declump_method="shape",
        markers_image=markers_image,
        connectivity=connectivity,
        compactness=compactness,
        watershed_line=watershed_line,
        downsample=downsample,
        structuring_element=structuring_element,
        structuring_element_size=structuring_element_size
        )

    numpy.testing.assert_array_equal(actual.segmented, expected)


def test_run_markers_declump_intensity(
    image, module, image_set, workspace, connectivity, compactness, watershed_line, downsample
):
    module.use_advanced.value = True

    module.watershed_method.value = "Markers"

    module.declump_method.value = "Intensity"

    module.x_name.value = "input_image"

    module.y_name.value = "watershed"

    module.markers_name.value = "markers"

    module.intensity_name.value = "intensity"

    module.connectivity.value = connectivity

    module.compactness.value = compactness

    module.watershed_line.value = watershed_line

    module.downsample.value = downsample

    if image.multichannel:
        image.pixel_data = skimage.color.rgb2gray(image.pixel_data)

    seed_coords = skimage.feature.peak_local_max(
        image.pixel_data,
        min_distance=1,
        threshold_rel=0,
        num_peaks=numpy.inf,
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

    expected = watershed(
        input_image, 
        method="markers", 
        declump_method="intensity",
        markers_image=markers_image,
        intensity_image=intensity_image,
        connectivity=connectivity,
        compactness=compactness,
        watershed_line=watershed_line,
        downsample=downsample,
        structuring_element=structuring_element,
        structuring_element_size=structuring_element_size
        )

    numpy.testing.assert_array_equal(actual.segmented, expected)


def test_run_markers_declump_none(
    image, module, image_set, workspace, connectivity, compactness, watershed_line, downsample
):
    module.use_advanced.value = True

    module.watershed_method.value = "Markers"

    module.declump_method.value = "None"

    module.x_name.value = "input_image"

    module.y_name.value = "watershed"

    module.markers_name.value = "markers"

    module.connectivity.value = connectivity

    module.compactness.value = compactness

    module.watershed_line.value = watershed_line

    module.downsample.value = downsample

    if image.multichannel:
        image.pixel_data = skimage.color.rgb2gray(image.pixel_data)

    seed_coords = skimage.feature.peak_local_max(
        image.pixel_data,
        min_distance=1,
        threshold_rel=0,
        num_peaks=numpy.inf,
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

    expected = watershed(
        input_image, 
        method="markers", 
        declump_method="none",
        markers_image=markers_image,
        connectivity=connectivity,
        compactness=compactness,
        watershed_line=watershed_line,
        downsample=downsample,
        structuring_element=structuring_element,
        structuring_element_size=structuring_element_size
        )

    numpy.testing.assert_array_equal(actual.segmented, expected)