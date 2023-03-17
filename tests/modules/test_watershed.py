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

instance = cellprofiler.modules.watershed.Watershed()


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


@pytest.fixture(scope="module", params=[False, True], ids=["noborder", "yesborder"])
def watershed_line(request):
    return request.param


def test_run_markers(
    image, module, image_set, workspace, connectivity, compactness, watershed_line
):
    module.use_advanced.value = False

    module.operation.value = "Markers"

    module.x_name.value = "gradient"

    module.y_name.value = "watershed"

    module.markers_name.value = "markers"

    module.connectivity.value = connectivity

    module.compactness.value = compactness

    module.watershed_line.value = watershed_line

    if image.multichannel or image.dimensions == 3:
        denoised = numpy.zeros_like(image.pixel_data)

        for idx, data in enumerate(image.pixel_data):
            denoised[idx] = skimage.filters.rank.median(
                data, skimage.morphology.disk(2)
            )
    else:
        denoised = skimage.filters.rank.median(
            image.pixel_data, skimage.morphology.disk(2)
        )

    denoised = denoised.astype(numpy.uint8)

    if image.multichannel or image.dimensions == 3:
        markers = numpy.zeros_like(denoised)

        for idx, data in enumerate(denoised):
            markers[idx] = skimage.filters.rank.gradient(
                data, skimage.morphology.disk(5)
            )
    else:
        markers = skimage.filters.rank.median(denoised, skimage.morphology.disk(5))

    markers = skimage.measure.label(markers)

    image_set.add(
        "markers",
        cellprofiler_core.image.Image(
            image=markers, convert=False, dimensions=image.dimensions
        ),
    )

    if image.multichannel or image.dimensions == 3:
        gradient = numpy.zeros_like(denoised)

        for idx, data in enumerate(denoised):
            gradient[idx] = skimage.filters.rank.gradient(
                data, skimage.morphology.disk(2)
            )
    else:
        gradient = skimage.filters.rank.median(denoised, skimage.morphology.disk(2))

    image_set.add(
        "gradient",
        cellprofiler_core.image.Image(
            image=gradient, convert=False, dimensions=image.dimensions
        ),
    )

    module.run(workspace)

    if image.multichannel:
        gradient = skimage.color.rgb2gray(gradient)

        markers = skimage.color.rgb2gray(markers)

    expected = skimage.segmentation.watershed(
        gradient,
        markers,
        connectivity=connectivity,
        compactness=compactness,
        watershed_line=watershed_line,
    )

    expected = skimage.measure.label(expected)

    actual = workspace.get_objects("watershed")

    numpy.testing.assert_array_equal(expected, actual.segmented)


def test_run_distance(image, module, image_set, workspace):
    module.use_advanced.value = False

    module.operation.value = "Distance"

    module.x_name.value = "binary"

    module.y_name.value = "watershed"

    module.footprint.value = 3

    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    threshold = skimage.filters.threshold_otsu(data)

    binary = data > threshold

    image_set.add(
        "binary",
        cellprofiler_core.image.Image(
            image=binary, convert=False, dimensions=image.dimensions
        ),
    )

    module.run(workspace)

    original_shape = binary.shape

    distance = scipy.ndimage.distance_transform_edt(binary)

    distance = mahotas.stretch(distance)

    surface = distance.max() - distance

    if image.volumetric:
        mahotas_footprint = numpy.ones((3, 3, 3))
    else:
        mahotas_footprint = numpy.ones((3, 3))

    peaks = mahotas.regmax(distance, mahotas_footprint)

    if image.volumetric:
        markers, _ = mahotas.label(peaks, numpy.ones((16, 16, 16)))
    else:
        markers, _ = mahotas.label(peaks, numpy.ones((16, 16)))

    expected = mahotas.cwatershed(surface, markers)

    expected = expected * binary

    expected = skimage.measure.label(expected)

    actual = workspace.get_objects("watershed")

    actual = actual.segmented

    numpy.testing.assert_array_equal(expected, actual)

# test for marker-based watershed with shape-based declumping
def test_run_markers_declump_shape(
    image, module, image_set, workspace, connectivity, compactness, watershed_line
):
    module.use_advanced.value = True

    module.operation.value = "Markers"

    module.x_name.value = "gradient"

    module.y_name.value = "watershed"

    module.markers_name.value = "markers"

    module.connectivity.value = connectivity

    module.compactness.value = compactness

    module.watershed_line.value = watershed_line

    module.declump_method.value = "Shape"

    if image.dimensions == 3:
        module.structuring_element.value = "Ball,1"
        footprint = skimage.morphology.ball(1)

    else:
        module.structuring_element.value = "Disk,1"
        footprint = skimage.morphology.disk(1)

    if image.multichannel or image.dimensions == 3:
        denoised = numpy.zeros_like(image.pixel_data)

        for idx, data in enumerate(image.pixel_data):
            denoised[idx] = skimage.filters.rank.median(
                data, skimage.morphology.disk(2)
            )
    else:
        denoised = skimage.filters.rank.median(
            image.pixel_data, skimage.morphology.disk(2)
        )

    denoised = denoised.astype(numpy.uint8)

    if image.multichannel or image.dimensions == 3:
        markers = numpy.zeros_like(denoised)

        for idx, data in enumerate(denoised):
            markers[idx] = skimage.filters.rank.gradient(
                data, skimage.morphology.disk(5)
            )
    else:
        markers = skimage.filters.rank.median(denoised, skimage.morphology.disk(5))

    markers = skimage.measure.label(markers)

    image_set.add(
        "markers",
        cellprofiler_core.image.Image(
            image=markers, convert=False, dimensions=image.dimensions
        ),
    )

    if image.multichannel or image.dimensions == 3:
        gradient = numpy.zeros_like(denoised)

        for idx, data in enumerate(denoised):
            gradient[idx] = skimage.filters.rank.gradient(
                data, skimage.morphology.disk(2)
            )
    else:
        gradient = skimage.filters.rank.median(denoised, skimage.morphology.disk(2))

    image_set.add(
        "gradient",
        cellprofiler_core.image.Image(
            image=gradient, convert=False, dimensions=image.dimensions
        ),
    )

    module.run(workspace)

    if image.multichannel:
        gradient = skimage.color.rgb2gray(gradient)

        markers = skimage.color.rgb2gray(markers)

    watershed_markers = skimage.segmentation.watershed(
        image=gradient,
        markers=markers,
        connectivity=connectivity,
        compactness=compactness,
        watershed_line=watershed_line,
    )

    peak_image = scipy.ndimage.distance_transform_edt(watershed_markers > 0)

    watershed_image = -peak_image
    watershed_image -= watershed_image.min()

    watershed_image = skimage.filters.gaussian(watershed_image, sigma=module.gaussian_sigma.value)

    seed_coords = skimage.feature.peak_local_max(peak_image,
                                                 min_distance=module.min_dist.value,
                                                 threshold_rel=module.min_intensity.value,
                                                 exclude_border=module.exclude_border.value,
                                                 num_peaks=module.max_seeds.value if module.max_seeds.value != -1
                                                 else numpy.inf)

    seeds = numpy.zeros_like(peak_image, dtype=bool)
    seeds[tuple(seed_coords.T)] = True

    seeds = skimage.morphology.binary_dilation(seeds, footprint)

    number_objects = skimage.measure.label(watershed_markers, return_num=True)[1]

    seeds_dtype = (numpy.uint16 if number_objects < numpy.iinfo(numpy.uint16).max else numpy.uint32)

    seeds = scipy.ndimage.label(seeds)[0]
    markers = numpy.zeros_like(seeds, dtype=seeds_dtype)
    markers[seeds > 0] = -seeds[seeds > 0]

    expected = skimage.segmentation.watershed(
        connectivity=connectivity,
        image=watershed_image,
        markers=markers,
        mask=gradient !=0
    )

    zeros = numpy.where(expected==0)
    expected += numpy.abs(numpy.min(expected)) + 1
    expected[zeros] = 0

    expected = skimage.measure.label(expected)

    actual = workspace.get_objects("watershed")

    numpy.testing.assert_array_equal(expected, actual.segmented)

# test for distance-based watershed with shape-based declumping
def test_run_distance_declump_shape(
    image, module, image_set, workspace, connectivity, compactness, watershed_line
):
    module.use_advanced.value = True

    module.operation.value = "Distance"

    module.x_name.value = "binary"

    module.y_name.value = "watershed"

    module.declump_method.value = "Shape"

    module.connectivity.value = connectivity

    module.footprint.value = 3

    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    threshold = skimage.filters.threshold_otsu(data)

    binary = data > threshold

    image_set.add(
        "binary",
        cellprofiler_core.image.Image(
            image=binary, convert=False, dimensions=image.dimensions
        ),
    )

    # set the structuring element, used for shape-based declumping
    if image.dimensions == 3:
        module.structuring_element.value = "Ball,1"
        footprint = skimage.morphology.ball(1)

    else:
        module.structuring_element.value = "Disk,1"
        footprint = skimage.morphology.disk(1)

    # run the module
    module.run(workspace)

    # distance-based watershed
    distance = scipy.ndimage.distance_transform_edt(binary)

    distance = mahotas.stretch(distance)

    surface = distance.max() - distance

    if image.volumetric:
        mahotas_footprint = numpy.ones((3, 3, 3))
    else:
        mahotas_footprint = numpy.ones((3, 3))

    peaks = mahotas.regmax(distance, mahotas_footprint)

    if image.volumetric:
        markers, _ = mahotas.label(peaks, numpy.ones((16, 16, 16)))
    else:
        markers, _ = mahotas.label(peaks, numpy.ones((16, 16)))

    watershed_distance = mahotas.cwatershed(surface, markers)

    watershed_distance = watershed_distance * binary

    # shape-based declumping
    peak_image = scipy.ndimage.distance_transform_edt(watershed_distance > 0)

    watershed_image = -peak_image
    watershed_image -= watershed_image.min()

    watershed_image = skimage.filters.gaussian(watershed_image, sigma=module.gaussian_sigma.value)

    seed_coords = skimage.feature.peak_local_max(peak_image,
                                                 min_distance=module.min_dist.value,
                                                 threshold_rel=module.min_intensity.value,
                                                 exclude_border=module.exclude_border.value,
                                                 num_peaks=module.max_seeds.value if module.max_seeds.value != -1
                                                 else numpy.inf)

    seeds = numpy.zeros_like(peak_image, dtype=bool)
    seeds[tuple(seed_coords.T)] = True

    seeds = skimage.morphology.binary_dilation(seeds, footprint)

    number_objects = skimage.measure.label(watershed_distance, return_num=True)[1]

    seeds_dtype = (numpy.uint16 if number_objects < numpy.iinfo(numpy.uint16).max else numpy.uint32)

    seeds = scipy.ndimage.label(seeds)[0]
    markers = numpy.zeros_like(seeds, dtype=seeds_dtype)
    markers[seeds > 0] = -seeds[seeds > 0]

    expected = skimage.segmentation.watershed(
        connectivity=connectivity,
        image=watershed_image,
        markers=markers,
        mask=binary !=0
    )

    zeros = numpy.where(expected==0)
    expected += numpy.abs(numpy.min(expected)) + 1
    expected[zeros] = 0

    expected = skimage.measure.label(expected)

    actual = workspace.get_objects("watershed")

    numpy.testing.assert_array_equal(expected, actual.segmented)

# test for marker-based watershed with intensity-based declumping
def test_run_markers_declump_intensity(
    image, module, image_set, workspace, connectivity, compactness, watershed_line
):
    module.use_advanced.value = True

    module.operation.value = "Markers"

    module.x_name.value = "gradient"

    module.y_name.value = "watershed"

    module.markers_name.value = "markers"

    module.connectivity.value = connectivity

    module.compactness.value = compactness

    module.watershed_line.value = watershed_line

    module.declump_method.value = "Intensity"

    module.reference_name.value = "gradient"

    module.gaussian_sigma.value = 1

    if image.dimensions == 3:
        module.structuring_element.value = "Ball,1"
        footprint = skimage.morphology.ball(1)

    else:
        module.structuring_element.value = "Disk,1"
        footprint = skimage.morphology.disk(1)

    if image.multichannel or image.dimensions == 3:
        denoised = numpy.zeros_like(image.pixel_data)

        for idx, data in enumerate(image.pixel_data):
            denoised[idx] = skimage.filters.rank.median(
                data, skimage.morphology.disk(2)
            )
    else:
        denoised = skimage.filters.rank.median(
            image.pixel_data, skimage.morphology.disk(2)
        )

    denoised = denoised.astype(numpy.uint8)

    if image.multichannel or image.dimensions == 3:
        markers = numpy.zeros_like(denoised)

        for idx, data in enumerate(denoised):
            markers[idx] = skimage.filters.rank.gradient(
                data, skimage.morphology.disk(5)
            )
    else:
        markers = skimage.filters.rank.median(denoised, skimage.morphology.disk(5))

    markers = skimage.measure.label(markers)

    image_set.add(
        "markers",
        cellprofiler_core.image.Image(
            image=markers, convert=False, dimensions=image.dimensions
        ),
    )

    if image.multichannel or image.dimensions == 3:
        gradient = numpy.zeros_like(denoised)

        for idx, data in enumerate(denoised):
            gradient[idx] = skimage.filters.rank.gradient(
                data, skimage.morphology.disk(2)
            )
    else:
        gradient = skimage.filters.rank.median(denoised, skimage.morphology.disk(2))

    image_set.add(
        "gradient",
        cellprofiler_core.image.Image(
            image=gradient, convert=False, dimensions=image.dimensions
        ),
    )

    module.run(workspace)

    if image.multichannel:
        gradient = skimage.color.rgb2gray(gradient)

        markers = skimage.color.rgb2gray(markers)

    watershed_markers = skimage.segmentation.watershed(
        image=gradient,
        markers=markers,
        connectivity=connectivity,
        compactness=compactness,
        watershed_line=watershed_line,
    )

    peak_image = scipy.ndimage.distance_transform_edt(watershed_markers > 0)

    image_data = image.pixel_data

    # Set the image as a float and rescale to full bit depth
    watershed_image = skimage.img_as_float(image_data, force_copy=True)
    watershed_image -= watershed_image.min()
    watershed_image = 1 - watershed_image

    if image.multichannel:
        watershed_image = skimage.color.rgb2gray(watershed_image)

    watershed_image = skimage.filters.gaussian(watershed_image, sigma=module.gaussian_sigma.value)

    seed_coords = skimage.feature.peak_local_max(peak_image,
                                                 min_distance=module.min_dist.value,
                                                 threshold_rel=module.min_intensity.value,
                                                 exclude_border=module.exclude_border.value,
                                                 num_peaks=module.max_seeds.value if module.max_seeds.value != -1
                                                 else numpy.inf)

    seeds = numpy.zeros_like(peak_image, dtype=bool)
    seeds[tuple(seed_coords.T)] = True

    seeds = skimage.morphology.binary_dilation(seeds, footprint)

    number_objects = skimage.measure.label(watershed_markers, return_num=True)[1]

    seeds_dtype = (numpy.uint16 if number_objects < numpy.iinfo(numpy.uint16).max else numpy.uint32)

    seeds = scipy.ndimage.label(seeds)[0]
    markers = numpy.zeros_like(seeds, dtype=seeds_dtype)
    markers[seeds > 0] = -seeds[seeds > 0]

    expected = skimage.segmentation.watershed(
        connectivity=connectivity,
        image=watershed_image,
        markers=markers,
        mask=gradient != 0
    )

    zeros = numpy.where(expected == 0)
    expected += numpy.abs(numpy.min(expected)) + 1
    expected[zeros] = 0

    expected = skimage.measure.label(expected)

    actual = workspace.get_objects("watershed")

    numpy.testing.assert_array_equal(expected, actual.segmented)

# test for distance-based watershed with intensity-based declumping
def test_run_distance_declump_intensity(
    image, module, image_set, workspace, connectivity, compactness, watershed_line
):
    module.use_advanced.value = True

    module.operation.value = "Distance"

    module.x_name.value = "binary"

    module.y_name.value = "watershed"

    module.connectivity.value = connectivity

    module.footprint.value = 3

    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    threshold = skimage.filters.threshold_otsu(data)

    binary = data > threshold

    image_set.add(
        "binary",
        cellprofiler_core.image.Image(
            image=binary, convert=False, dimensions=image.dimensions
        ),
    )

    module.declump_method.value = "Intensity"

    module.reference_name.value = "gradient"

    module.gaussian_sigma.value = 1

    # must pass pixel data into image set for intensity declumping
    gradient = image.pixel_data

    image_set.add(
        "gradient",
        cellprofiler_core.image.Image(
            image=gradient, convert=False, dimensions=image.dimensions
        ),
    )

    # set the structuring element, used for declumping
    if image.dimensions == 3:
        module.structuring_element.value = "Ball,1"
        footprint = skimage.morphology.ball(1)

    else:
        module.structuring_element.value = "Disk,1"
        footprint = skimage.morphology.disk(1)


    # run the module
    module.run(workspace)

    # distance-based watershed
    distance = scipy.ndimage.distance_transform_edt(binary)

    distance = mahotas.stretch(distance)

    surface = distance.max() - distance

    if image.volumetric:
        mahotas_footprint = numpy.ones((3, 3, 3))
    else:
        mahotas_footprint = numpy.ones((3, 3))

    peaks = mahotas.regmax(distance, mahotas_footprint)

    if image.volumetric:
        markers, _ = mahotas.label(peaks, numpy.ones((16, 16, 16)))
    else:
        markers, _ = mahotas.label(peaks, numpy.ones((16, 16)))

    watershed_distance = mahotas.cwatershed(surface, markers)

    watershed_distance = watershed_distance * binary

    # intensity-based declumping
    peak_image = scipy.ndimage.distance_transform_edt(watershed_distance > 0)

    # Set the image as a float and rescale to full bit depth
    watershed_image = skimage.img_as_float(gradient, force_copy=True)
    watershed_image -= watershed_image.min()
    watershed_image = 1 - watershed_image

    if image.multichannel:
        watershed_image = skimage.color.rgb2gray(watershed_image)

    watershed_image = skimage.filters.gaussian(watershed_image, sigma=module.gaussian_sigma.value)

    seed_coords = skimage.feature.peak_local_max(peak_image,
                                                 min_distance=module.min_dist.value,
                                                 threshold_rel=module.min_intensity.value,
                                                 exclude_border=module.exclude_border.value,
                                                 num_peaks=module.max_seeds.value if module.max_seeds.value != -1
                                                 else numpy.inf)

    seeds = numpy.zeros_like(peak_image, dtype=bool)
    seeds[tuple(seed_coords.T)] = True

    seeds = skimage.morphology.binary_dilation(seeds, footprint)

    number_objects = skimage.measure.label(watershed_distance, return_num=True)[1]

    seeds_dtype = (numpy.uint16 if number_objects < numpy.iinfo(numpy.uint16).max else numpy.uint32)

    seeds = scipy.ndimage.label(seeds)[0]
    markers = numpy.zeros_like(seeds, dtype=seeds_dtype)
    markers[seeds > 0] = -seeds[seeds > 0]

    expected = skimage.segmentation.watershed(
        connectivity=connectivity,
        image=watershed_image,
        markers=markers,
        mask=binary !=0
    )

    zeros = numpy.where(expected==0)
    expected += numpy.abs(numpy.min(expected)) + 1
    expected[zeros] = 0

    expected = skimage.measure.label(expected)

    actual = workspace.get_objects("watershed")

    numpy.testing.assert_array_equal(expected, actual.segmented)