import cellprofiler.modules.watershed
import cellprofiler.image
import mahotas
import numpy
import numpy.testing
import scipy.ndimage
import skimage.color
import skimage.feature
import skimage.filters
import skimage.filters.rank
import skimage.measure
import skimage.morphology
import skimage.segmentation
import skimage.util

instance = cellprofiler.modules.watershed.Watershed()


def test_run_markers(image, module, image_set, workspace):
    module.operation.value = "Markers"

    module.x_name.value = "gradient"

    module.y_name.value = "watershed"

    module.markers_name.value = "markers"

    if image.multichannel or image.dimensions is 3:
        denoised = numpy.zeros_like(image.pixel_data)

        for idx, data in enumerate(image.pixel_data):
            denoised[idx] = skimage.filters.rank.median(data, skimage.morphology.disk(2))
    else:
        denoised = skimage.filters.rank.median(image.pixel_data, skimage.morphology.disk(2))

    denoised = denoised.astype(numpy.uint8)

    if image.multichannel or image.dimensions is 3:
        markers = numpy.zeros_like(denoised)

        for idx, data in enumerate(denoised):
            markers[idx] = skimage.filters.rank.gradient(data, skimage.morphology.disk(5))
    else:
        markers = skimage.filters.rank.median(denoised, skimage.morphology.disk(5))

    markers = skimage.measure.label(markers)

    image_set.add(
        "markers",
        cellprofiler.image.Image(
            image=markers,
            convert=False,
            dimensions=image.dimensions
        )
    )

    if image.multichannel or image.dimensions is 3:
        gradient = numpy.zeros_like(denoised)

        for idx, data in enumerate(denoised):
            gradient[idx] = skimage.filters.rank.gradient(data, skimage.morphology.disk(2))
    else:
        gradient = skimage.filters.rank.median(denoised, skimage.morphology.disk(2))

    image_set.add(
        "gradient",
        cellprofiler.image.Image(
            image=gradient,
            convert=False,
            dimensions=image.dimensions
        )
    )

    module.run(workspace)

    expected = skimage.morphology.watershed(gradient, markers)

    expected = skimage.measure.label(expected)

    actual = workspace.get_objects("watershed")

    numpy.testing.assert_array_equal(expected, actual.segmented)


def test_run_distance(image, module, image_set, workspace):
    module.operation.value = "Distance"

    module.x_name.value = "binary"

    module.y_name.value = "watershed"

    module.radius.value = 3

    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    threshold = skimage.filters.threshold_otsu(data)

    binary = data > threshold

    image_set.add(
        "binary",
        cellprofiler.image.Image(
            image=binary,
            convert=False,
            dimensions=image.dimensions
        )
    )

    module.run(workspace)

    distance = mahotas.distance(binary)

    distance = mahotas.stretch(distance)

    radius = 3

    if image.dimensions is 2:
        shape = (radius, radius)
    else:
        shape = (radius, radius, radius)

    footprint = numpy.ones(shape)

    peaks = mahotas.morph.regmax(distance, footprint)

    markers, count = mahotas.label(peaks, footprint)

    surface = distance.max() - distance

    expected = mahotas.cwatershed(surface, markers)

    expected *= binary

    expected = skimage.measure.label(expected)

    expected, _, _ = skimage.segmentation.relabel_sequential(expected)

    actual = workspace.get_objects("watershed")

    actual = actual.segmented

    actual, _, _ = skimage.segmentation.relabel_sequential(actual)

    numpy.testing.assert_array_equal(expected, actual)
