import matplotlib.cm
import numpy
import skimage.color

from ...preferences import get_default_colormap


def crop_labels_and_image(labels, image):
    """Crop a labels matrix and an image to the lowest common size

    labels - a n x m labels matrix
    image - a 2-d or 3-d image

    Assumes that points outside of the common boundary should be masked.
    """
    min_dim1 = min(labels.shape[0], image.shape[0])
    min_dim2 = min(labels.shape[1], image.shape[1])

    if labels.ndim == 3:  # volume
        min_dim3 = min(labels.shape[2], image.shape[2])

        if image.ndim == 4:  # multichannel volume
            return (
                labels[:min_dim1, :min_dim2, :min_dim3],
                image[:min_dim1, :min_dim2, :min_dim3, :],
            )

        return (
            labels[:min_dim1, :min_dim2, :min_dim3],
            image[:min_dim1, :min_dim2, :min_dim3],
        )

    if image.ndim == 3:  # multichannel image
        return labels[:min_dim1, :min_dim2], image[:min_dim1, :min_dim2, :]

    return labels[:min_dim1, :min_dim2], image[:min_dim1, :min_dim2]


def size_similarly(labels, secondary):
    """Size the secondary matrix similarly to the labels matrix

    labels - labels matrix
    secondary - a secondary image or labels matrix which might be of
                different size.
    Return the resized secondary matrix and a mask indicating what portion
    of the secondary matrix is bogus (manufactured values).

    Either the mask is all ones or the result is a copy, so you can
    modify the output within the unmasked region w/o destroying the original.
    """
    if labels.shape[:2] == secondary.shape[:2]:
        return secondary, numpy.ones(secondary.shape, bool)
    if labels.shape[0] <= secondary.shape[0] and labels.shape[1] <= secondary.shape[1]:
        if secondary.ndim == 2:
            return (
                secondary[: labels.shape[0], : labels.shape[1]],
                numpy.ones(labels.shape, bool),
            )
        else:
            return (
                secondary[: labels.shape[0], : labels.shape[1], :],
                numpy.ones(labels.shape, bool),
            )

    #
    # Some portion of the secondary matrix does not cover the labels
    #
    result = numpy.zeros(
        list(labels.shape) + list(secondary.shape[2:]), secondary.dtype
    )
    i_max = min(secondary.shape[0], labels.shape[0])
    j_max = min(secondary.shape[1], labels.shape[1])
    if secondary.ndim == 2:
        result[:i_max, :j_max] = secondary[:i_max, :j_max]
    else:
        result[:i_max, :j_max, :] = secondary[:i_max, :j_max, :]
    mask = numpy.zeros(labels.shape, bool)
    mask[:i_max, :j_max] = 1
    return result, mask


def overlay_labels(pixel_data, labels, opacity=0.7, max_label=None, seed=None):
    colors = _colors(labels, max_label=max_label, seed=seed)

    if labels.ndim == 3:
        overlay = numpy.zeros(labels.shape + (3,), dtype=numpy.float32)

        for index, plane in enumerate(pixel_data):
            unique_labels = numpy.unique(labels[index])

            if unique_labels[0] == 0:
                unique_labels = unique_labels[1:]

            overlay[index] = skimage.color.label2rgb(
                labels[index],
                alpha=opacity,
                bg_color=[0, 0, 0],
                bg_label=0,
                colors=colors[unique_labels - 1],
                image=plane,
            )

        return overlay

    return skimage.color.label2rgb(
        labels,
        alpha=opacity,
        bg_color=[0, 0, 0],
        bg_label=0,
        colors=colors,
        image=pixel_data,
    )


def _colors(labels, max_label=None, seed=None):
    mappable = matplotlib.cm.ScalarMappable(
        cmap=matplotlib.cm.get_cmap(get_default_colormap())
    )

    colors = mappable.to_rgba(
        numpy.arange(labels.max() if max_label is None else max_label)
    )[:, :3]

    if seed is not None:
        # Resetting the random seed helps keep object label colors consistent in displays
        # where consistency is important, like RelateObjects.
        numpy.random.seed(seed)

    numpy.random.shuffle(colors)

    return colors
