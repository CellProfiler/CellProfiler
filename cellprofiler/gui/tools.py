""" tools.py - cpfigure tools that do not depend on WX
"""

import io

import centrosome.cpmorphology
import imageio
import matplotlib
import matplotlib.figure
import matplotlib.transforms
import numpy


def figure_to_image(figure, *args, **kwargs):
    """Convert a figure to a numpy array"""
    #
    # Save the figure as a .PNG and then load it using imageio.imread
    #
    fd = io.BytesIO()
    kwargs = kwargs.copy()
    kwargs["format"] = "png"
    figure.savefig(fd, *args, **kwargs)
    image = imageio.imread(fd.getvalue())
    return image[:, :, :3]


def only_display_image(figure, shape):
    """Set up a figure so that the image occupies the entire figure

    figure - a matplotlib figure
    shape - i/j size of the image being displayed
    """
    assert isinstance(figure, matplotlib.figure.Figure)
    figure.set_frameon(False)
    ax = figure.axes[0]
    ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, 0, 0)
    dpi = figure.dpi
    width = float(shape[1]) / dpi
    height = float(shape[0]) / dpi
    figure.set_figheight(height)
    figure.set_figwidth(width)
    bbox = matplotlib.transforms.Bbox(numpy.array([[0.0, 0.0], [width, height]]))
    transform = matplotlib.transforms.Affine2D(
        numpy.array([[dpi, 0, 0], [0, dpi, 0], [0, 0, 1]])
    )
    figure.bbox = matplotlib.transforms.TransformedBbox(bbox, transform)


def renumber_labels_for_display(labels):
    """Scramble the label numbers randomly to make the display more discernable

    The colors of adjacent indices in a color map are less discernable than
    those of far-apart indices. Nearby labels tend to be adjacent or close,
    so a random numbering has more color-distance between labels than a
    straightforward one
    """
    return centrosome.cpmorphology.distance_color_labels(labels)
