""" cpfigure_tools.py - cpfigure tools that do not depend on WX
"""
from cStringIO import StringIO

import matplotlib
import numpy as np
import scipy
from centrosome.cpmorphology import distance_color_labels


def figure_to_image(figure, *args, **kwargs):
    """Convert a figure to a numpy array
    :param kwargs:
    :param args:
    :param figure:
    """
    #
    # Save the figure as a .PNG and then load it using scipy.misc.imread
    #
    fd = StringIO()
    kwargs = kwargs.copy()
    kwargs["format"] = 'png'
    figure.savefig(fd, *args, **kwargs)
    fd.seek(0)
    image = scipy.misc.imread(fd)
    return image[:, :, :3]


def only_display_image(figure, shape):
    """Set up a figure so that the image occupies the entire figure

    figure - a matplotlib figure
    shape - i/j size of the image being displayed
    :param shape:
    :param figure:
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
    bbox = matplotlib.transforms.Bbox(
        np.array([[0.0, 0.0], [width, height]]))
    transform = matplotlib.transforms.Affine2D(
        np.array([[dpi, 0, 0],
                  [0, dpi, 0],
                  [0, 0, 1]]))
    figure.bbox = matplotlib.transforms.TransformedBbox(bbox, transform)


def renumber_labels_for_display(labels):
    """Scramble the label numbers randomly to make the display more discernable

    The colors of adjacent indices in a color map are less discernable than
    those of far-apart indices. Nearby labels tend to be adjacent or close,
    so a random numbering has more color-distance between labels than a
    straightforward one
    :param labels:
    """
    return distance_color_labels(labels)
