import centrosome.cpmorphology
import matplotlib
import matplotlib.collections
import numpy
import scipy.ndimage


class OutlineArtist(matplotlib.collections.LineCollection):
    """An artist that is a plot of the outline around an object

    This class is here so that we can add and remove artists for certain
    outlines.
    """

    def __init__(self, name, labels, *args, **kwargs):
        """Draw outlines for objects

        name - the name of the outline

        labels - a sequence of labels matrices

        kwargs - further arguments for Line2D
        """
        # get_outline_pts has its faults:
        # * it doesn't do holes
        # * it only does one of two disconnected objects
        #
        # We get around the second failing by resegmenting with
        # connected components and combining the original and new segmentation
        #
        lines = []
        for l in labels:
            new_labels, counts = scipy.ndimage.label(l != 0, numpy.ones((3, 3), bool))
            if counts == 0:
                continue
            l = l.astype(numpy.uint64) * counts + new_labels
            unique, idx = numpy.unique(l.flatten(), return_inverse=True)
            if unique[0] == 0:
                my_range = numpy.arange(len(unique))
            else:
                my_range = numpy.arange(1, len(unique))
            idx.shape = l.shape
            pts, offs, counts = centrosome.cpmorphology.get_outline_pts(idx, my_range)
            pts = pts[:, ::-1]  # matplotlib x, y reversed from i,j
            for off, count in zip(offs, counts):
                lines.append(numpy.vstack((pts[off : off + count], pts[off : off + 1])))
        matplotlib.collections.LineCollection.__init__(self, lines, *args, **kwargs)

    def get_outline_name(self):
        return self.__outline_name
