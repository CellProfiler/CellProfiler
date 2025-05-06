import logging
from ndv import ArrayViewer

LOGGER = logging.getLogger(__name__)

def ndv_display(img, ndv_viewer=None):
    if ndv_viewer is None:
        LOGGER.debug("Initializing ndv")
        ndv_viewer = ArrayViewer(img)

        # TODO: ndv - temporary
        #ndv_viewer._async = False
    else:
        LOGGER.debug("Updating ndv data")
        ndv_viewer.data = img

    LOGGER.debug("Rendering image for display in ndv")
    ndv_viewer.show()

    return ndv_viewer
